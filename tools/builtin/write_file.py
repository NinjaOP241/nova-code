from pydantic import BaseModel, Field

from tools.base import FileDiff, Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import ensure_parent_directory, resolve_path


class WriteFileParams(BaseModel):
    """
    A pydantic model that defines the arguments accepted by WriteFileTool.
    """

    path: str = Field(
        description="Path to the file to write (relative to working directory or absolute)"
    )

    content: str = Field(description="Content to write to the file")

    create_directories: bool = Field(
        True, description="Create parent directories if they don't exist"
    )


class WriteFileTool(Tool):
    """
    Create a new file or replace the contents of an existing file.

    The tool performs a complete write. It is intended for creating files
    or replacing all their contents, not for changing only part of a file.
    """

    name = "write_file"

    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "or overwrites if it does. Parent directories are created automatically. "
        "Use this for creating new files or completely replacing file contents. "
        "For partial modifications, use the edit_file tool instead."
    )

    kind = ToolKind.WRITE

    schema = WriteFileParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        # Validate the params
        params = WriteFileParams(**invocation.params)

        # Resolve the supplied path against the tool's working directory.
        # This gives us the actual path we will use to access the file.
        path = resolve_path(invocation.cwd, params.path)

        is_new_file = not path.exists()

        # If the file already exists, we will overwrite it,
        # preserve the old version so the result can include a diff.
        old_content = ""
        if not is_new_file:
            try:
                old_content = path.read_text(encoding="utf-8")
            except:
                pass

        try:
            # Create the parent directories if needed
            if params.create_directories:
                ensure_parent_directory(path)
            elif not path.parent.exists():
                return ToolResult.error_result(
                    f"Parent directory does not exist: {path.parent}"
                )

            # write_text creates the file if needed, or replaces the
            # complete contents of an existing file.
            path.write_text(params.content, encoding="utf-8")

            action = "Created" if is_new_file else "Updated"
            line_count = len(params.content.splitlines())

            return ToolResult.success_result(
                f"{action} {path} {line_count} lines",
                diff=FileDiff(
                    path=path,
                    old_content=old_content,
                    new_content=params.content,
                    is_new_file=is_new_file,
                ),
                metadata={
                    "path": str(path),
                    "is_new_file": is_new_file,
                    "line_count": line_count,
                    "bytes": len(params.content.encode("utf-8")),
                },
            )

        except OSError as e:
            return ToolResult.error_result(f"Failed to write file: {e}")
