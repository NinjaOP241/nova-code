from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from tools.base import FileDiff, Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import resolve_path


class EditParams(BaseModel):
    """
    A pydantic model that defines the arguments accepted by EditFileTool.
    """

    path: str = Field(
        min_length=1,
        description="Path to the existing file to edit (relative to working directory or absolute path)",
    )

    old_string: str = Field(
        min_length=1,
        description="The exact existing text to replace. Must match exactly, including whitespace, indentation, and line breaks.",
    )

    new_string: str = Field(
        description="The text to replace old_string with. Can be empty to delete text",
    )

    replace_all: bool = Field(
        default=False,
        description="Replace all occurrences of old_string (default: false)",
    )

    @model_validator(mode="after")
    def validate_strings_are_different(self) -> Self:
        """Reject edits that would leave the file unchanged."""

        if self.old_string == self.new_string:
            raise ValueError(
                "old_string and new_string must be different. "
                "No changes would be made."
            )

        return self


class EditFileTool(Tool):
    """
    Make precise modifications to an existing file.

    The tool finds an exact piece of text and replaces it with new text.
    Everything outside the matched text remains unchanged.

    Use write_file for creating files or completely replacing file contents.
    """

    name = "edit_file"

    description = (
        "Edit an existing file by replacing an exact piece of text. "
        "The old_string must match exactly, including whitespace, "
        "indentation, and line breaks. It must be unique in the file "
        "unless replace_all is true. Use this for precise, surgical edits. "
        "Use write_file tool for creating new files or completely rewriting files."
    )

    kind = ToolKind.WRITE

    schema = EditParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """
        Apply the requested modification to the existing file.

        Flow:
        1. Validate arguments and resolve the file path.
        2. Ensure the target is an existing regular file.
        3. Read the current content.
        4. Find and validate the exact match.
        5. Calculate the replacement content.
        6. Write the modified content.
        7. Return a diff and useful metadata.
        """

        # Validate the arguments before touching the filesystem.
        params = EditParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        # Ensure the target is an existing regular file.
        if not path.is_file():
            return ToolResult.error_result(
                f"Cannot edit file: {path}. "
                "The file does not exist or is not a regular file. "
                "Use write_file tool to create a new file."
            )

        old_content = ""
        try:
            old_content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as e:
            return ToolResult.error_result(f"Failed to read file {path}: {e}")

        try:
            # Find the requested text and calculate the new file content.
            new_content, replace_count = self._calculate_replacements(
                path,
                old_content,
                params,
            )

        except ValueError as e:
            return ToolResult.error_result(str(e))

        try:
            # Write the modified content back to the file.
            path.write_text(new_content, encoding="utf-8")
        except (OSError, UnicodeError) as e:
            return ToolResult.error_result(f"Failed to write file {path}: {e}")

        # Calculate the change in the total number of lines.
        old_lines_count = len(old_content.splitlines())
        new_lines_count = len(new_content.splitlines())
        line_diff = new_lines_count - old_lines_count

        diff_msg = ""
        if line_diff > 0:
            diff_msg = f" (+{line_diff} lines)"
        elif line_diff < 0:
            diff_msg = f" ({line_diff} lines)"

        return ToolResult.success_result(
            f"Edited {path}: replaced {replace_count} {('occurrence' if replace_count == 1 else 'occurrences')} {diff_msg}",
            diff=FileDiff(
                path=path,
                old_content=old_content,
                new_content=new_content,
            ),
            metadata={
                "path": str(path),
                "replaced_count": replace_count,
                "line_diff": line_diff,
            },
        )

    def _calculate_replacements(
        self,
        path: Path,
        old_content: str,
        params: EditParams,
    ) -> tuple[str, int]:
        """
        Validate the requested match and calculate the resulting content.

        It does not modify the actual file.
        """

        # Count occurrences of the old_string in the file.
        occurrence_count = old_content.count(params.old_string)

        if occurrence_count == 0:
            raise ValueError(
                f"Could not find the exact text to replace in {path}. "
                "No changes were made. The old_string must match the file "
                "exactly, including whitespace, indentation, and line breaks. "
                "Use the read_file tool to read the latest file contents, "
                "then retry with an exact, sufficiently unique snippet."
            )

        # Avoid changing multiple locations accidentally.
        # The LLM must explicitly request replace_all=True for that.
        if occurrence_count > 1 and not params.replace_all:
            raise ValueError(
                f"old_string found {occurrence_count} times in {path}. "
                "Provide more surrounding context to make the match unique, "
                "or set replace_all=true to replace every occurrence."
            )

        replace_count = occurrence_count
        new_content = old_content.replace(params.old_string, params.new_string)

        return new_content, replace_count
