from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import resolve_path


class ListDirParams(BaseModel):
    """
    A pydantic model that defines the arguments accepted by ListDirTool.
    """

    path: Annotated[
        str,
        StringConstraints(
            min_length=1,
            strip_whitespace=True,
        ),
    ] = Field(
        ".",
        description="Directory path to list (default: current directory)",
    )

    include_hidden: bool = Field(
        False,
        description="Whether to include hidden files and directories (default: false)",
    )


class ListDirTool(Tool):
    """
    Lists the immediate contents of a directory.

    Directories are displayed before files, and entries are sorted
    alphabetically within those groups.

    This tool is useful when the agent needs to understand a project's
    folder structure or discover which files are available.
    """

    name = "list_dir"

    description = "List contents of a directory"

    kind = ToolKind.READ

    schema = ListDirParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ListDirParams(**invocation.params)
        dir_path = resolve_path(invocation.cwd, params.path)

        if not dir_path.is_dir():
            return ToolResult.error_result(f"Directory does not exist: {dir_path}")

        try:
            # iterdir() returns an iterator containing the immediate
            # entries inside this directory (files and subdirectories).
            #
            # It does not recursively explore subdirectories.
            #
            # sorted() converts the iterator into a sorted list.
            #
            # The sorting key is a tuple:
            #   1. not p.is_dir() -> directories come before files.
            #   2. p.name.lower() -> entries are sorted alphabetically,
            #      without distinguishing uppercase and lowercase letters.
            items = sorted(
                dir_path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except Exception as e:
            return ToolResult.error_result(f"Error listing directory: {e}")

        # Exclude hidden files and directories if the required
        if not params.include_hidden:
            items = [item for item in items if not item.name.startswith(".")]

        if not items:
            return ToolResult.error_result(
                "Directory is empty",
                metadata={
                    "path": str(dir_path),
                    "entries": 0,
                },
            )

        list_of_entries = []
        for item in items:
            if item.is_dir():
                list_of_entries.append(f"{item.name}/")
            else:
                list_of_entries.append(item.name)

        return ToolResult.success_result(
            "\n".join(list_of_entries),
            metadata={
                "path": str(dir_path),
                "entries": len(list_of_entries),
            },
        )
