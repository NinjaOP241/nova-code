from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import resolve_path


class GlobParams(BaseModel):
    """
    A Pydantic model that defines the arguments accepted by GlobTool.
    """

    pattern: Annotated[
        str,
        StringConstraints(
            min_length=1,
            strip_whitespace=True,
        ),
    ] = Field(
        description="Glob pattern to match",
    )

    path: Annotated[
        str,
        StringConstraints(
            min_length=1,
            strip_whitespace=True,
        ),
    ] = Field(
        ".",
        description="Directory to search in (default: current directory)",
    )


class GlobTool(Tool):
    """
    Finds files matching a specified glob pattern inside a directory.

    Supports patterns such as:
    - *.py       : Match Python files in the current directory.
    - **/*.py    : Recursively match Python files in all subdirectories.
    - *.json     : Match JSON files in the current directory.
    - src/**/*.ts: Match TypeScript files inside src and its subdirectories.

    The tool returns the paths of matching files.
    """

    name = "glob"

    description = (
        "Find files matching a glob pattern. Supports ** for recursive matching."
    )

    kind = ToolKind.READ

    schema = GlobParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = GlobParams(**invocation.params)

        search_path = resolve_path(invocation.cwd, params.path)

        if not search_path.is_dir():
            return ToolResult.error_result(f"Directory does not exist: {search_path}")

        try:
            # Use the glob method to find files matching the glob pattern.
            matched_file_paths = list(search_path.glob(params.pattern))

            # Path.glob() can return both files and directories.
            # Keep only regular files in the results.
            matched_file_paths = [
                file_path for file_path in matched_file_paths if file_path.is_file()
            ]

            matches = len(matched_file_paths)
            truncated = len(matched_file_paths) > 1000
            matched_file_paths = matched_file_paths[:1000]
        except Exception as e:
            return ToolResult.error_result(f"Error searching: {e}")

        output_lines = []
        for file_path in matched_file_paths:
            try:
                display_path = file_path.relative_to(invocation.cwd)
            except ValueError:
                display_path = file_path

            output_lines.append(str(display_path))

        if truncated:
            output_lines.append("... (truncated, showing first 1000 matches)")

        return ToolResult.success_result(
            "\n".join(output_lines),
            metadata={
                "path": str(search_path),
                "matches": matches,
            },
        )
