import os
from pathlib import Path
import re
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import is_binary_file, resolve_path


class GrepParams(BaseModel):
    """
    A pydantic model that defines the arguments accepted by GrepTool.
    """

    pattern: str = Field(
        min_length=1,
        description="Regular expression pattern to search for",
    )

    path: Annotated[
        str,
        StringConstraints(
            min_length=1,
            strip_whitespace=True,
        ),
    ] = Field(
        ".",
        description="File or directory to search in (default: current directory)",
    )

    case_sensitive: bool = Field(
        True,
        description="Case-sensitive search (default: true)",
    )

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, pattern: str) -> str:
        if not pattern.strip():
            raise ValueError(
                "pattern must contain at least one non-whitespace character"
            )

        return pattern


class GrepTool(Tool):
    """
    Searches for a regular expression pattern inside file contents.

    The search can target either a single file or an entire directory.

    For every matching line, the tool returns:
    - The file containing the match.
    - The line number where the match was found.
    - The actual matching line.

    This helps the agent locate relevant code without having to
    read every file individually.
    """

    name = "grep"

    description = (
        "Search for a regex pattern in file contents. "
        "Returns matching lines with file paths and line numbers."
    )

    kind = ToolKind.READ

    schema = GrepParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = GrepParams(**invocation.params)
        search_path = resolve_path(invocation.cwd, params.path)

        if not search_path.exists():
            return ToolResult.error_result(f"Path does not exist: {search_path}")

        try:
            # Compile the regex string into a pattern object.
            # This also validates the regex syntax.
            flags = 0 if params.case_sensitive else re.IGNORECASE
            pattern = re.compile(params.pattern, flags)
        except re.error as e:
            return ToolResult.error_result(f"Invalid regex pattern: {e}")

        # Determine which files need to be searched.
        #
        # There are two possible situations:
        #
        # 1. The requested path is a directory:
        #    Discover the files inside it using _find_files().
        #
        # 2. The requested path is a single file:
        #    Search only that particular file.
        truncated = False
        if search_path.is_dir():
            files, truncated = self._find_files(search_path)
        else:
            files = [] if is_binary_file(search_path) else [search_path]

        # Stores the final formatted output that will be returned
        # to the LLM.
        #
        # Example:
        #
        # === src/main.py ===
        # 10: def execute():
        # 25: async def execute()
        #
        # === src/utils.py ===
        # 5: def execute():
        output_lines = []

        # Counts the number of matching lines across all searched files.
        matching_lines = 0

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            lines = content.splitlines()

            # Tracks whether we have found at least one matching line
            # in the current file.
            file_has_matching_pattern = False

            for line_number, line in enumerate(lines, start=1):
                if pattern.search(line):
                    matching_lines += 1

                    # Found the first matching line in this file, so we add the file header.
                    if not file_has_matching_pattern:
                        file_has_matching_pattern = True

                        try:
                            display_path = file_path.relative_to(invocation.cwd)
                        except ValueError:
                            display_path = file_path

                        output_lines.append(f"=== {display_path} ===")

                    # Add the matching line to the output.
                    output_lines.append(f"{line_number}: {line}")

            # Add a blank line after the last matching line of the current file
            if file_has_matching_pattern:
                output_lines.append("")

        # Create the final output string to display to the user.
        output = (
            "\n".join(output_lines)
            if output_lines
            else f"No matches found for pattern '{params.pattern}' in {search_path}"
        )

        if truncated:
            output += "\n... (truncated, searched first 500 files only)"

        return ToolResult.success_result(
            output,
            metadata={
                "path": str(search_path),
                "matching_lines": matching_lines,
                "files_searched": len(files),
            },
        )

    def _find_files(self, search_path: Path) -> tuple[list[Path], bool]:
        """
        Recursively finds all files in the given directory.
        """

        IGNORED_DIRS = {
            "node_modules",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
        }

        files = []

        # os.walk() recursively explores a directory.
        #
        # For every directory it visits, it provides three values:
        #
        # root:
        #   The current directory being explored.
        #
        # dirs:
        #   A list of subdirectory names inside root.
        #
        # filenames:
        #   A list of file names inside root.
        for root, dirs, filenames in os.walk(search_path):

            # Filter out ignored directories so that os.walk() doesn't explore them.
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            # Iterate over the files in the current directory.
            for filename in filenames:
                if filename.startswith("."):
                    continue

                file_path = Path(root) / filename

                if not is_binary_file(file_path):
                    files.append(file_path)

                    if len(files) >= 500:
                        return files, True

        return files, False
