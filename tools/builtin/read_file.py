from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import is_binary_file, resolve_path
from utils.text import count_tokens, truncate_text


class ReadFileParams(BaseModel):
    """
    A pydantic model that defines the arguments accepted by ReadFileTool.

    Pydantic uses this model to validate the arguments and generate
    the JSON Schema that we send to the LLM.
    """

    path: str = Field(
        description="Path to the file to read (relative to working directory or absolute)"
    )

    offset: int = Field(
        1,
        ge=1,
        description="Line number to start reading from (1-based). Defaults to 1",
    )

    limit: int | None = Field(
        None,
        ge=1,
        description="Maximum number of lines to read. If not specified, reads entire file.",
    )


class ReadFileTool(Tool):
    name = "read_file"

    # The LLM uses this description to understand what the tool does
    # and when it should call it.
    description = (
        "Read the contents of a text file. Returns the file content with line numbers. "
        "For large files, use offset and limit to read specific portions. "
        "Cannot read binary files (images, executables, etc.)."
    )

    # Reading a file doesn't normally modify anything, so this is
    # classified as a READ tool and doesn't require confirmation by default.
    kind = ToolKind.READ

    # Tell the base Tool class which schema belongs to this tool.
    # The base class uses it for parameter validation and for generating
    # the JSON Schema sent to the LLM.
    schema = ReadFileParams

    # Don't attempt to read files larger than 10 MB.
    MAX_FILE_SIZE = 10 * 1024 * 1024

    # Limit the amount of content returned to the LLM.
    # Even if a file is within the size limit, its contents might be
    # too large to fit comfortably in the model's context.
    MAX_OUTPUT_TOKENS = 25000

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        # Convert the raw arguments into a Pydantic model.
        # This validates the arguments and applies defaults for omitted
        # optional parameters, such as offset=1 and limit=None.
        #
        # validate_params() may already have checked these arguments,
        # but we create the model here because we need its validated
        # values and defaults when executing the tool.
        params = ReadFileParams(**invocation.params)

        # Resolve the supplied path against the tool's working directory.
        # This gives us the actual path we will use to access the file.
        path = resolve_path(invocation.cwd, params.path)

        # Check whether the requested path actually exists.
        if not path.exists():
            return ToolResult.error_result(f"File not found: {path}")

        # The path might exist but point to a directory rather than a file.
        if not path.is_file():
            return ToolResult.error_result(f"Path is not a file: {path}")

        # Get the file size in bytes before reading its contents.
        # This lets us reject very large files without loading them into memory.
        file_size = path.stat().st_size

        if file_size > self.MAX_FILE_SIZE:
            return ToolResult.error_result(
                f"File too large ({file_size / (1024*1024):.1f}MB). "
                f"Maximum is {self.MAX_FILE_SIZE / (1024*1024):.0f}MB."
            )

        # Check if the file is binary. If it is, we cannot read it as text.
        if is_binary_file(path):
            file_size_mb = file_size / (1024 * 1024)

            # Display the size in MB for larger files, otherwise use bytes.
            size_str = (
                f"{file_size_mb:.2f}MB" if file_size_mb >= 1 else f"{file_size} bytes"
            )

            return ToolResult.error_result(
                f"Cannot read binary file: {path.name} ({size_str}) "
                f"This tool only reads text files."
            )

        try:
            # Read the file content. We first try UTF-8 encoding, and if that fails, we fall back to Latin-1 encoding.
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")

            # Split the file into individual lines so we can support
            # reading a particular range instead of always returning
            # the entire file.
            #
            # splitlines() removes the newline characters from each line.
            lines = content.splitlines()
            total_lines = len(lines)

            # File is empty
            if total_lines == 0:
                return ToolResult.success_result(
                    "File is empty.",
                    metadata={
                        "lines": 0,
                    },
                )

            # Determine the range of lines to return based on the offset and limit parameters.
            start_idx = max(0, params.offset - 1)

            if params.limit is not None:
                end_idx = min(start_idx + params.limit, total_lines)
            else:
                end_idx = total_lines

            selected_lines = lines[start_idx:end_idx]

            # Add line numbers to each selected line.
            formatted_lines = []

            for i, line in enumerate(selected_lines, start=start_idx + 1):
                formatted_lines.append(f"{i:6}|{line}")

            # Combine the formatted lines into a single string to return to the LLM.
            output = "\n".join(formatted_lines)

            # Estimate the number of tokens in the output. If it exceeds the maximum allowed, truncate it.
            token_count = count_tokens(output)
            truncated = False
            if token_count > self.MAX_OUTPUT_TOKENS:
                output = truncate_text(
                    output,
                    self.MAX_OUTPUT_TOKENS,
                    suffix=f"\n... [truncated {total_lines} total lines]",
                )
                truncated = True

            metadata_lines = []
            if start_idx > 0 or end_idx < total_lines:
                metadata_lines.append(
                    f"Showing lines {start_idx+1}-{end_idx} of {total_lines}"
                )

            if metadata_lines:
                header = " | ".join(metadata_lines) + "\n\n"
                output = header + output

            return ToolResult.success_result(
                output,
                truncated=truncated,
                metadata={
                    "path": str(path),
                    "total_lines": total_lines,
                    "shown_start": start_idx + 1,
                    "shown_end": end_idx,
                },
            )
        except Exception as e:
            # Catch unexpected errors during file reading or processing
            # and return them as a ToolResult
            return ToolResult.error_result(f"Failed to read file: {e}")
