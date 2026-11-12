from pathlib import Path
from typing import Any
import re

from rich.console import Console, Group
from rich.theme import Theme
from rich.rule import Rule
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.syntax import Syntax

from config.config import Config
from utils.paths import display_path_rel_to_cwd
from utils.text import truncate_text

# Rich allows us to define named styles that can be reused throughout the UI.
#
# For example:
#   "error": "bright_red bold"
#
# Now, whenever we print something with style="error", Rich automatically
# applies bright red and bold formatting.
#
# Defining these styles in one place makes the terminal UI consistent and
# makes it easy to change the overall appearance later.
AGENT_THEME = Theme(
    {
        # General styles
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "muted": "grey50",
        "border": "grey35",
        "highlight": "bold cyan",
        # Styles used to distinguish different message roles
        "user": "bright_blue bold",
        "assistant": "bright_white",
        # Styles used for different tool categories
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        # Style used when displaying code or tool arguments
        "code": "white",
    }
)

# We keep one shared Rich Console instance for the application.
#
# The Console is responsible for writing formatted content to the terminal.
# Both the CLI and TUI can use the same instance rather than creating
# multiple independent Console objects.
_console: Console | None = None


def get_console() -> Console:
    """
    Return the shared Rich Console instance.

    We create the Console only when it is first requested. This is called
    lazy initialization.

    Subsequent calls return the same Console object, ensuring that the
    CLI and TUI share the same terminal output configuration.
    """
    global _console

    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)

    return _console


class TUI:
    """
    Handles the visual presentation of the AI coding agent in the terminal.

    The CLI decides what is happening, while this class decides how that
    information should look when displayed to the user.

    For example:
    - begin_assistant() displays the Assistant heading.
    - stream_assistant_delta() displays incoming response text.
    - tool_call_start() displays a tool execution panel.
    - tool_call_complete() displays the tool's result.
    - handle_confirmation() displays an approval prompt.
    """

    def __init__(
        self,
        config: Config,
        console: Console | None = None,
    ) -> None:
        # Use the Console passed by the CLI if one is provided.
        # Otherwise, obtain the application's shared Console instance.
        self.console = console or get_console()

        # Tracks whether an assistant response is currently being displayed.
        #
        # begin_assistant() sets this to True, and end_assistant() sets it
        # back to False.
        #
        # This helps us avoid printing unnecessary blank lines when
        # finishing an assistant response.
        self._assistant_stream_open = False

        # Remember each tool call's arguments so we can display them when that tool call completes.
        self._tool_args_by_tool_call_id: dict[str, dict[str, Any]] = {}

        self.config = config

        # Current working directory
        self.cwd = self.config.cwd

        self._max_block_tokens = 500

    def begin_assistant(self) -> None:
        """
        Start displaying a new assistant response.

        This prints a blank line followed by a visual heading:

        ───────────── Assistant ─────────────

        """

        # Add some spacing before the assistant's response.
        self.console.print()

        # Display a horizontal separator with "Assistant" in the middle.
        self.console.print(Rule(Text("Assistant", style="assistant")))

        # Remember that assistant output has started.
        self._assistant_stream_open = True

    def end_assistant(self) -> None:
        """
        Finish displaying the current assistant response.

        The assistant's text arrives in multiple chunks. After all those
        chunks have been displayed, we need to finish the terminal output
        cleanly.
        """

        # Only print a final newline when an assistant response was opened.
        if self._assistant_stream_open:
            self.console.print()

        self._assistant_stream_open = False

    def stream_assistant_delta(self, contnt: str) -> None:
        """
        Display one small piece of the assistant's response.

        The LLM streams its response incrementally rather than returning
        the entire response at once.

        For example, the response:

            Hello, how are you?

        might arrive as:

            "Hello"
            ", how"
            " are"
            " you?"

        Each piece is passed to this method and printed immediately.
        """

        # end="" prevents Rich from adding a newline after every chunk.
        # This makes all incoming chunks appear as one continuous response.
        #
        # markup=False is important because LLM-generated text might contain
        # strings such as "[bold]hello[/bold]".
        # We want to display those strings literally instead of asking Rich
        # to interpret them as formatting instructions.
        self.console.print(contnt, end="", markup=False)

    def _ordered_args(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> list[tuple[str, Any]]:
        """
        Return tool arguments in a more readable order.

        For example, when displaying read_file arguments, we prefer:

            path
            offset
            limit

        rather than displaying the arguments in an arbitrary order.
        """

        # Define the preferred display order for commonly used tools.
        _PREFERRED_ORDER = {
            "read_file": ["path", "offset", "limit"],
            "write_file": ["path", "create_directories", "content"],
            "edit_file": ["path", "replace_all", "old_string", "new_string"],
            "shell": ["command", "cwd", "timeout"],
            "list_dir": ["path", "include_hidden"],
            "grep": ["path", "pattern", "case_sensitive"],
            "glob": ["path", "pattern"],
            "web_search": ["query", "max_results", "timelimit"],
            "web_fetch": ["url", "timeout"],
            "todos": ["id", "action", "content"],
            "memory": ["action", "key", "value"],
        }

        # Get the preferred order for this tool
        preferred = _PREFERRED_ORDER.get(tool_name, [])

        ordered: list[tuple[str, Any]] = []
        seen = set()

        # First add arguments that are in the preferred order
        for key in preferred:
            if key in arguments:
                ordered.append((key, arguments[key]))
                seen.add(key)

        # Add any remaining arguments that were not part of the preferred
        remaining_keys = set(arguments.keys() - seen)
        ordered.extend((key, arguments[key]) for key in remaining_keys)

        return ordered

    def _render_args_table(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Table:
        """
        Convert tool arguments into a Rich table.

        Instead of displaying arguments as raw JSON, we use a two-column
        table:

            path       src/main.py
            offset     10
            limit      50
        """

        table = Table.grid(padding=(0, 1))

        # First column contains argument names.
        table.add_column(style="muted", justify="right", no_wrap=True)

        # Second column contains argument values.
        table.add_column(style="code", overflow="fold")

        # Display arguments in our preferred order.
        for key, value in self._ordered_args(tool_name, arguments):

            # Large file contents or replacement strings can make a tool
            # panel extremely large.
            #
            # We display a short summary for these arguments instead of
            # printing the entire content inside the argument table.
            if key in {"content", "old_string", "new_string"} and isinstance(
                value, str
            ):
                line_count = len(value.splitlines())
                byte_count = len(value.encode("utf-8", errors="replace"))

                value = f"<{line_count} lines • {byte_count} bytes>"

            table.add_row(key, str(value))

        return table

    def tool_call_start(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_kind: str | None,
        arguments: dict[str, Any],
    ) -> None:
        """
        Display a panel when the Agent starts executing a tool.

        For example:

            ⏺ read_file  #12345678
            path    src/main.py
            offset  1
            limit   100
                                    running

        We display the tool name, its arguments, and its current status.
        """

        # Save the tool call's arguments so we can display them later when the tool call completes.
        self._tool_args_by_tool_call_id[tool_call_id] = arguments

        # Use a different border style depending on the tool category.
        border_style = f"tool.{tool_kind}" if tool_kind else "tool"

        # Build the panel title
        title = Text.assemble(
            ("⏺ ", "muted"),
            (tool_name, "tool"),
            ("  ", "muted"),
            (f"#{tool_call_id[:20]}", "muted"),
        )

        display_args = dict(arguments)
        for key in ("path", "cwd"):
            val = display_args.get(key)

            if isinstance(val, str) and self.cwd:
                display_args[key] = str(display_path_rel_to_cwd(val, self.cwd))

        # Create a visual panel for the tool call
        panel = Panel(
            (
                # Show arguments in a table when arguments are available.
                self._render_args_table(tool_name, display_args)
                # Some tools may not need any arguments.
                # Show a small message instead of an empty table.
                if display_args
                else Text(
                    "(no args)",
                    style="muted",
                )
            ),
            title=title,
            title_align="left",
            subtitle=Text("running", style="muted"),
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )

        self.console.print()
        self.console.print(panel)

    def _extract_read_file_code(
        self,
        text: str,
    ) -> tuple[int, str] | None:
        body = text

        # read_file may include a header describing which lines were shown.
        header_match = re.match(
            r"^Showing lines (\d+)-(\d+) of (\d+)\n\n",
            text,
        )

        # Remove the header from the body if it exists.
        # Example:
        # old_body = Showing lines 1-10 of 223\n\n1|import sys\n2|from typing import Any\n....
        # new_body = 1|import sys\n2|from typing import Any\n....
        if header_match:
            body = text[header_match.end() :]

        # code_lines will contain the lines of code extracted from the body, after removing any line numbers or prefixes.
        code_lines: list[str] = []
        start_line: int | None = None

        for line in body.splitlines():
            # Example of line:
            # 1| def main():
            # 2|    print("Hello, world!")

            # Capture the line number and the actual source code with indentation separately, using regex.
            line_match = re.match(r"^\s*(\d+)\|(.*)$", line)

            if not line_match:
                return None

            line_no = int(line_match.group(1))

            if start_line is None:
                start_line = line_no

            code_lines.append(line_match.group(2))

        if start_line is None:
            return None

        return start_line, "\n".join(code_lines)

    def print_welcome(
        self,
        title: str,
        lines: list[str],
    ) -> None:
        """
        Display the welcome screen when the CLI starts.

        The caller provides the title and the lines to display.
        This method is only responsible for arranging and styling them.
        """

        # Combine the individual lines into one block of text.
        body = "\n".join(lines)

        # Wrap the welcome content inside a styled Rich panel.
        self.console.print(
            Panel(
                Text(body, style="code"),
                title=Text(title, style="highlight"),
                title_align="left",
                border_style="border",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def tool_call_complete(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_kind: str | None,
        success: bool,
        output: str,
        error: str | None,
        metadata: dict[str, Any] | None,
        diff: str | None,
        truncated: bool,
        exit_code: int | None,
    ) -> None:
        # Choose the tool's border style based on its category.
        border_style = f"tool.{tool_kind}" if tool_kind else "tool"

        # Decide which status icon and style to show.
        status_icon = "✓" if success else "✗"
        status_style = "success" if success else "error"

        # Build the panel title showing whether the tool succeeded or failed.
        title = Text.assemble(
            (f"{status_icon} ", status_style),
            (tool_name, "tool"),
            ("  ", "muted"),
            (f"#{tool_call_id[:20]}", "muted"),
        )

        # Retrieve the arguments saved when this tool call started.
        arguments = self._tool_args_by_tool_call_id.get(tool_call_id, {})

        # Store the Rich renderable blocks that will appear in the tool result panel.
        blocks = []

        primary_path = None
        if isinstance(metadata, dict) and isinstance(metadata.get("path"), str):
            primary_path = metadata.get("path")

        # ---------------------------------------------------------------
        # read_file
        # ---------------------------------------------------------------
        if tool_name == "read_file" and success:
            if primary_path:
                # Remove the line-number prefixes from the tool output
                start_line, code = self._extract_read_file_code(output)

                # Metadata tells us which part of the file was actually read.
                shown_start = metadata.get("shown_start")
                shown_end = metadata.get("shown_end")
                total_lines = metadata.get("total_lines")

                # Guess the syntax highlighter for the code.
                lexer = Syntax.guess_lexer(primary_path, code)

                # Construct a small header describing the displayed file.
                header_parts = [display_path_rel_to_cwd(primary_path, self.cwd)]
                header_parts.append(" • ")

                if shown_start and shown_end and total_lines:
                    header_parts.append(
                        f"lines {shown_start}-{shown_end} of {total_lines}"
                    )

                header = "".join(header_parts)

                blocks.append(Text(header, style="muted"))
                blocks.append(
                    Syntax(
                        code,
                        lexer,
                        theme="monokai",
                        line_numbers=True,
                        start_line=start_line,
                        word_wrap=False,
                    )
                )

            else:
                blocks.append(
                    Syntax(
                        output,
                        "text",
                        theme="monokai",
                        word_wrap=False,
                    )
                )

        # ---------------------------------------------------------------
        # write_file / edit_file
        # ---------------------------------------------------------------
        elif tool_name in {"write_file", "edit_file"} and success and diff:
            output_line = output.strip() if output.strip() else "Completed"
            blocks.append(Text(output_line, style="muted"))

            diff_display = truncate_text(
                diff,
                self._max_block_tokens,
                model=self.config.model_name,
            )

            blocks.append(
                Syntax(
                    diff_display,
                    "diff",
                    theme="monokai",
                    word_wrap=True,
                )
            )

        # ---------------------------------------------------------------
        # shell
        # ---------------------------------------------------------------
        elif tool_name == "shell" and success:
            command = arguments.get("command")
            blocks.append(Text(f"$ {command.strip()}", style="muted"))

            if exit_code is not None:
                blocks.append(Text(f"exit_code={exit_code}", style="muted"))

            output_display = truncate_text(
                output,
                self._max_block_tokens,
                model=self.config.model_name,
            )

            blocks.append(
                Syntax(
                    output_display,
                    "text",
                    theme="monokai",
                    word_wrap=True,
                )
            )

        # ---------------------------------------------------------------
        # list_dir
        # ---------------------------------------------------------------
        elif tool_name == "list_dir" and success:
            entries = metadata.get("entries")
            path = metadata.get("path")

            summary = [
                path,
                f"{entries} entries",
            ]

            if summary:
                blocks.append(Text(" • ".join(summary), style="muted"))

            output_display = truncate_text(
                output,
                self._max_block_tokens,
                model=self.config.model_name,
            )

            blocks.append(
                Syntax(
                    output_display,
                    "text",
                    theme="monokai",
                    word_wrap=True,
                )
            )

        # ---------------------------------------------------------------
        # grep
        # ---------------------------------------------------------------
        elif tool_name == "grep" and success:
            matching_lines = metadata.get("matching_lines")
            files_searched = metadata.get("files_searched")

            summary = [
                f"{matching_lines} matching lines",
                f"searched {files_searched} files",
            ]

            if summary:
                blocks.append(Text(" • ".join(summary), style="muted"))

            output_display = truncate_text(
                output,
                self._max_block_tokens,
                model=self.config.model_name,
            )

            blocks.append(
                Syntax(
                    output_display,
                    "text",
                    theme="monokai",
                    word_wrap=True,
                )
            )

        # ---------------------------------------------------------------
        # todos
        # ---------------------------------------------------------------
        elif tool_name == "todos" and success:
            action = arguments.get("action")
            todos_count = metadata.get("todos_count")

            if action == "list" and todos_count > 0:
                todo_table = Table(
                    box=box.SIMPLE,
                    border_style="border",
                    expand=True,
                    show_header=True,
                )
                todo_table.add_column("ID", style="highlight", no_wrap=True)
                todo_table.add_column("Task", style="code")

                todo_rows = 0
                for line in output.splitlines()[1:]:
                    match = re.match(r"- \[([^\]]+)\]:\s*(.*)", line)
                    if match:
                        todo_table.add_row(match.group(1), match.group(2))
                        todo_rows += 1

                if todo_rows:
                    blocks.append(Text("Task list", style="muted"))
                    blocks.append(todo_table)
                else:
                    blocks.append(Text("No todos.", style="muted"))
            else:
                content = arguments.get("content")
                todo_id = arguments.get("id")

                if action == "add":
                    blocks.append(
                        Text(
                            f"Added task: {content or '(no description)'}",
                            style="success",
                        )
                    )
                elif action == "complete":
                    blocks.append(
                        Text(
                            f"Completed task: {todo_id or '(unknown id)'}",
                            style="success",
                        )
                    )
                elif action == "clear":
                    blocks.append(Text("Cleared all tasks", style="success"))
                else:
                    blocks.append(Text(output.strip(), style="muted"))

        # ---------------------------------------------------------------
        # memory
        # ---------------------------------------------------------------
        elif tool_name == "memory" and success:
            action = arguments.get("action")

            if action == "list":
                entries = metadata.get("entries")

                memory_table = Table(
                    box=box.SIMPLE,
                    border_style="border",
                    expand=True,
                    show_header=True,
                )
                memory_table.add_column(
                    "Key",
                    style="highlight",
                    no_wrap=True,
                )
                memory_table.add_column("Value", style="code")

                for key, value in entries.items():
                    memory_table.add_row(str(key), str(value))

                blocks.append(
                    Text(
                        f"{metadata.get('memory_count', len(entries))} "
                        "stored memories",
                        style="muted",
                    )
                )
                blocks.append(memory_table)

            elif action == "get":
                key = metadata.get("key", arguments.get("key", "unknown"))
                value = metadata.get("value")
                if metadata.get("found") and value is not None:
                    blocks.append(
                        Panel(
                            Text(str(value), style="code"),
                            title=Text(str(key), style="highlight"),
                            border_style="border",
                            box=box.ROUNDED,
                            padding=(0, 1),
                        )
                    )
                else:
                    blocks.append(Text(output.strip(), style="muted"))
            elif action == "set":
                blocks.append(
                    Text(
                        f"Saved memory: {metadata.get('key', arguments.get('key', 'unknown'))}",
                        style="success",
                    )
                )
            elif action == "delete":
                blocks.append(
                    Text(
                        f"Deleted memory: {metadata.get('key', arguments.get('key', 'unknown'))}",
                        style="success",
                    )
                )
            elif action == "clear":
                blocks.append(
                    Text(
                        f"Cleared {metadata.get('cleared_count', 0)} memories",
                        style="success",
                    )
                )
            else:
                blocks.append(Text(output.strip(), style="muted"))

        # ---------------------------------------------------------------
        # glob
        # ---------------------------------------------------------------
        elif tool_name == "glob" and success:
            matches = metadata.get("matches")
            path = metadata.get("path")

            summary = [
                f"{matches} matches",
                f"searched {path}",
            ]

            if summary:
                blocks.append(Text(" • ".join(summary), style="muted"))

            output_display = truncate_text(
                output,
                self._max_block_tokens,
                model=self.config.model_name,
            )

            blocks.append(
                Syntax(
                    output_display,
                    "text",
                    theme="monokai",
                    word_wrap=True,
                )
            )

        # ---------------------------------------------------------------
        # web_search
        # ---------------------------------------------------------------
        elif tool_name == "web_search" and success:
            results = metadata.get("results")
            query = arguments.get("query")

            summary = [
                f"query: {query}",
                f"{results} results",
            ]

            if summary:
                blocks.append(Text(" • ".join(summary), style="muted"))

            output_display = truncate_text(
                output,
                self._max_block_tokens,
                model=self.config.model_name,
            )

            blocks.append(
                Syntax(
                    output_display,
                    "text",
                    theme="monokai",
                    word_wrap=True,
                )
            )

        # ---------------------------------------------------------------
        # web_fetch
        # ---------------------------------------------------------------
        elif tool_name == "web_fetch" and success:
            url = arguments.get("url")
            status_code = metadata.get("status_code")
            content_length = metadata.get("content_length")

            summary = [
                f"Status Code: {status_code}",
                f"Content Length: {content_length} bytes",
                f"URL: {url}",
            ]

            if summary:
                blocks.append(Text(" • ".join(summary), style="muted"))

            output_display = truncate_text(
                output,
                self._max_block_tokens,
                model=self.config.model_name,
            )

            blocks.append(
                Syntax(
                    output_display,
                    "text",
                    theme="monokai",
                    word_wrap=True,
                )
            )

        if error and not success:
            blocks.append(Text(error, style="error"))

            output_display = truncate_text(
                output,
                self._max_block_tokens,
                model=self.config.model_name,
            )

            if output_display.strip():
                blocks.append(
                    Syntax(
                        output_display,
                        "text",
                        theme="monokai",
                        word_wrap=True,
                    )
                )
            else:
                blocks.append(Text("(no output)", style="muted"))

        if truncated:
            blocks.append(Text("note: tool output was truncated", style="warning"))

        panel = Panel(
            Group(*blocks),
            title=title,
            title_align="left",
            subtitle=Text(
                "done" if success else "failed",
                style=status_style,
            ),
            subtitle_align="right",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(1, 2),
        )

        # Display the completed tool panel.
        self.console.print()
        self.console.print(panel)
