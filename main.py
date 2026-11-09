from pathlib import Path
import sys
import signal
from typing import Any

from agent.agent import Agent
from agent.events import AgentEventType
from client.llm_client import LLMClient
import asyncio
import click

from config.config import Config
from config.loader import load_config
from ui.tui import TUI, get_console

# Create a shared Rich Console instance.
# Both CLI and TUI use this console to display terminal output.
console = get_console()


class CLI:
    """
    Handles the command-line interface of the AI coding agent.

    Responsibilities:
        1. Accept user input.
        2. Distinguish normal messages from slash commands.
        3. Send normal messages to the Agent.
        4. Receive AgentEvents and ask the TUI to display them.
        5. Manage session-related commands such as save, resume,
           checkpoint and restore.

    Important distinction:
        CLI  -> Handles user interaction and application flow.
        TUI  -> Handles how information looks in the terminal.
        Agent -> Handles the actual AI agentic loop and tool execution.
    """

    def __init__(self, config: Config):
        # The Agent is created when we enter run_single() or
        # run_interactive(), not when the CLI object is constructed.
        self.agent: Agent | None = None

        self.config = config

        # The TUI is responsible for rendering messages, tool calls,
        # results, errors, approval prompts, etc.
        self.tui = TUI(config, console)

        # Reference to the event loop running our interactive session.
        # The Ctrl+C handler uses it to schedule cancellation of the
        # current Agent task without stopping the interactive session.
        self._loop: asyncio.AbstractEventLoop | None = None

        # Reference to the task processing the current user message.
        # Ctrl+C cancels this task, not the entire interactive session.
        # The user must use /exit to leave the session.
        self._active_task: asyncio.Task | None = None

    async def run_single(self, message: str) -> str | None:
        """
        Run the agent for one user message and then exit.

        Used when the user supplies a prompt directly through the
        terminal, for example:

            python main.py "What is the capital of India?"

        The async context manager initializes the Agent before use
        and cleans up its resources when the block finishes.
        """

        async with Agent(self.config) as agent:
            # Keep a reference to the active Agent so the other
            # CLI methods can access its session and tool registry.
            self.agent = agent

            # Process the user's message and return the final response.
            return await self._process_message(message)

    def _handle_sigint(self, signum, frame):
        """
        Custom Ctrl+C handler used during interactive sessions.

        asyncio.run() (Python 3.11+) installs its own SIGINT handler that
        cancels the main task, which kills the whole session. We replace it:

            - Agent is busy  -> cancel only the current response.
            - Waiting at the prompt -> raise KeyboardInterrupt so the
              blocking console.input() call is interrupted.
        """

        task = self._active_task

        if task is not None and self._loop is not None and not task.done():
            self._loop.call_soon_threadsafe(task.cancel)
        else:
            raise KeyboardInterrupt

    async def run_interactive(self) -> str | None:
        """
        Start an interactive session.

        Unlike run_single(), this method keeps accepting user input
        until the user enters /exit, /quit, or closes the input stream.

        Example interaction:

            [user]> Explain this project
            [assistant]> This project ...

            [user]> /stats
            Session statistics ...

            [user]> /exit
            Goodbye!
        """
        # Store the event loop currently running this interactive session.
        self._loop = asyncio.get_running_loop()

        # register our custom signal handler
        previous_handler = signal.signal(signal.SIGINT, self._handle_sigint)

        self.tui.print_welcome(
            "NOVA CODE",
            lines=[
                f"model: {self.config.model_name}",
                f"cwd: {self.config.cwd}",
                "commands: /help /config /approval /model /exit",
            ],
        )

        try:
            async with Agent(self.config) as agent:
                self.agent = agent

                # Keep accepting user input until the user decides to exit.
                while True:
                    try:
                        # Read input directly from the terminal.
                        user_input = console.input("\n[user]>[/user] ").strip()

                        # Ignore empty messages instead of sending them to the Agent.
                        if not user_input:
                            continue

                        self._active_task = asyncio.create_task(
                            self._process_message(user_input)
                        )
                        await self._active_task

                    except KeyboardInterrupt:
                        # Ctrl+C interrupts the current input/operation.
                        # We keep the interactive session open and let
                        # the user explicitly exit using /exit.
                        console.print("\n[dim]Use /exit to quit[/dim]")

                    except asyncio.CancelledError:
                        # Ctrl+C cancels the current Agent task, but the session remains active.
                        console.print("\n[dim]Interrupted. Use /exit to quit[/dim]")

                    except EOFError:
                        console.print("\n[dim]Input closed.[/dim]")
                        break

                    finally:
                        self._active_task = None
        finally:
            # restore the original handler when the session ends
            signal.signal(signal.SIGINT, previous_handler)
            self._loop = None

        console.print("\n[dim]Goodbye![/dim]")

    def _get_tool_kind(self, tool_name: str) -> str | None:
        """
        Get the category of a tool using its name.
        """

        if self.agent is None:
            return None

        tool = self.agent.session.tool_registry.get(tool_name)

        if tool is None:
            return None

        # Return the string value of the ToolKind enum.
        # Example: ToolKind.READ.value -> "read".
        return tool.kind.value

    async def _process_message(self, message: str) -> str | None:
        """
        Send a normal user message to the Agent and display its events.

        This method connects the Agent with the TUI.

        The Agent does not directly print its responses.
        Instead, it yields AgentEvents such as:

            TEXT_DELTA
            TEXT_COMPLETE
            TOOL_CALL_START
            TOOL_CALL_COMPLETE
            AGENT_ERROR

        This method listens to those events and tells the TUI
        what to display.

        Returns the final assistant response, if one was produced.
        """

        # There is nothing to process if the Agent hasn't been created.
        if not self.agent:
            return None

        # Tracks whether we have started displaying the assistant's
        # streamed response. We only need to print the Assistant
        # heading once, before the first text chunk.
        assistant_streaming = False

        # Stores the complete response when the Agent tells us
        # that a message has finished.
        final_response: str | None = None

        # Agent.run() is an async generator.
        # It produces events one at a time as the Agent works.
        async for event in self.agent.run(message):

            # ---------------------------------------------------------
            # 1. Assistant text is arriving incrementally from the LLM.
            # ---------------------------------------------------------
            if event.type == AgentEventType.TEXT_DELTA:

                # Get the text fragment from the event.
                # Use an empty string if the content field is missing.
                content = event.data.get("content", "")

                # If this is the first text delta, begin the assistant streaming
                if not assistant_streaming:
                    # Display a UI component indicating that the assistant is responding
                    self.tui.begin_assistant()
                    assistant_streaming = True

                # Display this text fragment immediately.
                # This creates the streaming effect in the terminal.
                self.tui.stream_assistant_delta(content)

            # ---------------------------------------------------------
            # 2. The Agent has completed its assistant response.
            # ---------------------------------------------------------
            elif event.type == AgentEventType.TEXT_COMPLETE:

                # Save the complete response.
                # Useful when run_single() needs to return the final text.
                final_response = event.data.get("content", "")

                if assistant_streaming:
                    self.tui.end_assistant()
                    assistant_streaming = False

            # ---------------------------------------------------------
            # 3. The Agent encountered an error.
            # ---------------------------------------------------------
            elif event.type == AgentEventType.AGENT_ERROR:

                # Extract and display the error information.
                error = event.data.get("error", "Unknown error")
                console.print(f"\n[error]Error: {error}[/error]")

            # ---------------------------------------------------------
            # 4. The Agent is about to execute a tool.
            # ---------------------------------------------------------
            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("tool_name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)

                # Ask the TUI to display the tool name and arguments
                # in a panel with a "running" status.
                self.tui.tool_call_start(
                    event.data.get("tool_call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("arguments", {}),
                )

            # ---------------------------------------------------------
            # 5. The tool has finished executing.
            # ---------------------------------------------------------
            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("tool_name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)

                # Ask the TUI to display the result of the tool call.
                self.tui.tool_call_complete(
                    event.data.get("tool_call_id", ""),
                    tool_name,
                    tool_kind,
                    event.data.get("success", False),
                    event.data.get("output", ""),
                    event.data.get("error"),
                    event.data.get("metadata"),
                    event.data.get("truncated", False),
                )

        # The Agent has finished yielding events for this message.
        # Return the final assistant response, if available.
        return final_response


async def run(messages: list[dict[str, Any]]):
    client = LLMClient()
    async for event in client.chat_completion(messages, True):
        print(event)


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "-c",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Current working directory",
)
def main(
    prompt: str | None,
    cwd: Path | None,
):
    try:
        config = load_config(cwd)
    except Exception as e:
        console.print(f"Configuration error: {e}")

    errors = config.validate()
    if errors:
        for error in errors:
            console.print(f"[error]{error}[/error]")
        sys.exit(1)

    cli = CLI(config)

    # messages = [{"role": "user", "content": prompt}]

    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


main()
