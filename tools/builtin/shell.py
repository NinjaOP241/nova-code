import asyncio
import fnmatch
import os
import sys
import signal
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from config.config import DEFAULT_EXCLUDE_PATTERNS
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult
from utils.paths import resolve_path
from utils.text import truncate_text

BLOCKED_COMMANDS = {
    "rm -rf /",
    "rm -rf ~",
    "rm -rf /*",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "mkfs",
    "fdisk",
    "parted",
    ":(){ :|:& };:",  # Fork bomb
    "chmod 777 /",
    "chmod -R 777",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init 0",
    "init 6",
}


class ShellParams(BaseModel):
    """
    A pydantic model that defines the arguments accepted by ShellTool.
    """

    command: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
        ),
    ] = Field(
        description="The shell command to execute.",
    )

    timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Timeout in seconds (default: 120)",
    )

    cwd: str | None = Field(
        None,
        description="Working directory for the command",
    )


class ShellTool(Tool):
    """
    Execute system commands, scripts, and CLI tools.

    The agent uses this tool whenever it needs to interact with the
    operating system, run programs, install dependencies, execute tests,
    or perform other command-line operations.

    The command is then executed in a subprocess, and its output and
    exit status are returned to the agent.
    """

    name = "shell"

    kind = ToolKind.SHELL

    description = "Execute a shell command. Use this for running system commands, scripts and CLI tools."

    schema = ShellParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ShellParams(**invocation.params)

        command = params.command.strip()

        # Check for blocked commands to prevent dangerous operations
        for blocked_command in BLOCKED_COMMANDS:
            if blocked_command in command:
                return ToolResult.error_result(
                    f"Command blocked for safety: {params.command}"
                )

        # Determine where the shell command should be run.
        cwd = (
            resolve_path(invocation.cwd, params.cwd)
            if params.cwd is not None
            else invocation.cwd.resolve()
        )

        # A subprocess cannot start in a directory that doesn't exist
        if not cwd.is_dir():
            return ToolResult.error_result(f"Working directory doesn't exist: {cwd}")

        env_for_subprocess = self._build_environment()

        # Choose the appropriate shell command based on the operating system.
        if sys.platform == "win32":
            shell_cmd = ["cmd.exe", "/c", params.command]
        else:
            shell_cmd = ["/bin/bash", "-c", params.command]

        # Run the shell command in a subprocess
        process = await asyncio.create_subprocess_exec(
            *shell_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env_for_subprocess,
            start_new_session=True,
        )

        try:
            # Wait for the command to complete, with a timeout.
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(), timeout=params.timeout
            )
        except asyncio.TimeoutError:
            if sys.platform == "win32":
                process.kill()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)

            # Wait for the terminated process to finish before returning.
            await process.wait()

            return ToolResult.error_result(f"Command timed out after {params.timeout}s")

        stdout = stdout_data.decode("utf-8", errors="replace")
        stderr = stderr_data.decode("utf-8", errors="replace")
        exit_code = process.returncode

        output = ""

        if stdout.strip():
            output += stdout.rstrip()

        if stderr.strip():
            if output:
                output += "\n"
            output += "--- stderr ---\n"
            output += stderr.rstrip()

        if exit_code != 0:
            if output:
                output += "\n"
            output += f"Exit code: {exit_code}"

        output = truncate_text(
            output,
            self.MAX_OUTPUT_TOKENS,
            suffix="\n... [output truncated]",
        )

        return ToolResult(
            success=exit_code == 0,
            output=output,
            error=stderr if exit_code != 0 else None,
            exit_code=exit_code,
        )

    def _build_environment(self) -> dict[str, str]:
        """
        Build the environment variables passed to shell subprocess.

        A child process normally inherits environment variables from
        its parent process.

        However, we may not want to expose sensitive variables such as
        API keys and tokens to commands executed by the agent.

        This method creates a filtered copy of the environment and
        applies any additional variables configured by the user.
        """

        # Start with a copy of the current process's environment variables.
        env_for_subprocess = os.environ.copy()

        # Read the shell environment policy from the application config.
        shell_environment_policy = self.config.shell_environment_policy

        exclude_patterns: list[str] = shell_environment_policy.exclude_patterns.copy()

        if not shell_environment_policy.ignore_default_excludes:
            exclude_patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

        # Remove any environment variables that match the exclusion patterns.
        for pattern in exclude_patterns:
            keys_to_remove = [
                key
                for key in env_for_subprocess.keys()
                if fnmatch.fnmatch(key.upper(), pattern.upper())
            ]

            # Remove the matching variables from the child environment.
            for key in keys_to_remove:
                del env_for_subprocess[key]

        # Add or override environment variables explicitly configured
        # by the user for shell commands.
        #
        # For example:
        #   NODE_ENV=development
        #   LOG_LEVEL=debug
        if shell_environment_policy.set_vars:
            env_for_subprocess.update(shell_environment_policy.set_vars)

        return env_for_subprocess
