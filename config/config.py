import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    """Settings that control which LLM model we use and how it behaves."""

    # The name of the LLM model to use.
    name: str = "cohere/north-mini-code:free"

    # Controls the randomness of the LLM's responses.
    # Lower values produce more predictable output;
    # higher values produce more varied and creative output.
    temperature: float = Field(default=1, ge=0.0, le=2.0)

    # The maximum context size we expect the model to support.
    context_window: int = 256_000


DEFAULT_EXCLUDE_PATTERNS = [
    "*KEY*",
    "*TOKEN*",
    "*SECRET*",
]


class ShellEnvironmentPolicy(BaseModel):
    """
    Controls which environment variables are passed to shell subprocesses.
    """

    ignore_default_excludes: bool = False

    # Additional patterns supplied by the user.
    exclude_patterns: list[str] = Field(default_factory=list)

    set_vars: dict[str, str] = Field(default_factory=dict)


class Config(BaseModel):
    """
    Central configuration for the AI agent.

    Groups all agent settings—LLM parameters, working directory, tool
    permissions, approval policies, hooks, MCP servers, and instructions
    into one validated object that can be accessed throughout the application.
    """

    model: ModelConfig = Field(default_factory=ModelConfig)

    cwd: Path = Field(default_factory=Path.cwd)

    shell_environment_policy: ShellEnvironmentPolicy = Field(
        default_factory=ShellEnvironmentPolicy,
    )

    # Maximum number of iterations the agentic loop can execute
    # while processing a single user request.
    max_turns: int = 100

    developer_instructions: str | None = None
    user_instructions: str | None = None

    # For debugging purposes, to print logger messages to the console.
    debug: bool = False

    @property
    def api_key(self) -> str | None:
        return os.environ.get("API_KEY")

    @property
    def base_url(self) -> str | None:
        return os.environ.get("BASE_URL")

    # Convenience properties let other parts of the application access
    # commonly used nested settings without writing config.model.name.
    #
    # For example:
    #     config.model.name
    # can also be accessed as:
    #     config.model_name
    @property
    def model_name(self) -> str:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str) -> None:
        # Keep the nested ModelConfig as the single source of truth.
        self.model.name = value

    @property
    def temperature(self) -> float:
        return self.model.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.model.temperature = value

    def validate(self) -> list[str]:
        errors: list[str] = []

        if not self.api_key:
            errors.append("No API key found. Set API_KEY in your .env file")

        if not self.cwd.exists():
            errors.append(f"Working directory does not exist: {self.cwd}")

        return errors
