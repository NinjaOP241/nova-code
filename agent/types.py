from dataclasses import dataclass
from enum import Enum


class AgentName(str, Enum):
    MAIN_AGENT = "main_agent"
    CODEBASE_INVESTIGATOR = "codebase_investigator"
    CODE_REVIEWER = "code_reviewer"


@dataclass
class SubagentDefinition:
    # The name of the sub-agent.
    # Example:
    #   "codebase_investigator"
    #   "code_reviewer"
    name: AgentName

    # Description shown to the MAIN agent.
    # Helps it decide when to delegate work to this specialist.
    description: str

    # Controls which tools the sub-agent is allowed to access.
    allowed_tools: list[str] | None = None

    # Whether the sub-agent is allowed to modify the workspace.
    may_modify_workspace: bool = False

    # Maximum number of agentic loop iterations the sub-agent
    # is allowed to execute for one delegated task.
    max_turns: int = 20

    # Maximum amount of time the sub-agent is allowed to run,
    # expressed in seconds.
    timeout_seconds: float = 600

    def __post_init__(self) -> None:
        if self.name == AgentName.MAIN_AGENT:
            raise ValueError("SubagentDefinition cannot have 'main_agent' as its name.")
