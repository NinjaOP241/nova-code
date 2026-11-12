from agent.types import AgentName
from tools.base import Tool


def get_subagent_system_prompt(
    agent_name: AgentName,
    tools: list[Tool] | None = None,
) -> str:
    if agent_name == AgentName.CODEBASE_INVESTIGATOR:
        return codebase_investigator_system_prompt()
    elif agent_name == AgentName.CODE_REVIEWER:
        return code_reviewer_system_prompt()
    else:
        raise ValueError(f"Unknown subagent: {agent_name}")


def codebase_investigator_system_prompt() -> str:
    pass


def code_reviewer_system_prompt() -> str:
    pass
