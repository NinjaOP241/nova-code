import asyncio
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from agent.agent import Agent
from agent.events import AgentEventType
from agent.types import SubagentDefinition
from config.config import Config
from tools.base import Tool, ToolInvocation, ToolResult


class SubagentParams(BaseModel):
    goal: Annotated[
        str,
        StringConstraints(min_length=1, strip_whitespace=True),
    ] = Field(
        description="The specific task or goal for the subagent to accomplish",
    )


class SubagentTool(Tool):
    schema = SubagentParams

    def __init__(self, config: Config, subagent_definition: SubagentDefinition):
        super().__init__(config)
        self.subagent_definition = subagent_definition

        # The name of the sub-agent tool is derived from the sub-agent's name.
        self.name = f"subagent_{subagent_definition.name}"

        self.description = subagent_definition.description

    def is_mutating(self, params):
        return self.subagent_definition.may_modify_workspace

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = SubagentParams(**invocation.params)

        # Convert the Config pydantic model to a dictionary so that we can override some properties for the sub-agent.
        config_dict = self.config.to_dict()

        # Override the max_turns property for the sub-agent based on its definition.
        config_dict["max_turns"] = self.subagent_definition.max_turns

        # Apply the tool restrictions if this sub-agent has
        # an explicit list of allowed tools.
        if self.subagent_definition.allowed_tools:
            config_dict["allowed_tools"] = self.subagent_definition.allowed_tools

        # Now, create a new Config pydantic model for the sub-agent using the modified dictionary.
        subagent_config = Config(**config_dict)

        # Store the names of all tools invoked by the sub-agent.
        # The main agent only cares about which tools were used and the final summary instead of every tool call result.
        tool_calls = []

        # Stores the final response from the sub-agent after it completes its task.
        final_response = None

        # Stores an error message if the sub-agent encounters an execution error.
        error = None

        # Describes why the sub-agent stopped running.
        #
        # The default value is "goal-completed", meaning the task is expected
        # to finish normally after the agent completes its work.
        terminate_response = "goal-completed"

        try:
            # Wrap the sub-agent's execution in a timeout context to ensure it doesn't run indefinitely.
            async with asyncio.timeout(self.subagent_definition.timeout_seconds):

                # Create a completely separate Agent instance for the sub-agent, using its own configuration.
                #
                # The sub-agent runs its own agentic loop and has its own
                # context, tool registry and execution state.
                async with Agent(
                    config=subagent_config,
                    agent_name=self.subagent_definition.name,
                ) as subagent:
                    async for agent_event in subagent.run(message=params.goal):
                        if agent_event.type == AgentEventType.TOOL_CALL_START:
                            tool_calls.append(agent_event.data.get("tool_name"))
                        elif agent_event.type == AgentEventType.TEXT_COMPLETE:
                            final_response = agent_event.data.get("content")
                        elif agent_event.type == AgentEventType.AGENT_ERROR:
                            error = agent_event.data.get("error")
                            final_response = f"Sub-agent encountered an error: {error}"
                            break
        except TimeoutError:
            terminate_response = "timeout"
            error = f"Exceeded timeout of {self.subagent_definition.timeout_seconds} seconds"
            final_response = (
                f"Sub-agent timed out after "
                f"{self.subagent_definition.timeout_seconds} seconds."
            )
        except Exception as e:
            terminate_response = "error"
            error = str(e)
            final_response = f"Sub-agent failed: {e}"

        # Format the final report for the main agent.
        status = {
            "goal": "COMPLETED",
            "timeout": "TIMED OUT",
            "error": "FAILED",
        }.get(terminate_response, "UNKNOWN")

        result = "\n".join(
            [
                "=== SUBAGENT REPORT ===",
                f"Agent: {self.subagent_definition.name}",
                f"Status: {status}",
                f"Task: {params.goal}",
                f"Tools used ({len(tool_calls)}): {', '.join(tool_calls) if tool_calls else 'None'}",
                *([f"Error: {error}"] if error else []),
                "",
                "=== RESULT ===",
                (final_response or "No response provided.").strip(),
                "",
                "=== END OF SUBAGENT REPORT ===",
            ]
        )

        if error:
            return ToolResult.error_result(result)

        return ToolResult.success_result(result)
