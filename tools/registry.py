import logging
from pathlib import Path
from typing import Any
from tools.base import Tool, ToolInvocation, ToolResult
from tools.builtin import get_all_builtin_tools

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central place for managing all tools available to our AI coding agent.

    The registry maintains two collections:

    1. Regular tools implemented by our application (read_file, write_file, shell, etc.).
    2. Tools provided by external MCP servers (GitHub, PostgreSQL, etc.).

    It also provides methods to:
    - Register and remove tools.
    - Find a tool using its name.
    - Get all available tools.
    - Convert tools into the schema format understood by the LLM.
    - Validate, approve and execute tool calls.

    The Agent doesn't need to manage the internal dictionaries or
    the entire tool execution process itself. It can delegate those
    responsibilities to this registry.
    """

    def __init__(self):
        # Stores regular tools, using their names as dictionary keys.
        #
        # Example:
        # {
        #     "read_file": ReadFileTool instance,
        #     "write_file": WriteFileTool instance,
        #     "shell": ShellTool instance
        # }
        #
        # We store actual Python objects here because we need their
        # execute() methods when the LLM requests a tool.
        self._tools: dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        """
        Add a regular tool to the registry.

        The tool's name is used as the dictionary key, so we can
        retrieve the tool later using that name.
        """

        # Check whether a tool with the same name is already registered.
        if tool.name in self._tools:
            # Registering another tool with the same name will replace
            # the previously registered tool.
            logger.warning(f"Overwriting existing tool: {tool.name}")

        # Store the actual tool instance in the dictionary.
        #
        # Example:
        # self._tools["read_file"] = ReadFileTool(config)
        self._tools[tool.name] = tool

        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, tool_name: str) -> bool:
        """
        Remove a regular tool from the registry.

        Returns:
            True  - if the tool was found and removed.
            False - if the tool was not found.
        """

        # Check whether the requested tool exists in our regular tools.
        if tool_name in self._tools:
            del self._tools[tool_name]

            # Remove the tool from the dictionary.
            # It will no longer be available through this registry.
            return True

        # The requested tool was not found.
        return False

    def get(self, tool_name: str) -> Tool | None:
        """
        Find and return an actual tool instance using its name.

        The tool could be either a regular tool or an MCP tool.

        Returns:
            Tool instance - if a matching tool is found.
            None         - if no matching tool exists.
        """

        if tool_name in self._tools:
            return self._tools[tool_name]

        return None

    def get_tools(self) -> list[Tool]:
        """
        Collect all available tool instances into a single list.

        Our tools are stored in two separate dictionaries, but sometimes
        we need to work with all of them together.

        This method combines regular tools and MCP tools into one list.

        It also applies the allowed_tools configuration, if provided,
        to determine which tools should actually be available to the Agent.
        """

        # Start with an empty list where we will collect tool instances.
        tools: list[Tool] = []

        # Add all regular application tools.
        for tool in self._tools.values():
            tools.append(tool)

        return tools

    def get_schemas(self) -> list[dict[str, Any]]:
        """
        Convert all available tools into OpenAI-compatible schemas.

        The LLM doesn't receive our actual Python tool objects.
        It only needs descriptions of the available tools and their parameters.

        Every Tool instance inherits to_openai_schema() from the base Tool class.
        We call that method to generate the schema for each available tool.

        The returned list can then be passed to our LLM client.
        """

        return [tool.to_openai_schema() for tool in self.get_tools()]

    async def invoke(
        self,
        tool_name: str,
        params: dict[str, Any],
        cwd: Path,
    ) -> ToolResult:
        """
        Handle the complete process of invoking a tool.

        The Agent calls this method after the LLM requests a tool.

        This method takes care of:
        1. Finding the requested tool.
        2. Validating the supplied parameters.
        3. Triggering the before-tool hooks.
        4. Checking whether the operation requires approval.
        5. Executing the tool.
        6. Handling unexpected execution errors.
        7. Triggering the after-tool hooks.

        Instead of making the Agent manage all these steps itself,
        we centralize the tool invocation process here.
        """

        # ---------------------------------------------------------
        # STEP 1: Find the requested tool.
        # ---------------------------------------------------------

        # The LLM provides the tool's name, not the actual Python object.
        # Retrieve the corresponding tool instance from the registry.
        tool = self.get(tool_name)

        # If no tool with the requested name exists, we cannot execute it.
        if tool is None:
            result = ToolResult.error_result(
                f"Unknown tool: {tool_name}",
                metadata={"tool_name": tool_name},
            )

            return result

        # ---------------------------------------------------------
        # STEP 2: Validate the arguments supplied by the LLM.
        # ---------------------------------------------------------

        # Each tool defines its own parameter schema.
        # validate_params() checks the supplied arguments against that schema.
        validation_errors = tool.validate_params(params)

        # A non-empty list means that one or more parameters are invalid.
        if validation_errors:
            result = ToolResult.error_result(
                f"Invalid parameters: {'; '.join(validation_errors)}",
                metadata={
                    "tool_name": tool_name,
                    "validation_errors": validation_errors,
                },
            )

            return result

        # ---------------------------------------------------------
        # STEP 3: Actually execute the tool.
        # ---------------------------------------------------------
        invocation = ToolInvocation(
            params=params,
            cwd=cwd,
        )

        result = None
        try:
            result = await tool.execute(invocation)
        except Exception as e:
            logger.exception(f"Tool {tool_name} raised unexpected error")
            result = ToolResult.error_result(
                f"Internal error: {str(e)}",
                metadata={
                    "tool_name": tool_name,
                },
            )

        return result


def create_default_registry() -> ToolRegistry:
    """
    Create and populate a registry with the application's default tools.

    Called by Session.__init__() to initialize the session's tool registry.

    It creates the registry, instantiates all built-in tools,
    creates the default subagent tools, and registers them.
    """

    # Create a new registry with the application configuration.
    registry = ToolRegistry()

    # Get all built-in tool classes and register an instance of each.
    #
    # get_all_builtin_tools() returns classes, not instances.
    # We create each instance by passing the application config.
    #
    # Example:
    # ReadFileTool(config)
    # WriteFileTool(config)
    # ShellTool(config)
    for tool_class in get_all_builtin_tools():
        registry.register_tool(tool_class())

    return registry
