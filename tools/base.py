from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from anyio import Path
from pydantic import BaseModel
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic.json_schema import model_json_schema


class ToolKind(str, Enum):
    """
    Represents the category of a tool.

    We have different kinds of tools in our coding agent:
    - READ: Reading files or inspecting project contents.
    - WRITE: Creating, editing, or deleting files.
    - SHELL: Executing terminal commands.
    - NETWORK: Accessing the internet.
    - MEMORY: Reading or modifying agent memory.
    - MCP: Tools provided through external MCP servers.

    WHY DO WE NEED THIS?

    Instead of checking individual tool names everywhere, we can
    identify tools by their category.

    For example, our approval system can check whether a tool is
    a WRITE or SHELL tool and decide whether user confirmation is needed.
    """

    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"


@dataclass
class ToolResult:
    """
    Standard result returned by every tool after execution.

    Different tools perform different operations, but we want them
    all to return results in the same structure.

    For example:

    read_file:
        ToolResult(success=True, output="File contents...")

    shell:
        ToolResult(success=True, output="Tests passed", exit_code=0)

    write_file:
        ToolResult(success=True, output="File updated", diff=...)

    WHY DO WE NEED THIS?

    Our Agent should not need to understand a different result format
    for every individual tool.

    It can work with ToolResult regardless of which tool was executed.
    """

    # Whether the tool operation completed successfully.
    success: bool

    # Main output produced by the tool.
    # For example, file contents or terminal output.
    output: str

    # Error message when the operation fails.
    # None means no error was reported.
    error: str | None = None

    # Additional structured information about the tool's result.
    #
    # The output field contains the main result, usually as text.
    # Metadata can contain extra details that our application may need.
    #
    # For example, a list_directory tool might return the directory
    # listing in output, while metadata contains the total number
    # of files and directories found.
    metadata: dict[str, Any] = field(default_factory=dict)

    # Whether the output was truncated because it was too large.
    truncated: bool = False

    @classmethod
    def error_result(cls, error: str, output: str = "", **kwargs: Any):
        """
        Convenient way to create a failed ToolResult.

        Instead of manually writing:

            ToolResult(
                success=False,
                output="",
                error="File not found"
            )

        We can simply write:

            ToolResult.error_result("File not found")

        **kwargs allows us to pass additional fields such as metadata,
        exit_code, or diff when needed.
        """

        return cls(
            success=False,
            output=output,
            error=error,
            **kwargs,
        )

    @classmethod
    def success_result(cls, output: str, **kwargs: Any):
        """
        Convenient way to create a successful ToolResult.

        Example:

            ToolResult.success_result("File read successfully")

        We automatically set success=True and error=None.

        Additional result information can be passed through **kwargs.
        """

        return cls(
            success=True,
            output=output,
            error=None,
            **kwargs,
        )

    def to_model_output(self) -> str:
        """
        Convert the ToolResult into text that can be sent back to the LLM.
        """

        if self.success:
            return self.output

        return f"Error: {self.error}\n\nOutput:\n{self.output}"


@dataclass
class ToolInvocation:
    """
    Contains everything needed to execute a tool.

    Instead of passing multiple separate arguments to every tool,
    we group the execution information into one object.

    For example, when executing read_file:

        ToolInvocation(
            params={"path": "main.py"},
            cwd=Path("/my-project")
        )

    params contains the arguments requested by the LLM.

    cwd tells the tool which project directory it should operate in.
    """

    # Arguments required by the particular tool.
    # Example: {"path": "main.py"}
    params: dict[str, Any]

    # Current working directory of the project.
    # One of the usecase: If the agent is trying to go outside the project, we might ask user's permission.
    cwd: Path


@dataclass
class ToolConfirmation:
    """
    Contains information needed to ask the user for permission
    before executing a operation that may require user approval.

    For example, before writing a file, outside of cwd, the agent can show:
    - Which tool wants to run.
    - What parameters it received.
    - What operation it intends to perform.
    - Which files will be affected.
    - What changes will be made.

    The UI can use this information to display a confirmation prompt.
    """

    # Name of the tool requesting permission.
    tool_name: str

    # Parameters with which the tool will be executed.
    params: dict[str, Any]

    # The description of the tool
    description: str


class Tool(ABC):
    """
    An abstract base class for every tool in our coding agent.

    Every individual tool (read_file, write_file, shell, etc.)
    inherits from this class.

    The goal is to define common properties and behaviours in one place
    rather than implementing them separately in every tool.

    For example, a ReadFileTool and WriteFileTool both need:
    - A name and description.
    - A parameter schema.
    - An execute() method.
    - Parameter validation.
    - A way to describe themselves to the LLM.

    We implement shared behaviour here and let individual tools
    provide their own specific implementation.
    """

    # Default values. Individual tools should override these.
    name: str = "base_tool"
    description: str = "Base tool"
    kind: ToolKind = ToolKind.READ

    def __init__(self) -> None:
        pass

    @property
    def schema(self) -> dict[str, Any] | type["BaseModel"]:
        """
        Defines the parameters that the LLM can provide when calling this tool.

        We support two ways of defining a tool's schema:

        1. Pydantic model:
        Used for our built-in tools, where we define and validate
        the expected parameters ourselves.

        2. Dictionary:
        Used for MCP tools, where the external MCP server already
        provides the parameter schema as a JSON Schema dictionary.

        Both formats are supported by to_openai_schema(), which converts
        them into the format expected by the LLM API.
        """

        # The base Tool class doesn't know what arguments an individual
        # tool needs. So, each tool must provide its own schema.
        #
        # For example, the ReadFileTool might define its schema as:
        # class ReadFileSchema(BaseModel):
        #     path: str
        #
        # The ReadFileTool will return this pydantic model class through its schema property:
        # class ReadFileTool(Tool):
        #
        #     @property
        #     def schema(self):
        #         return ReadFileSchema
        raise NotImplementedError("Tool must define schema property or class attribute")

    @abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        """
        Execute the actual operation performed by this tool.

        This method is abstract, meaning every concrete tool must
        implement its own execute() method.

        For example:

        ReadFileTool.execute()  -> reads a file.
        WriteFileTool.execute() -> creates or updates a file.
        ShellTool.execute()     -> executes a terminal command.

        We define the method signature here so every tool follows
        the same execution interface.

        async is used because tool operations may involve I/O,
        such as reading files, running commands, or making network requests.
        """

        pass

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """
        Validate the arguments provided to this tool.

        The LLM might generate incorrect arguments.

        For example, suppose read_file expects:

            {"path": "main.py"}

        But the LLM provides:

            {"file": "main.py"}

        We should detect invalid parameters before executing the tool.

        If validation succeeds, return an empty list.

        If validation fails, return a list of human-readable validation error messages.
        """

        # Get the schema defined by the individual tool.
        #
        # For our ReadFileTool, this gives us:
        # schema = ReadFileSchema
        # This is the pydantic model class that defines the expected parameters.
        schema = self.schema

        # Check whether the schema is a Pydantic model class.
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Try validating the arguments.
            try:
                # This is equivalent to:
                # ReadFileSchema(path="main.py")
                schema(**params)
            except ValidationError as e:

                # Convert Pydantic's validation errors into readable strings.
                errors = []

                for error in e.errors():

                    # Identify which parameter caused the error.
                    # loc can contain nested field names.
                    field = ".".join(str(x) for x in error.get("loc", []))

                    # Get the explanation of what went wrong.
                    msg = error.get("msg", "Validation error")

                    errors.append(f"Parameter '{field}': {msg}")

                return errors

            except Exception as e:
                # Handle unexpected errors.
                return [str(e)]

        # No validation errors found
        return []

    def is_mutating(self, params: dict[str, Any]) -> bool:
        """
        Return whether this tool should be treated as mutating
        and therefore require user approval.

        By default, WRITE, SHELL, NETWORK, and MEMORY tools are
        treated as mutating. Subclasses can override this method
        to make the decision based on the provided parameters.
        """

        return self.kind in {
            ToolKind.WRITE,
            ToolKind.SHELL,
            ToolKind.NETWORK,
            ToolKind.MEMORY,
        }

    def get_confirmation(self, invocation: ToolInvocation):
        """
        Decide whether the tool should request user confirmation.

        If the operation is not considered mutating, return None.

        Otherwise, create a ToolConfirmation object containing the
        information needed by the UI to ask the user for permission.
        """

        # No confirmation is needed for tools that are not considered mutating.
        if not self.is_mutating(invocation.params):
            return None

        # Build the confirmation object
        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=f"Execute {self.name}",
        )

    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert this tool's schema into the function-tool format
        expected by the OpenAI-compatible API.

        WHY DO WE NEED THIS?

        Our Python tools have their own implementation, but the LLM
        doesn't know how to call arbitrary Python functions.

        We need to describe each tool to the LLM:
        - What is the tool called?
        - What does it do?
        - What parameters does it accept?

        The LLM uses this description to decide when and how to call it.

        This method supports both Pydantic model schemas and dictionary schemas.
        """

        # Get the schema defined by the individual tool.
        schema = self.schema

        # CASE 1: The tool uses a Pydantic model as its schema.
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            # Convert the Pydantic model into a JSON Schema dictionary.
            json_schema = model_json_schema(schema, mode="serialization")

            # Build the OpenAI-compatible schema dictionary.
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", []),
                },
            }

        # CASE 2: The tool already provides a dictionary schema.
        if isinstance(schema, dict):

            return {
                "name": self.name,
                "description": self.description,
            }

            # Some dictionary schemas may already contain a "parameters"
            # field, while others may directly represent the parameters schema.
            if "parameters" in schema:
                result["parameters"] = schema["parameters"]
            else:
                result["parameters"] = schema

        # If the schema is neither a Pydantic model nor a dictionary,
        # we cannot convert it into the expected function-tool format.
        raise ValueError(f"Invalid schema type for tool {self.name}: {type(schema)}")
