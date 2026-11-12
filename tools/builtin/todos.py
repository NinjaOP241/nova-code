from typing import Literal
import uuid

from pydantic import BaseModel, Field

from config.config import Config
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class TodosParam(BaseModel):
    action: Literal["add", "complete", "list", "clear"] = Field(
        description=(
            "Operation to perform: "
            "'add' to create a todo, "
            "'complete' to remove a finished todo, "
            "'list' to view all todos, or "
            "'clear' to remove all todos."
        )
    )

    id: str | None = Field(
        default=None,
        description=(
            "ID of an existing todo. Required for the 'complete' action. "
            "Use the ID returned when the todo was added or displayed by 'list'."
        ),
    )

    content: str | None = Field(
        default=None,
        description=(
            "Description of the task to add. Required for the 'add' action. "
            "Provide a clear and actionable task description."
        ),
    )


class TodosTool(Tool):
    name = "todos"

    description = (
        "Manage the current session's task list for tracking progress "
        "on multi-step tasks. Use 'add' to record tasks, 'list' to view "
        "them, 'complete' to remove a finished task using its ID, and "
        "'clear' to remove all tasks."
    )

    kind = ToolKind.MEMORY

    schema = TodosParam

    def __init__(self, config: Config):
        super().__init__(config)
        self._todos: dict[str, str] = {}

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = TodosParam(**invocation.params)

        if params.action == "add":
            if not params.content:
                return ToolResult.error_result(
                    "'content' is required for the 'add' action."
                )
            todo_id = str(uuid.uuid4())[:8]
            self._todos[todo_id] = params.content
            return ToolResult.success_result(
                f"Added todo [{todo_id}]: {params.content}",
                metadata={"todos_count": len(self._todos)},
            )
        elif params.action == "complete":
            if not params.id:
                return ToolResult.error_result(
                    "'id' is required for the 'complete' action."
                )

            if not params.id in self._todos:
                return ToolResult.error_result(f"Todo not found: {params.id}")

            content = self._todos.pop(params.id)
            return ToolResult.success_result(
                f"Completed todo [{params.id}]: {content}",
                metadata={"todos_count": len(self._todos)},
            )
        elif params.action == "list":
            if not self._todos:
                return ToolResult.success_result(
                    "No todos.",
                    metadata={"todos_count": 0},
                )

            lines = ["Todos:"]
            for todo_id, content in self._todos.items():
                lines.append(f"- [{todo_id}]: {content}")

            return ToolResult.success_result(
                "\n".join(lines),
                metadata={"todos_count": len(self._todos)},
            )
        else:  # params.action == "clear"
            count = len(self._todos)
            self._todos.clear()
            return ToolResult.success_result(
                f"Cleared {count} todos.",
                metadata={"todos_count": 0},
            )
