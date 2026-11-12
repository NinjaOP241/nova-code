import json
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from config.loader import get_data_dir
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class MemoryData(TypedDict):
    entries: dict[str, str]


class MemoryParams(BaseModel):
    """
    A pydantic model thatdefines the arguments accepted by MemoryTool.
    """

    action: Literal["get", "set", "delete", "list", "clear"] = Field(
        description=(
            "Operation to perform: "
            "'get' to retrieve a memory by its key, "
            "'set' to store or update a memory, "
            "'delete' to remove a specific memory, "
            "'list' to view all stored memories, or "
            "'clear' to remove all stored memories."
        )
    )

    key: str | None = Field(
        default=None,
        description=(
            "Descriptive identifier for the memory. Required for "
            "'get', 'set', and 'delete'. Use a consistent, meaningful "
            "key so the memory can be retrieved or deleted later."
        ),
    )

    value: str | None = Field(
        default=None,
        description=(
            "Information to store under the specified key. Required "
            "for the 'set' action. Providing an existing key updates "
            "its stored value."
        ),
    )


class MemoryTool(Tool):
    """
    Manages persistent user memory using a local JSON file.

    This tool allows the agent to remember useful information across
    different sessions and application restarts.

    Examples of information worth remembering:
    - User preferences.
    - Coding conventions.
    - Frequently used project context.
    - Important notes that may be useful in future conversations.

    Memories are stored as key-value pairs in user_memory.json.
    """

    name = "memory"

    description = (
        "Manage persistent user memory across sessions and application "
        "restarts. Use 'set' to store or update durable user preferences, "
        "important context, or reusable notes under a descriptive key. "
        "Use 'get' to retrieve a specific memory, 'list' to view all "
        "stored memories, 'delete' to remove a specific memory, and "
        "'clear' to remove all memories. Do not use this tool for "
        "temporary task progress or sensitive credentials."
    )

    kind = ToolKind.MEMORY

    schema = MemoryParams

    def _load_memory(self) -> MemoryData:
        """
        Loads the stored memories from user_memory.json.

        If the file doesn't exist or cannot be read, return an empty
        memory structure instead.

        Expected JSON structure:

        {
            "entries": {
                "preferred_language": "Python",
                "coding_style": "Clean and maintainable code"
            }
        }
        """

        # Get the application's data directory.
        #
        # This is where the application stores its internal data,
        # separately from the user's project/workspace directory.
        data_dir = get_data_dir()

        # Ensure that the data directory exists before trying to
        # access files inside it.
        data_dir.mkdir(parents=True, exist_ok=True)

        # Construct the path to the JSON file that stores user memory.
        memory_file_path = data_dir / "user_memory.json"

        if not memory_file_path.exists():
            return {"entries": {}}

        try:
            content = memory_file_path.read_text(encoding="utf-8")

            # Convert the JSON string into a Python dictionary.
            return json.loads(content)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            return {"entries": {}}

    def _save_memory(self, memory: MemoryData):
        """
        Saves the supplied memory dictionary into user_memory.json.

        This method converts the Python dictionary into JSON
        and writes it to the application's data directory.
        """

        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        memory_file_path = data_dir / "user_memory.json"

        memory_file_path.write_text(
            # Convert the Python dictionary into a JSON string
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = MemoryParams(**invocation.params)

        memory = self._load_memory()
        if params.action == "set":
            if not params.key or not params.value:
                return ToolResult.error_result(
                    "Both 'key' and 'value' are required for the 'set' action."
                )

            memory["entries"][params.key] = params.value

            self._save_memory(memory)

            return ToolResult.success_result(
                f"Memory set: '{params.key}' = '{params.value}'",
                metadata={
                    "memory_count": len(memory["entries"]),
                    "entries": dict(sorted(memory["entries"].items())),
                    "key": params.key,
                    "value": params.value,
                },
            )
        elif params.action == "get":
            if not params.key:
                return ToolResult.error_result(
                    "'key' is required for the 'get' action."
                )

            if params.key not in memory["entries"]:
                return ToolResult.error_result(
                    f"No memory found for key '{params.key}'.",
                    metadata={
                        "found": False,
                    },
                )

            return ToolResult.success_result(
                f"Memory found: '{params.key}' = '{memory['entries'][params.key]}'",
                metadata={
                    "found": True,
                    "key": params.key,
                    "value": memory["entries"][params.key],
                },
            )
        elif params.action == "delete":
            if not params.key:
                return ToolResult.error_result(
                    "'key' is required for the 'delete' action."
                )

            if params.key not in memory["entries"]:
                return ToolResult.error_result(
                    f"No memory found for key '{params.key}'.",
                    metadata={
                        "deleted": False,
                    },
                )

            deleted_value = memory["entries"][params.key]
            del memory["entries"][params.key]
            self._save_memory(memory)

            return ToolResult.success_result(
                f"Memory deleted: '{params.key}'",
                metadata={
                    "deleted": True,
                    "key": params.key,
                    "deleted_value": deleted_value,
                },
            )

        elif params.action == "list":
            entries = memory.get("entries")

            if not entries:
                return ToolResult.success_result(
                    "No memories stored.",
                    metadata={
                        "found": False,
                        "memory_count": 0,
                        "entries": {},
                    },
                )

            lines = ["## Stored Memories: "]

            for key, value in sorted(entries.items()):
                lines.append(f"- **{key}**: {value}")

            return ToolResult.success_result(
                "\n".join(lines),
                metadata={
                    "found": True,
                    "memory_count": len(entries),
                    "entries": dict(sorted(entries.items())),
                },
            )
        elif params.action == "clear":
            count = len(memory.get("entries"))

            if not count:
                return ToolResult.success_result(
                    "No memories to clear.",
                    metadata={
                        "cleared": False,
                        "memory_count": 0,
                        "entries": {},
                    },
                )

            memory["entries"] = {}
            self._save_memory(memory)

            return ToolResult.success_result(
                f"Memory cleared. {count} entries removed.",
                metadata={
                    "cleared": True,
                    "cleared_count": count,
                    "memory_count": 0,
                    "entries": {},
                },
            )
        else:
            return ToolResult.error_result(
                f"Invalid action '{params.action}'. "
                "Valid actions are: 'get', 'set', 'delete', 'list', 'clear'."
            )
