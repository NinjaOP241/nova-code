from tools.base import Tool
from tools.builtin.read_file import ReadFileTool
from tools.builtin.write_file import WriteFileTool

__all__ = [
    "ReadFileTool",
]


def get_all_builtin_tools() -> list[type[Tool]]:
    """
    Return the classes of all built-in tools.

    The tool registry uses these classes to create and register
    tool instances when the application starts.
    """
    return [
        ReadFileTool,
        WriteFileTool,
    ]
