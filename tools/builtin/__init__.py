from tools.base import Tool
from tools.builtin.edit_file import EditFileTool
from tools.builtin.list_dir import ListDirTool
from tools.builtin.read_file import ReadFileTool
from tools.builtin.shell import ShellTool
from tools.builtin.write_file import WriteFileTool

__all__ = [
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "ShellTool",
    "ListDirTool",
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
        EditFileTool,
        ShellTool,
        ListDirTool,
    ]
