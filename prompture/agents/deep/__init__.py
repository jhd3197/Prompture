"""Built-in tool factories and middleware for :class:`DeepAgent`.

Each module here exposes one or more factories that take a
:class:`DeepAgentState` (and any other dependencies they need) and
return :class:`ToolDefinition` objects closing over that state. The
:class:`DeepAgent` constructor wires the factories together.
"""

from .planner import make_write_todos_tool
from .subagents import make_task_tool
from .summarizer import SummarizationMiddleware
from .vfs import make_vfs_tools

__all__ = [
    "SummarizationMiddleware",
    "make_task_tool",
    "make_vfs_tools",
    "make_write_todos_tool",
]
