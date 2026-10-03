"""First-party Prompture tools ready to drop into a :class:`ToolRegistry`.

Each module here exports one or more tool builders that return a
:class:`prompture.ToolDefinition`.  Bring your own
:class:`~prompture.ToolRegistry` and add them with ``registry.add(tool)``.

Example::

    from prompture import ToolRegistry, DeepAgent
    from prompture.tools import PythonSandboxTool

    registry = ToolRegistry()
    registry.add(PythonSandboxTool().to_tool_definition())

    agent = DeepAgent(model="openai/gpt-4o", tools=registry)

The web capability (search, fetch, URL readers, platform search) lives in
:mod:`prompture.tools.web`; ``WebToolkit().register_on(registry)`` adds it all.
"""

from .code_exec import PythonSandboxTool, python_execute_tool
from .web import (
    FetchResult,
    ReadResult,
    SearchResponse,
    WebToolkit,
    read_url,
    register_reader,
    search_platform,
    web_fetch,
)
from .web_search import SearchResult, WebSearchTool, web_search_tool

__all__ = [
    "FetchResult",
    "PythonSandboxTool",
    "ReadResult",
    "SearchResponse",
    "SearchResult",
    "WebSearchTool",
    "WebToolkit",
    "python_execute_tool",
    "read_url",
    "register_reader",
    "search_platform",
    "web_fetch",
    "web_search_tool",
]
