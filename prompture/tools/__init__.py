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

from .cli import (
    CLIArg,
    CLICommand,
    CLITool,
    builtin_cli_tools,
    gh_tool,
    load_cli_tools,
    resolve_cli_tools,
    yt_dlp_tool,
)
from .code_exec import PythonSandboxTool, python_execute_tool
from .packs import PackTool, ToolPack, get_pack, list_packs, register_pack, resolve_pack_tools
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
    "CLIArg",
    "CLICommand",
    "CLITool",
    "FetchResult",
    "PackTool",
    "PythonSandboxTool",
    "ReadResult",
    "SearchResponse",
    "SearchResult",
    "ToolPack",
    "WebSearchTool",
    "WebToolkit",
    "builtin_cli_tools",
    "get_pack",
    "gh_tool",
    "list_packs",
    "load_cli_tools",
    "python_execute_tool",
    "read_url",
    "register_pack",
    "register_reader",
    "resolve_cli_tools",
    "resolve_pack_tools",
    "search_platform",
    "web_fetch",
    "web_search_tool",
    "yt_dlp_tool",
]
