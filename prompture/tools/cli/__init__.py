"""Wrap command-line programs as safe, read-only agent tools.

* :class:`CLITool` / :class:`CLICommand` / :class:`CLIArg` — declare a program,
  its allowlisted commands and their arguments; argv is built without a shell.
* :func:`gh_tool`, :func:`yt_dlp_tool` — shipped definitions.
* :func:`load_cli_tools` — user definitions from ``.prompture/tools.yaml``.
* :func:`resolve_cli_tools` — the ``cli:`` namespace for
  ``Agent(..., tools=["cli:gh"])`` (also ``cli:yt-dlp``, ``cli:all`` and
  user-defined names). Tools whose binary is missing or broken resolve to
  nothing (with a warning naming the fix).
"""

from __future__ import annotations

import logging

from ...agents.tools_schema import ToolDefinition
from .adapter import CLIArg, CLICommand, CLIResult, CLITool, CLIToolError, run_process
from .builtin import builtin_cli_tools, gh_tool, vtt_to_text, yt_dlp_tool
from .config import config_errors, config_paths, load_cli_tools, parse_cli_tools

logger = logging.getLogger("prompture.tools.cli")

_ALIASES = {"yt_dlp": "yt-dlp", "ytdlp": "yt-dlp", "youtube-dl": "yt-dlp", "github": "gh"}


def all_cli_tools(*, include_config: bool = True) -> dict[str, CLITool]:
    """Shipped tools overlaid with user-defined ones (user definitions win)."""
    tools = builtin_cli_tools()
    if include_config:
        for tool in load_cli_tools():
            tools[tool.name] = tool
    return tools


def resolve_cli_tools(name: str) -> list[ToolDefinition]:
    """Resolve ``cli:<name>`` into tool definitions.

    ``all`` (or ``*``) returns every active tool. A known but inactive tool
    (binary missing or broken) resolves to an empty list and logs the fix.

    Raises:
        ValueError: *name* is not a known CLI tool.
    """
    key = (name or "").strip()
    tools = all_cli_tools()
    lowered = {k.lower(): v for k, v in tools.items()}
    if key.lower() in ("all", "*"):
        return [td for tool in tools.values() if tool.is_active() for td in tool.to_tool_definitions()]
    tool = lowered.get(key.lower()) or lowered.get(_ALIASES.get(key.lower(), ""))
    if tool is None:
        known = ", ".join(sorted(tools)) or "none"
        raise ValueError(f"Unknown CLI tool {name!r}. Known: {known}, all")
    if not tool.is_active():
        status = tool.check()
        logger.warning("cli:%s is not available (%s). %s", tool.name, status.status, status.fix_hint or "")
        return []
    return tool.to_tool_definitions()


__all__ = [
    "CLIArg",
    "CLICommand",
    "CLIResult",
    "CLITool",
    "CLIToolError",
    "all_cli_tools",
    "builtin_cli_tools",
    "config_errors",
    "config_paths",
    "gh_tool",
    "load_cli_tools",
    "parse_cli_tools",
    "resolve_cli_tools",
    "run_process",
    "vtt_to_text",
    "yt_dlp_tool",
]
