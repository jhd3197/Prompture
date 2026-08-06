"""Prompted tool-use grammars for drivers without native function-calling.

When a driver lacks native tool-calling (HuggingFace inference, raw local
HTTP, older Ollama models, AirLLM, …) Prompture can still deliver the
``Conversation.ask_live`` interleaved streaming-tool experience by:

1. Injecting tool schemas into the system prompt using a chosen *grammar*
   (e.g. ``<tool_call name="x">{...}</tool_call>`` tags).
2. Streaming the model's text output through
   :class:`~prompture.drivers._prompted_tool_stream.PromptedToolStreamParser`,
   which detects the grammar's delimiters mid-stream and emits the same
   :mod:`~prompture.agents.live_events` as a native streaming-tool driver.

The default grammar — :data:`XML_TAGS_GRAMMAR` — is the most robust choice
for small open-weight models: explicit unambiguous delimiters, doesn't
interfere with markdown narration, and exposes the tool name at the
opening tag so ``ToolUseStart`` can fire before the args finish streaming.

Other formats (``json_fence``, classic ReAct ``Action:`` / ``Action Input:``)
can be plugged in via :func:`register_grammar`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolGrammar:
    """Describes one prompted tool-use grammar.

    The grammar drives both directions:

    - **System-prompt injection** — :attr:`render_system_prompt` builds the
      instruction block appended to the system message so the model emits
      tool calls in the expected shape.
    - **Streaming parser** — :attr:`open_regex` / :attr:`close_marker`
      delimit tool-call blocks in the token stream;
      :attr:`parse_open_tag` extracts ``(name, id)`` from the opening
      tag so :class:`~prompture.agents.live_events.ToolUseStart` can fire
      immediately, before the arguments finish streaming.
    """

    name: str
    """Stable identifier (``"xml_tags"``, ``"react"`` …) used as the
    grammar key in :data:`GRAMMARS` and the ``prompted_tool_grammar``
    driver attribute."""

    open_regex: re.Pattern[str]
    """Compiled regex matching the opening delimiter. Must include a
    sentinel terminator (e.g. trailing ``>``) so the parser can detect
    completeness even when the chunk boundary falls inside the tag."""

    close_marker: str
    """Literal string that closes a tool-call block (e.g. ``</tool_call>``)."""

    parse_open_tag: Callable[[re.Match[str]], tuple[str, str | None]]
    """Given the regex match for the opening tag, return ``(tool_name, tool_id)``.
    ``tool_id`` may be ``None`` — the parser will auto-generate one."""

    render_system_prompt: Callable[[list[dict[str, Any]]], str]
    """Given the OpenAI-style ``tools`` list, return the instruction block
    appended to the system prompt so the model emits this grammar."""

    open_prefix: str = "<tool_call"
    """Literal leading characters of :attr:`open_regex`. The streaming
    parser derives its holdback from this: while in narration it never
    emits a trailing run of characters that could still grow into this
    prefix, so a delimiter split across chunks is still detected."""


# ---------------------------------------------------------------------------
# Default grammar: XML-style tags with name + id attributes
# ---------------------------------------------------------------------------

# Matches  <tool_call name="x">  or  <tool_call name="x" id="y">
# Attribute values may be double- or single-quoted.
_XML_OPEN_RE = re.compile(
    r"<tool_call\s+name\s*=\s*(?:\"(?P<name_dq>[^\"]+)\"|'(?P<name_sq>[^']+)')"
    r"(?:\s+id\s*=\s*(?:\"(?P<id_dq>[^\"]+)\"|'(?P<id_sq>[^']+)'))?\s*>"
)


def _xml_parse_open(match: re.Match[str]) -> tuple[str, str | None]:
    name = match.group("name_dq") or match.group("name_sq")
    tag_id = match.group("id_dq") or match.group("id_sq")
    return name, tag_id


def _xml_render_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Build the XML-tag tool-use instruction block.

    Each tool becomes a bullet point with its description and JSON schema.
    The instructions specify the exact opening/closing tag format and
    show one worked example so the model anchors on it.
    """
    lines: list[str] = [
        "",
        "## Available tools",
        "",
        "You have access to the following tools. Call them when you need information",
        "or actions that require them — otherwise answer directly.",
        "",
    ]
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        desc = fn.get("description", "") or "(no description)"
        params = fn.get("parameters", {}) or {}
        params_json = json.dumps(params, separators=(",", ":"))
        desc_lines = desc.strip().split("\n")
        lines.append(f"- **{name}** — {desc_lines[0]}")
        # Indent continuation lines so a multi-line description stays inside the bullet.
        lines.extend(f"  {line.strip()}" if line.strip() else "" for line in desc_lines[1:])
        lines.append(f"  Parameters JSON Schema: `{params_json}`")
    lines.extend(
        [
            "",
            "## How to call a tool",
            "",
            "When you need to call a tool, emit a block exactly like this in your reply:",
            "",
            '<tool_call name="TOOL_NAME">',
            '{"arg1": "value1", "arg2": 42}',
            "</tool_call>",
            "",
            "Rules:",
            '- The opening tag must be on its own and include `name="..."` matching one of the tools above.',
            "- The body between the tags must be a single valid JSON object — the tool arguments.",
            "- The closing tag is `</tool_call>` on its own line (or immediately after the JSON).",
            "- You may narrate normally before, between, and after tool calls.",
            "- Wait for the tool result before deciding what to do next; results arrive as a `tool` role message.",
            "- If no tool is needed, just answer the user directly with no `<tool_call>` block.",
            "",
        ]
    )
    return "\n".join(lines)


XML_TAGS_GRAMMAR = ToolGrammar(
    name="xml_tags",
    open_regex=_XML_OPEN_RE,
    close_marker="</tool_call>",
    parse_open_tag=_xml_parse_open,
    render_system_prompt=_xml_render_system_prompt,
    open_prefix="<tool_call",
)
"""Default prompted-tool grammar — XML-style ``<tool_call name="..." [id="..."]>{json}</tool_call>``.

Robust for small open-weight models: explicit delimiters, doesn't clash
with markdown, and exposes the tool name at the opening tag so the live
stream can emit :class:`~prompture.agents.live_events.ToolUseStart`
before the arguments finish."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRAMMARS: dict[str, ToolGrammar] = {
    "xml_tags": XML_TAGS_GRAMMAR,
}
"""Global registry of available prompted-tool grammars, keyed by
:attr:`ToolGrammar.name`."""


def register_grammar(grammar: ToolGrammar) -> None:
    """Add a grammar to the registry (overwrites if name already exists)."""
    GRAMMARS[grammar.name] = grammar


def get_grammar(name: str) -> ToolGrammar:
    """Look up a grammar by name. Raises ``KeyError`` if not registered."""
    try:
        return GRAMMARS[name]
    except KeyError as exc:
        available = ", ".join(sorted(GRAMMARS)) or "(none registered)"
        raise KeyError(f"Unknown tool grammar {name!r}. Available: {available}") from exc


def inject_tool_instructions(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    grammar: ToolGrammar,
) -> list[dict[str, Any]]:
    """Return a copy of *messages* with the grammar's tool instructions
    appended to the system message.

    If no system message exists, one is inserted at index 0. The original
    messages list is not mutated.
    """
    if not tools:
        return list(messages)

    instructions = grammar.render_system_prompt(tools)
    out = [dict(m) for m in messages]
    for msg in out:
        if msg.get("role") == "system":
            existing = msg.get("content", "") or ""
            msg["content"] = (existing.rstrip() + "\n" + instructions) if existing else instructions
            return out
    out.insert(0, {"role": "system", "content": instructions})
    return out


__all__ = [
    "GRAMMARS",
    "XML_TAGS_GRAMMAR",
    "ToolGrammar",
    "get_grammar",
    "inject_tool_instructions",
    "register_grammar",
]
