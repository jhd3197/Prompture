"""Structured events emitted by coding-agent CLIs.

Coding-agent CLIs that support a JSON event stream (Claude Code's
``--output-format stream-json``, Codex's ``--json``, etc.) emit one JSON
object per event. Each per-CLI parser in this module normalises that stream
into a sequence of :class:`CodingAgentEvent` instances so callers don't have
to know each CLI's payload shape.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Literal

EventType = Literal[
    "system",
    "message",
    "tool_call",
    "tool_result",
    "question",
    "done",
    "error",
]


@dataclasses.dataclass(frozen=True)
class CodingAgentEvent:
    """One normalised event emitted by a coding-agent CLI."""

    type: EventType
    text: str | None = None
    tool_name: str | None = None
    tool_input: Mapping[str, Any] | None = None
    tool_output: str | None = None
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int | None = None
    error: str | None = None
    choices: tuple[str, ...] | None = None
    session_id: str | None = None
    raw: Mapping[str, Any] | None = None


def parse_claude_stream_json_lines(lines: Iterable[str]) -> Iterator[CodingAgentEvent]:
    """Yield :class:`CodingAgentEvent` per ``claude --output-format stream-json`` line."""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            yield CodingAgentEvent(
                type="error",
                error=f"invalid JSON line: {stripped[:200]}",
            )
            continue
        if not isinstance(obj, dict):
            continue
        yield from _claude_event_to_events(obj)


def _claude_event_to_events(obj: Mapping[str, Any]) -> Iterator[CodingAgentEvent]:
    kind = obj.get("type")
    if kind == "system":
        sid = obj.get("session_id")
        yield CodingAgentEvent(
            type="system",
            session_id=sid if isinstance(sid, str) else None,
            raw=obj,
        )
        return
    if kind == "result":
        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else {}
        is_error = bool(obj.get("is_error"))
        text = obj.get("result") if isinstance(obj.get("result"), str) else None
        sid = obj.get("session_id")
        yield CodingAgentEvent(
            type="done",
            text=text,
            cost_usd=_safe_float(obj.get("total_cost_usd")),
            input_tokens=_safe_int(usage.get("input_tokens")),
            output_tokens=_safe_int(usage.get("output_tokens")),
            duration_ms=_safe_int(obj.get("duration_ms")),
            error=text if is_error else None,
            session_id=sid if isinstance(sid, str) else None,
            raw=obj,
        )
        question = _question_event_for_text(text, obj)
        if question is not None:
            yield question
        return
    if kind in ("assistant", "user"):
        message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, list):
            return
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text" and isinstance(part.get("text"), str):
                yield CodingAgentEvent(type="message", text=part["text"], raw=obj)
                question = _question_event_for_text(part["text"], obj)
                if question is not None:
                    yield question
            elif part_type == "tool_use":
                yield CodingAgentEvent(
                    type="tool_call",
                    tool_name=part.get("name") if isinstance(part.get("name"), str) else None,
                    tool_input=part.get("input") if isinstance(part.get("input"), dict) else None,
                    raw=obj,
                )
            elif part_type == "tool_result":
                yield CodingAgentEvent(
                    type="tool_result",
                    tool_output=_flatten_tool_result(part.get("content")),
                    raw=obj,
                )


def _flatten_tool_result(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks) if chunks else None
    return None


def parse_codex_json_lines(lines: Iterable[str]) -> Iterator[CodingAgentEvent]:
    """Yield :class:`CodingAgentEvent` per ``codex exec --json`` line.

    Codex emits one JSON object per event with shape
    ``{"id": "...", "msg": {"type": ..., ...}}``. We translate the subset
    of event types that have analogues in the normalised event model.
    """
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            yield CodingAgentEvent(
                type="error",
                error=f"invalid JSON line: {stripped[:200]}",
            )
            continue
        if not isinstance(obj, dict):
            continue
        yield from _codex_event_to_events(obj)


def _codex_event_to_events(obj: Mapping[str, Any]) -> Iterator[CodingAgentEvent]:
    msg = obj.get("msg") if isinstance(obj.get("msg"), dict) else None
    if msg is None:
        return
    kind = msg.get("type")
    if kind in ("task_started", "session_configured"):
        # Codex emits session_id either nested in msg or at the envelope level.
        sid = msg.get("session_id") if isinstance(msg.get("session_id"), str) else None
        if sid is None and isinstance(obj.get("session_id"), str):
            sid = obj["session_id"]
        yield CodingAgentEvent(type="system", session_id=sid, raw=obj)
        return
    if kind == "agent_message":
        text = msg.get("message")
        if isinstance(text, str):
            yield CodingAgentEvent(type="message", text=text, raw=obj)
            question = _question_event_for_text(text, obj)
            if question is not None:
                yield question
        return
    if kind == "agent_message_delta":
        delta = msg.get("delta")
        if isinstance(delta, str) and delta:
            yield CodingAgentEvent(type="message", text=delta, raw=obj)
        return
    if kind == "exec_command_begin":
        command = msg.get("command")
        yield CodingAgentEvent(
            type="tool_call",
            tool_name="exec",
            tool_input={"command": command} if command is not None else None,
            raw=obj,
        )
        return
    if kind == "exec_command_end":
        chunks = []
        for key in ("stdout", "stderr"):
            value = msg.get(key)
            if isinstance(value, str) and value:
                chunks.append(value)
        yield CodingAgentEvent(
            type="tool_result",
            tool_output="\n".join(chunks) if chunks else None,
            raw=obj,
        )
        return
    if kind in ("mcp_tool_call_begin",):
        yield CodingAgentEvent(
            type="tool_call",
            tool_name=msg.get("tool") if isinstance(msg.get("tool"), str) else None,
            tool_input=msg.get("arguments") if isinstance(msg.get("arguments"), dict) else None,
            raw=obj,
        )
        return
    if kind in ("mcp_tool_call_end",):
        result = msg.get("result")
        yield CodingAgentEvent(
            type="tool_result",
            tool_output=result if isinstance(result, str) else None,
            raw=obj,
        )
        return
    if kind == "task_complete":
        usage = msg.get("token_usage") if isinstance(msg.get("token_usage"), dict) else {}
        yield CodingAgentEvent(
            type="done",
            text=msg.get("last_agent_message") if isinstance(msg.get("last_agent_message"), str) else None,
            input_tokens=_safe_int(usage.get("input_tokens")),
            output_tokens=_safe_int(usage.get("output_tokens")),
            raw=obj,
        )
        return
    if kind == "error":
        yield CodingAgentEvent(
            type="error",
            error=msg.get("message") if isinstance(msg.get("message"), str) else "codex error",
            raw=obj,
        )
        return


_QUESTION_PHRASE_RE = re.compile(
    r"\b(would you like|do you want|should i|which (?:one|option|of|approach)|"
    r"can you (?:clarify|confirm|specify)|please (?:clarify|confirm|specify|choose)|"
    r"how would you like|what would you like|let me know which)",
    re.IGNORECASE,
)

_CHOICE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?P<num>\d+)[.\):]\s+|"     # 1. foo  1) foo  1: foo
    r"[-*•]\s+|"                  # - foo   * foo   • foo
    r"\([a-zA-Z]\)\s+|"           # (a) foo
    r"[a-zA-Z][.\)]\s+"           # a) foo  a. foo
    r")(?P<body>.+\S)\s*$"
)


def detect_question(text: str | None) -> tuple[bool, tuple[str, ...] | None]:
    """Return ``(is_question, choices)`` for an assistant message text.

    A message is treated as a question when it ends in ``?`` or contains a
    question-phrase trigger ("do you want", "would you like", …). Choices
    are extracted from numbered, bulleted, or lettered list lines — at least
    two such lines must appear together for them to count as choices.
    """
    if not text:
        return False, None
    stripped = text.rstrip()
    if not stripped:
        return False, None

    is_question = stripped.endswith("?") or bool(_QUESTION_PHRASE_RE.search(stripped))

    choices: list[str] = []
    consecutive: list[str] = []
    for raw_line in stripped.splitlines():
        match = _CHOICE_LINE_RE.match(raw_line)
        if match:
            consecutive.append(match.group("body").strip())
        else:
            if len(consecutive) >= 2:
                choices.extend(consecutive)
            consecutive = []
    if len(consecutive) >= 2:
        choices.extend(consecutive)

    # Choices in the absence of a `?` or phrase trigger don't make a question.
    if choices and not is_question:
        is_question = True

    return is_question, tuple(choices) if choices else None


def _question_event_for_text(text: str | None, raw: Mapping[str, Any] | None) -> CodingAgentEvent | None:
    is_q, choices = detect_question(text)
    if not is_q:
        return None
    return CodingAgentEvent(type="question", text=text, choices=choices, raw=raw)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
