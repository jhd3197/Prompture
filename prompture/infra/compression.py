"""Lossless-ish prompt compression for chat messages.

``compress_messages`` applies conservative, meaning-preserving reductions
that matter most for agentic traffic, where tool output dominates the
token bill:

* collapse runs of blank lines and trailing whitespace
* drop a system message that repeats an earlier one verbatim
* trim oversized tool results to a head + tail window with a marker, so
  the model still sees how the output starts and ends

It never rewrites user or assistant prose, and never touches the most
recent tool result (the one the model is about to reason over) unless
``trim_latest_tool=True``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_BLANK_RUN = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_TRAILING_WS = re.compile(r"[ \t]+\n")


@dataclass
class CompressionStats:
    chars_before: int = 0
    chars_after: int = 0
    tool_results_trimmed: int = 0
    duplicate_system_removed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def saved_chars(self) -> int:
        return self.chars_before - self.chars_after

    @property
    def ratio(self) -> float:
        return self.chars_after / self.chars_before if self.chars_before else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "saved_chars": self.saved_chars,
            "tool_results_trimmed": self.tool_results_trimmed,
            "duplicate_system_removed": self.duplicate_system_removed,
        }


def _tidy(text: str) -> str:
    text = _TRAILING_WS.sub("\n", text)
    return _BLANK_RUN.sub("\n\n", text)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n[... {omitted:,} characters of tool output omitted ...]\n\n{text[-tail:]}"


def _size(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(b.get("text", "")) for b in content if isinstance(b, dict))
    return 0


def compress_messages(
    messages: list[dict[str, Any]],
    *,
    max_tool_chars: int = 6000,
    trim_latest_tool: bool = False,
    collapse_whitespace: bool = True,
    dedupe_system: bool = True,
) -> tuple[list[dict[str, Any]], CompressionStats]:
    """Return ``(compressed_copy, stats)``. The input list is not modified."""
    stats = CompressionStats(chars_before=sum(_size(m.get("content")) for m in messages))
    last_tool = max((i for i, m in enumerate(messages) if m.get("role") == "tool"), default=-1)
    seen_system: set[str] = set()
    out: list[dict[str, Any]] = []

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if role == "system" and dedupe_system and isinstance(content, str):
            if content in seen_system:
                stats.duplicate_system_removed += 1
                continue
            seen_system.add(content)
        if isinstance(content, str):
            new = _tidy(content) if collapse_whitespace and role in ("system", "tool") else content
            if role == "tool" and (i != last_tool or trim_latest_tool):
                trimmed = _trim(new, max_tool_chars)
                if trimmed != new:
                    stats.tool_results_trimmed += 1
                new = trimmed
            if new != content:
                msg = {**msg, "content": new}
        out.append(msg)

    stats.chars_after = sum(_size(m.get("content")) for m in out)
    return out, stats
