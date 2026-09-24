"""Reasoning replay for stateless HTTP clients.

Thinking models (DeepSeek reasoner, Kimi thinking, …) return
``reasoning_content`` and some reject a follow-up turn whose earlier
assistant/tool-call messages lost it. OpenAI-style clients never send that
field back, so a gateway remembers it — keyed by tool-call id, or by the
assistant text when there were no tool calls — and re-attaches it on the
next request.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

from .openai_format import ChatOutcome


def _text_key(text: str) -> str:
    return "t:" + hashlib.sha256(text.strip().encode()).hexdigest()[:32]


def _call_ids(message: dict[str, Any]) -> list[str]:
    return [str(tc.get("id")) for tc in message.get("tool_calls") or [] if tc.get("id")]


class ReasoningCache:
    """Bounded, TTL'd LRU of ``key → reasoning_content``. Thread-safe."""

    def __init__(self, max_entries: int = 5000, ttl: float = 6 * 3600) -> None:
        self.max_entries = max_entries
        self.ttl = ttl
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def _put(self, key: str, reasoning: str) -> None:
        self._data[key] = (time.monotonic(), reasoning)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def _get(self, key: str) -> str | None:
        item = self._data.get(key)
        if item is None:
            return None
        stored_at, reasoning = item
        if time.monotonic() - stored_at > self.ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return reasoning

    def remember(self, outcome: ChatOutcome) -> None:
        """Store the reasoning of a finished turn (no-op when there was none)."""
        if not outcome.reasoning:
            return
        with self._lock:
            for tc in outcome.tool_calls:
                if tc.get("id"):
                    self._put(f"c:{tc['id']}", outcome.reasoning)
            if outcome.text:
                self._put(_text_key(outcome.text), outcome.reasoning)

    def restore(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Copy of *messages* with cached ``reasoning_content`` re-attached to assistant turns."""
        out: list[dict[str, Any]] = []
        with self._lock:
            for msg in messages:
                if msg.get("role") != "assistant" or msg.get("reasoning_content"):
                    out.append(msg)
                    continue
                reasoning = None
                for cid in _call_ids(msg):
                    reasoning = self._get(f"c:{cid}")
                    if reasoning:
                        break
                content = msg.get("content")
                if reasoning is None and isinstance(content, str) and content.strip():
                    reasoning = self._get(_text_key(content))
                out.append({**msg, "reasoning_content": reasoning} if reasoning else msg)
        return out

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_default = ReasoningCache()


def get_reasoning_cache() -> ReasoningCache:
    """Process-wide cache shared by gateway endpoints."""
    return _default
