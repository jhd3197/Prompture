"""Reusing answers for background requests whose answer doesn't depend on the conversation.

Most coding-CLI requests must never be answered from a cache: each turn
depends on the whole conversation. A few don't. A title for a new
conversation is a function of its first prompt and a fixed instruction, so
the same (or nearly the same) prompt deserves the same title. :class:`AnswerCache`
keeps the vendor's reply to such requests and replays it for a near-identical
one, when ``routes.json`` turns it on::

    {"cache": {"enabled": true, "kinds": ["title"], "similarity": 0.9}}

Two requests match when their instructions (system prompt and tools) are
identical and their prompt text is at least ``similarity`` alike, by the
share of words they have in common. Entries last :data:`TTL_SECONDS` and
only replies that succeeded are kept. Kept in memory only; a restart starts
empty.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

TTL_SECONDS = 7 * 86400
MAX_ENTRIES = 500
#: Replies larger than this aren't kept (a title is a few hundred bytes).
MAX_BYTES = 256 * 1024
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass
class Answer:
    """A kept reply: replayed byte for byte."""

    status: int
    content_type: str
    body: bytes
    words: frozenset[str]
    at: float
    usage: dict[str, int]
    model: str | None


def _words(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in _WORD.findall(text))


def _similar(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if a | b else 0.0


def split_request(dialect: str, body: dict[str, Any]) -> tuple[str, str]:
    """``(fixed part, prompt text)``: what must match exactly, and what may be merely alike."""
    if dialect == "anthropic":
        fixed = {"system": body.get("system"), "tools": body.get("tools"), "model": body.get("model")}
        messages = body.get("messages") or []
        prompt = messages[-1].get("content") if messages and isinstance(messages[-1], dict) else ""
    elif dialect == "gemini":
        inner = body.get("request") if isinstance(body.get("request"), dict) else body
        fixed = {"system": inner.get("systemInstruction"), "tools": inner.get("tools"), "model": body.get("model")}
        contents = inner.get("contents") or []
        prompt = contents[-1].get("parts") if contents and isinstance(contents[-1], dict) else ""
    else:
        fixed = {"instructions": body.get("instructions"), "tools": body.get("tools"), "model": body.get("model")}
        items = body.get("input") or []
        prompt = items[-1].get("content") if items and isinstance(items[-1], dict) else ""
    text = prompt if isinstance(prompt, str) else json.dumps(prompt, default=str)
    return json.dumps(fixed, sort_keys=True, default=str), text


class AnswerCache:
    """Replies to cacheable requests, per exact instructions, matched by prompt similarity. Thread-safe."""

    def __init__(self) -> None:
        self._entries: OrderedDict[str, list[Answer]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0

    @staticmethod
    def _key(tool: str, kind: str, fixed: str) -> str:
        return hashlib.sha256(f"{tool}\0{kind}\0{fixed}".encode()).hexdigest()

    def lookup(
        self, tool: str, dialect: str, kind: str, body: dict[str, Any], similarity: float
    ) -> tuple[Answer, float] | None:
        fixed, text = split_request(dialect, body)
        words = _words(text)
        if len(words) < 3:
            return None  # too little to judge likeness
        now = time.time()
        with self._lock:
            answers = self._entries.get(self._key(tool, kind, fixed)) or []
            best: tuple[Answer, float] | None = None
            for answer in answers:
                if now - answer.at > TTL_SECONDS:
                    continue
                score = _similar(words, answer.words)
                if score >= similarity and (best is None or score > best[1]):
                    best = (answer, score)
            if best:
                self.hits += 1
            return best

    def store(
        self,
        tool: str,
        dialect: str,
        kind: str,
        body: dict[str, Any],
        *,
        status: int,
        content_type: str,
        data: bytes,
        usage: dict[str, int],
        model: str | None,
    ) -> None:
        if status != 200 or not data or len(data) > MAX_BYTES:
            return
        fixed, text = split_request(dialect, body)
        answer = Answer(status, content_type, data, _words(text), time.time(), usage, model)
        key = self._key(tool, kind, fixed)
        with self._lock:
            answers = self._entries.pop(key, [])
            answers = [*[a for a in answers if a.words != answer.words][-19:], answer]
            self._entries[key] = answers
            while len(self._entries) > MAX_ENTRIES:
                self._entries.popitem(last=False)
