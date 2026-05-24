"""Pluggable long-term memory for swarm agents.

The :class:`MemoryStore` protocol defines the contract every backend
must satisfy.  :class:`InMemoryStore` is the default implementation: a
thread-safe per-agent recency log with token-overlap keyword search.

Third-party backends (Zep, Mem0, a vector store, etc.) plug in by
implementing the same four methods.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from .types import Memory

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    """Lower-cased word tokens for naive keyword overlap scoring."""
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


@runtime_checkable
class MemoryStore(Protocol):
    """Per-agent long-term memory store.

    All operations are agent-scoped: an agent only ever sees its own
    memories.  Implementations should be thread-safe — the swarm
    runtime may invoke ``add``, ``search``, and ``recent`` concurrently
    when an async/parallel scheduler is used.
    """

    def add(
        self,
        agent_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        ts: float | None = None,
        tick: int | None = None,
    ) -> None:
        """Append a memory for *agent_id*.

        Args:
            agent_id: Owning agent.
            content: Free-form text payload.
            metadata: Optional structured metadata.
            ts: Wall-clock timestamp.  Defaults to ``time.time()``.
            tick: Optional round index of authorship.
        """
        ...

    def search(self, agent_id: str, query: str, *, k: int = 5) -> list[Memory]:
        """Return the *k* memories most relevant to *query* for *agent_id*.

        Relevance is implementation-defined.  Backends may use exact
        keyword match, BM25, embeddings, or a remote service.
        """
        ...

    def recent(self, agent_id: str, *, k: int = 5) -> list[Memory]:
        """Return the *k* most recently added memories for *agent_id*.

        Results are ordered oldest-first so they read naturally when
        spliced into a prompt.
        """
        ...

    def clear(self, agent_id: str | None = None) -> None:
        """Drop memories.

        When *agent_id* is ``None`` clear the entire store; otherwise
        clear only memories owned by that agent.
        """
        ...


class InMemoryStore:
    """Thread-safe in-process implementation of :class:`MemoryStore`.

    Stores a per-agent list of :class:`Memory` records.  Keyword search
    scores each candidate by the number of query tokens that appear in
    the memory's content (case-insensitive); ties break toward more
    recent memories.

    This is the default backend; it has no external dependencies and is
    suitable for tests, short-lived runs, and small populations.  For
    long-running simulations consider plugging in a vector store or a
    hosted service (Zep, Mem0) by implementing :class:`MemoryStore`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, list[Memory]] = defaultdict(list)

    def add(
        self,
        agent_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        ts: float | None = None,
        tick: int | None = None,
    ) -> None:
        record = Memory(
            agent_id=agent_id,
            content=content,
            ts=ts if ts is not None else time.time(),
            tick=tick,
            metadata=dict(metadata) if metadata else {},
        )
        with self._lock:
            self._records[agent_id].append(record)

    def search(self, agent_id: str, query: str, *, k: int = 5) -> list[Memory]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return self.recent(agent_id, k=k)

        with self._lock:
            candidates = list(self._records.get(agent_id, ()))

        scored: list[tuple[int, float, Memory]] = []
        for m in candidates:
            overlap = len(_tokens(m.content) & q_tokens)
            if overlap > 0:
                scored.append((overlap, m.ts, m))

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [m for _, _, m in scored[:k]]

    def recent(self, agent_id: str, *, k: int = 5) -> list[Memory]:
        with self._lock:
            records = list(self._records.get(agent_id, ()))
        if k <= 0:
            return []
        return records[-k:]

    def clear(self, agent_id: str | None = None) -> None:
        with self._lock:
            if agent_id is None:
                self._records.clear()
            else:
                self._records.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Introspection helpers (not part of the protocol)
    # ------------------------------------------------------------------

    def all_for(self, agent_id: str) -> list[Memory]:
        """Return every memory owned by *agent_id* (test/debug helper)."""
        with self._lock:
            return list(self._records.get(agent_id, ()))

    def agent_ids(self) -> list[str]:
        """Return every agent ID with at least one stored memory."""
        with self._lock:
            return list(self._records.keys())
