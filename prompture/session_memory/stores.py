"""Storage backends for :class:`SessionMemory`.

Two implementations ship out of the box:

- :class:`InMemorySessionStore` — thread-safe, no dependencies.
  Suitable for tests, scripts, and single-process services.
- :class:`SQLiteSessionStore` — file-backed durable storage with the
  same indexing pattern as :class:`~prompture.persistence.ConversationStore`.

Both satisfy the :class:`SessionMemoryStore` protocol.  Bring-your-own
backends (Postgres, Redis, a vector store, Zep, Mem0) implement the
same four methods.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .types import MemoryFact

_DEFAULT_DB_DIR = Path.home() / ".prompture" / "session_memory"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "session_memory.db"
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


@runtime_checkable
class SessionMemoryStore(Protocol):
    """Pluggable persistence contract for :class:`SessionMemory`.

    All operations are user-scoped: callers identify themselves with a
    ``user_id`` and an optional ``session_id``.  Implementations should
    be thread-safe.
    """

    def add(self, fact: MemoryFact) -> None:
        """Persist a new :class:`MemoryFact` (or upsert by id)."""
        ...

    def list(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[MemoryFact]:
        """Return facts for *user_id*, optionally narrowed to a session/kind.

        Results are oldest-first.  When ``session_id`` is ``None``
        every session for the user is included; pass a string to
        restrict to one session (use the literal sentinel value
        ``"__user__"`` to mean "user-wide facts only").
        """
        ...

    def search(
        self,
        user_id: str,
        query: str,
        *,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        k: int = 5,
    ) -> list[MemoryFact]:
        """Return up to *k* facts relevant to *query* for *user_id*.

        Relevance is implementation-defined; default backends use
        token-overlap scoring weighted by importance and recency.
        """
        ...

    def delete(self, user_id: str, *, session_id: str | None = None, fact_id: str | None = None) -> int:
        """Drop facts.  Returns the number deleted.

        At least one of ``session_id`` / ``fact_id`` should be supplied
        — implementations may choose to refuse a totally-unscoped
        delete to avoid accidental data loss.
        """
        ...


# ----------------------------------------------------------------------
# In-memory backend
# ----------------------------------------------------------------------


class InMemorySessionStore:
    """Thread-safe in-process implementation of :class:`SessionMemoryStore`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Two-level index: user_id → list[MemoryFact]
        self._by_user: dict[str, list[MemoryFact]] = defaultdict(list)

    def add(self, fact: MemoryFact) -> None:
        with self._lock:
            records = self._by_user[fact.user_id]
            # Upsert by id
            for i, existing in enumerate(records):
                if existing.id == fact.id:
                    records[i] = fact
                    return
            records.append(fact)

    def list(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[MemoryFact]:
        with self._lock:
            records = list(self._by_user.get(user_id, ()))

        out: list[MemoryFact] = []
        for r in records:
            if session_id == "__user__":
                if r.session_id is not None:
                    continue
            elif session_id is not None and r.session_id != session_id:
                continue
            if kinds is not None and r.kind not in kinds:
                continue
            out.append(r)
        out.sort(key=lambda r: r.ts)
        if limit is not None and limit >= 0:
            out = out[-limit:]
        return out

    def search(
        self,
        user_id: str,
        query: str,
        *,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        k: int = 5,
    ) -> list[MemoryFact]:
        candidates = self.list(user_id, session_id=session_id, kinds=kinds)
        q = _tokens(query)
        if not q:
            return candidates[-k:] if k > 0 else []

        now = time.time()
        scored: list[tuple[float, MemoryFact]] = []
        for fact in candidates:
            overlap = len(_tokens(fact.content) & q)
            if overlap == 0:
                continue
            # Importance + recency boost.  Half-life ~7 days.
            age_days = max(0.0, (now - fact.ts) / 86400.0)
            recency = 0.5 ** (age_days / 7.0)
            score = overlap + (fact.importance * 0.5) + (recency * 0.25)
            scored.append((score, fact))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [f for _, f in scored[:k]] if k > 0 else []

    def delete(self, user_id: str, *, session_id: str | None = None, fact_id: str | None = None) -> int:
        if session_id is None and fact_id is None:
            raise ValueError(
                "Refusing to delete every memory for a user without a "
                "session_id or fact_id filter — pass session_id='__user__' "
                "to clear user-wide facts explicitly."
            )

        with self._lock:
            records = self._by_user.get(user_id, [])
            keep: list[MemoryFact] = []
            removed = 0
            for r in records:
                drop = False
                if fact_id is not None and r.id == fact_id:
                    drop = True
                elif session_id is not None:
                    if session_id == "__user__":
                        drop = r.session_id is None
                    else:
                        drop = r.session_id == session_id
                if drop:
                    removed += 1
                else:
                    keep.append(r)
            self._by_user[user_id] = keep
            return removed


# ----------------------------------------------------------------------
# SQLite backend
# ----------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_memory (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT,
    content TEXT NOT NULL,
    kind TEXT NOT NULL,
    ts REAL NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_session_memory_user
    ON session_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_session_memory_user_session
    ON session_memory(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_session_memory_ts
    ON session_memory(user_id, ts);
"""


class SQLiteSessionStore:
    """File-backed durable :class:`SessionMemoryStore`.

    Mirrors :class:`~prompture.persistence.ConversationStore`: single
    file at ``~/.prompture/session_memory/session_memory.db`` by
    default, WAL mode, thread-safe via a single ``threading.Lock``.

    Args:
        db_path: Override the default DB location.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, fact: MemoryFact) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO session_memory
                        (id, user_id, session_id, content, kind, ts, importance, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content = excluded.content,
                        kind = excluded.kind,
                        ts = excluded.ts,
                        importance = excluded.importance,
                        metadata = excluded.metadata
                    """,
                    (
                        fact.id,
                        fact.user_id,
                        fact.session_id,
                        fact.content,
                        fact.kind,
                        fact.ts,
                        fact.importance,
                        json.dumps(fact.metadata, ensure_ascii=False),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def list(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[MemoryFact]:
        sql = "SELECT * FROM session_memory WHERE user_id = ?"
        params: list[Any] = [user_id]
        if session_id == "__user__":
            sql += " AND session_id IS NULL"
        elif session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if kinds is not None and len(kinds) > 0:
            placeholders = ",".join("?" * len(kinds))
            sql += f" AND kind IN ({placeholders})"
            params.extend(kinds)
        sql += " ORDER BY ts ASC"

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()

        out = [self._row_to_fact(r) for r in rows]
        if limit is not None and limit >= 0:
            out = out[-limit:]
        return out

    def search(
        self,
        user_id: str,
        query: str,
        *,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        k: int = 5,
    ) -> list[MemoryFact]:
        # Reuse the in-memory scorer over the SQL-filtered candidate set.
        candidates = self.list(user_id, session_id=session_id, kinds=kinds)
        q = _tokens(query)
        if not q:
            return candidates[-k:] if k > 0 else []

        now = time.time()
        scored: list[tuple[float, MemoryFact]] = []
        for fact in candidates:
            overlap = len(_tokens(fact.content) & q)
            if overlap == 0:
                continue
            age_days = max(0.0, (now - fact.ts) / 86400.0)
            recency = 0.5 ** (age_days / 7.0)
            score = overlap + (fact.importance * 0.5) + (recency * 0.25)
            scored.append((score, fact))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [f for _, f in scored[:k]] if k > 0 else []

    def delete(self, user_id: str, *, session_id: str | None = None, fact_id: str | None = None) -> int:
        if session_id is None and fact_id is None:
            raise ValueError(
                "Refusing to delete every memory for a user without a "
                "session_id or fact_id filter."
            )

        sql = "DELETE FROM session_memory WHERE user_id = ?"
        params: list[Any] = [user_id]
        if fact_id is not None:
            sql += " AND id = ?"
            params.append(fact_id)
        elif session_id == "__user__":
            sql += " AND session_id IS NULL"
        elif session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)

        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(sql, params)
                conn.commit()
                return int(cursor.rowcount)
            finally:
                conn.close()

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> MemoryFact:
        return MemoryFact(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=row["content"],
            kind=row["kind"],
            ts=float(row["ts"]),
            importance=float(row["importance"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
