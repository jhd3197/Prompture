"""Conversation persistence — file and SQLite storage backends.

Provides:

- :func:`save_to_file` / :func:`load_from_file` for simple JSON file storage.
- :class:`ConversationStore` for SQLite-backed storage with tag search.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# File-based persistence
# ------------------------------------------------------------------


def save_to_file(data: dict[str, Any], path: str | Path) -> None:
    """Write a conversation export dict as JSON to *path*.

    Creates parent directories if they don't exist.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_from_file(path: str | Path) -> dict[str, Any]:
    """Read a conversation export dict from a JSON file.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Conversation file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# SQLite-backed ConversationStore
# ------------------------------------------------------------------

_DEFAULT_DB_DIR = Path.home() / ".prompture" / "conversations"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "conversations.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_active TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS conversation_tags (
    conversation_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (conversation_id, tag),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_tag ON conversation_tags(tag);
CREATE INDEX IF NOT EXISTS idx_conversations_last_active ON conversations(last_active);
"""


class ConversationStore:
    """SQLite-backed conversation storage with tag search.

    Thread-safe — uses an internal :class:`threading.Lock` for all
    database operations (mirrors the pattern used by ``cache.py``).

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``~/.prompture/conversations/conversations.db``.
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
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #

    def save(self, conversation_id: str, data: dict[str, Any]) -> None:
        """Upsert a conversation and replace its tags."""
        meta = data.get("metadata", {})
        model_name = data.get("model_name", "")
        created_at = meta.get("created_at", datetime.now(timezone.utc).isoformat())
        last_active = meta.get("last_active", datetime.now(timezone.utc).isoformat())
        turn_count = meta.get("turn_count", 0)
        tags = meta.get("tags", [])

        data_json = json.dumps(data, ensure_ascii=False)

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO conversations (id, model_name, data, created_at, last_active, turn_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        model_name = excluded.model_name,
                        data = excluded.data,
                        last_active = excluded.last_active,
                        turn_count = excluded.turn_count
                    """,
                    (conversation_id, model_name, data_json, created_at, last_active, turn_count),
                )
                # Replace tags
                conn.execute(
                    "DELETE FROM conversation_tags WHERE conversation_id = ?",
                    (conversation_id,),
                )
                if tags:
                    conn.executemany(
                        "INSERT INTO conversation_tags (conversation_id, tag) VALUES (?, ?)",
                        [(conversation_id, t) for t in tags],
                    )
                conn.commit()
            finally:
                conn.close()

    def load(self, conversation_id: str) -> dict[str, Any] | None:
        """Load a conversation by ID.  Returns ``None`` if not found."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT data FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if row is None:
                    return None
                return json.loads(row["data"])
            finally:
                conn.close()

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation.  Returns *True* if it existed."""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (conversation_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # Search / listing
    # ------------------------------------------------------------------ #

    def find_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Return summary dicts for all conversations with the given tag."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT c.id, c.model_name, c.created_at, c.last_active, c.turn_count
                    FROM conversations c
                    INNER JOIN conversation_tags ct ON c.id = ct.conversation_id
                    WHERE ct.tag = ?
                    ORDER BY c.last_active DESC
                    """,
                    (tag,),
                ).fetchall()
                return [self._row_to_summary(conn, r) for r in rows]
            finally:
                conn.close()

    def find_by_id(self, conversation_id: str) -> dict[str, Any] | None:
        """Return a summary dict (with tags) for a conversation, or ``None``."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, model_name, created_at, last_active, turn_count FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if row is None:
                    return None
                return self._row_to_summary(conn, row)
            finally:
                conn.close()

    def list_all(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return summary dicts ordered by ``last_active`` descending."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id, model_name, created_at, last_active, turn_count
                    FROM conversations
                    ORDER BY last_active DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
                return [self._row_to_summary(conn, r) for r in rows]
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    @staticmethod
    def _row_to_summary(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        """Build a summary dict from a DB row, including tags."""
        cid = row["id"]
        tag_rows = conn.execute(
            "SELECT tag FROM conversation_tags WHERE conversation_id = ?",
            (cid,),
        ).fetchall()
        return {
            "id": cid,
            "model_name": row["model_name"],
            "created_at": row["created_at"],
            "last_active": row["last_active"],
            "turn_count": row["turn_count"],
            "tags": [tr["tag"] for tr in tag_rows],
        }
