"""Storage backends for run checkpoints.

Three implementations ship out of the box:

- :class:`InMemoryCheckpointStore` — process-local, no dependencies.
- :class:`FileCheckpointStore` — one JSON file per ``run_id`` in a
  directory of your choosing.  Atomic writes via temp-file + rename.
- :class:`SQLiteCheckpointStore` — single-file database, indexed by
  ``run_id``, mirrors the persistence patterns used elsewhere in the
  codebase.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .types import Checkpoint

_DEFAULT_DB_DIR = Path.home() / ".prompture" / "checkpoints"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "checkpoints.db"


@runtime_checkable
class CheckpointStore(Protocol):
    """Persistence contract for :class:`Checkpoint`.

    Backends must be thread-safe — the same ``run_id`` may be saved
    concurrently from multiple threads (an LLM call writing progress
    while a UI thread reads it).
    """

    def save(self, checkpoint: Checkpoint) -> None:
        """Upsert *checkpoint* by ``run_id``."""
        ...

    def load(self, run_id: str) -> Checkpoint | None:
        """Return the latest :class:`Checkpoint` for *run_id*, or ``None``."""
        ...

    def list_runs(self, *, status: str | None = None) -> list[Checkpoint]:
        """List the latest checkpoint for every known ``run_id``.

        Filter by *status* when set.
        """
        ...

    def delete(self, run_id: str) -> bool:
        """Drop the checkpoint for *run_id*.  Returns whether one existed."""
        ...


# ----------------------------------------------------------------------
# In-memory
# ----------------------------------------------------------------------


class InMemoryCheckpointStore:
    """Thread-safe in-process :class:`CheckpointStore`."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_run: dict[str, Checkpoint] = {}

    def save(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            self._by_run[checkpoint.run_id] = checkpoint

    def load(self, run_id: str) -> Checkpoint | None:
        with self._lock:
            return self._by_run.get(run_id)

    def list_runs(self, *, status: str | None = None) -> list[Checkpoint]:
        with self._lock:
            records = list(self._by_run.values())
        if status is not None:
            records = [c for c in records if c.status == status]
        records.sort(key=lambda c: c.ts, reverse=True)
        return records

    def delete(self, run_id: str) -> bool:
        with self._lock:
            return self._by_run.pop(run_id, None) is not None


# ----------------------------------------------------------------------
# File-based (JSON-per-run)
# ----------------------------------------------------------------------


class FileCheckpointStore:
    """Persist checkpoints as JSON files in a directory.

    File layout: ``<directory>/<run_id>.json``.  Writes are atomic via
    ``tempfile + os.replace`` so a crash mid-write never leaves a
    truncated file.

    Args:
        directory: Where to store JSON files.  Created if missing.

    Example::

        store = FileCheckpointStore("./.runs")
        store.save(checkpoint)
        cp = store.load("run-42")
    """

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path_for(self, run_id: str) -> Path:
        # Defensive: refuse path-traversal in run ids.
        safe = run_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        path = self._path_for(checkpoint.run_id)
        payload = json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2)
        with self._lock:
            # Atomic replace: write to a sibling temp file, then rename.
            fd, tmp_name = tempfile.mkstemp(prefix=".chkpt-", dir=str(self._dir))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fp:
                    fp.write(payload)
                os.replace(tmp_name, path)
            except Exception:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
                raise

    def load(self, run_id: str) -> Checkpoint | None:
        path = self._path_for(run_id)
        with self._lock:
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return Checkpoint.from_dict(data)

    def list_runs(self, *, status: str | None = None) -> list[Checkpoint]:
        with self._lock:
            paths = sorted(self._dir.glob("*.json"))
        out: list[Checkpoint] = []
        for p in paths:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cp = Checkpoint.from_dict(data)
            if status is None or cp.status == status:
                out.append(cp)
        out.sort(key=lambda c: c.ts, reverse=True)
        return out

    def delete(self, run_id: str) -> bool:
        path = self._path_for(run_id)
        with self._lock:
            if path.exists():
                path.unlink()
                return True
            return False


# ----------------------------------------------------------------------
# SQLite
# ----------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id TEXT PRIMARY KEY,
    id TEXT NOT NULL,
    agent_name TEXT,
    model TEXT,
    status TEXT NOT NULL,
    iteration INTEGER NOT NULL DEFAULT 0,
    ts REAL NOT NULL,
    data TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON checkpoints(status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_ts ON checkpoints(ts);
"""


class SQLiteCheckpointStore:
    """SQLite-backed :class:`CheckpointStore`.

    Args:
        db_path: Override the default DB location
            (``~/.prompture/checkpoints/checkpoints.db``).
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

    def save(self, checkpoint: Checkpoint) -> None:
        data_json = json.dumps(checkpoint.to_dict(), ensure_ascii=False)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO checkpoints
                        (run_id, id, agent_name, model, status, iteration, ts, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        id = excluded.id,
                        agent_name = excluded.agent_name,
                        model = excluded.model,
                        status = excluded.status,
                        iteration = excluded.iteration,
                        ts = excluded.ts,
                        data = excluded.data
                    """,
                    (
                        checkpoint.run_id,
                        checkpoint.id,
                        checkpoint.agent_name,
                        checkpoint.model,
                        checkpoint.status,
                        checkpoint.iteration,
                        checkpoint.ts,
                        data_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def load(self, run_id: str) -> Checkpoint | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT data FROM checkpoints WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        return Checkpoint.from_dict(json.loads(row["data"]))

    def list_runs(self, *, status: str | None = None) -> list[Checkpoint]:
        sql = "SELECT data FROM checkpoints"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY ts DESC"
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        return [Checkpoint.from_dict(json.loads(r["data"])) for r in rows]

    def delete(self, run_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE run_id = ?",
                    (run_id,),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()
