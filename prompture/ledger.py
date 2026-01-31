"""Persistent model usage ledger — tracks which LLM models have been used.

Stores per-model usage stats (call count, tokens, cost, timestamps) in a
SQLite database at ``~/.prompture/usage/model_ledger.db``.  The public
convenience functions are fire-and-forget: they never raise exceptions so
they cannot break existing extraction/conversation flows.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("prompture.ledger")

_DEFAULT_DB_DIR = Path.home() / ".prompture" / "usage"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "model_ledger.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_usage (
    model_name  TEXT NOT NULL,
    api_key_hash TEXT NOT NULL,
    use_count   INTEGER NOT NULL DEFAULT 1,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost  REAL NOT NULL DEFAULT 0.0,
    first_used  TEXT NOT NULL,
    last_used   TEXT NOT NULL,
    last_status TEXT NOT NULL DEFAULT 'success',
    PRIMARY KEY (model_name, api_key_hash)
);
"""


class ModelUsageLedger:
    """SQLite-backed model usage tracker.

    Thread-safe via an internal :class:`threading.Lock`.

    Args:
        db_path: Path to the SQLite database file.  Defaults to
            ``~/.prompture/usage/model_ledger.db``.
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
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_usage(
        self,
        model_name: str,
        *,
        api_key_hash: str = "",
        tokens: int = 0,
        cost: float = 0.0,
        status: str = "success",
    ) -> None:
        """Record a model usage event (upsert).

        On conflict the row's counters are incremented and ``last_used``
        is updated.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO model_usage
                        (model_name, api_key_hash, use_count, total_tokens, total_cost,
                         first_used, last_used, last_status)
                    VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT(model_name, api_key_hash) DO UPDATE SET
                        use_count    = use_count + 1,
                        total_tokens = total_tokens + excluded.total_tokens,
                        total_cost   = total_cost + excluded.total_cost,
                        last_used    = excluded.last_used,
                        last_status  = excluded.last_status
                    """,
                    (model_name, api_key_hash, tokens, cost, now, now, status),
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_model_stats(self, model_name: str, api_key_hash: str = "") -> dict[str, Any] | None:
        """Return stats for a specific model + key combination, or ``None``."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM model_usage WHERE model_name = ? AND api_key_hash = ?",
                    (model_name, api_key_hash),
                ).fetchone()
                if row is None:
                    return None
                return dict(row)
            finally:
                conn.close()

    def get_verified_models(self) -> set[str]:
        """Return model names that have at least one successful usage."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT DISTINCT model_name FROM model_usage WHERE last_status = 'success'"
                ).fetchall()
                return {r["model_name"] for r in rows}
            finally:
                conn.close()

    def get_recently_used(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent model usage rows ordered by ``last_used`` descending."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM model_usage ORDER BY last_used DESC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_all_stats(self) -> list[dict[str, Any]]:
        """Return all usage rows."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT * FROM model_usage ORDER BY last_used DESC").fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_ledger: ModelUsageLedger | None = None
_ledger_lock = threading.Lock()


def _get_ledger() -> ModelUsageLedger:
    """Return (and lazily create) the module-level singleton ledger."""
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = ModelUsageLedger()
    return _ledger


# ------------------------------------------------------------------
# Public convenience functions (fire-and-forget)
# ------------------------------------------------------------------


def record_model_usage(
    model_name: str,
    *,
    api_key_hash: str = "",
    tokens: int = 0,
    cost: float = 0.0,
    status: str = "success",
) -> None:
    """Record a model usage event.  Never raises — all exceptions are swallowed."""
    try:
        _get_ledger().record_usage(
            model_name,
            api_key_hash=api_key_hash,
            tokens=tokens,
            cost=cost,
            status=status,
        )
    except Exception:
        logger.debug("Failed to record model usage for %s", model_name, exc_info=True)


def get_recently_used_models(limit: int = 10) -> list[dict[str, Any]]:
    """Return recently used models.  Returns empty list on error."""
    try:
        return _get_ledger().get_recently_used(limit)
    except Exception:
        logger.debug("Failed to get recently used models", exc_info=True)
        return []


# ------------------------------------------------------------------
# API key hash helper
# ------------------------------------------------------------------

_LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio", "local_http", "airllm"})


def _resolve_api_key_hash(model_name: str) -> str:
    """Derive an 8-char hex hash of the API key for the given model's provider.

    Local providers (ollama, lmstudio, etc.) return ``""``.
    """
    try:
        provider = model_name.split("/", 1)[0].lower() if "/" in model_name else model_name.lower()
        if provider in _LOCAL_PROVIDERS:
            return ""

        from .settings import settings

        key_map: dict[str, str | None] = {
            "openai": settings.openai_api_key,
            "claude": settings.claude_api_key,
            "google": settings.google_api_key,
            "groq": settings.groq_api_key,
            "grok": settings.grok_api_key,
            "openrouter": settings.openrouter_api_key,
            "azure": settings.azure_api_key,
            "huggingface": settings.hf_token,
        }
        api_key = key_map.get(provider)
        if not api_key:
            return ""
        return hashlib.sha256(api_key.encode()).hexdigest()[:8]
    except Exception:
        return ""
