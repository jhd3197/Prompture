"""The local companion's data: Prompture's own usage ledger.

Every Prompture driver call on this machine is recorded in
``~/.prompture/usage/usage.db`` (see :mod:`prompture.infra.tracker`). The
:class:`LedgerSource` reads that file — it never writes to it — to answer the
companion views, and :meth:`LedgerSource.tail` turns newly written rows into
``request.finished`` live events.

The ledger only knows about calls once they finish, so the local companion has
no "running now" view; prompture-hub, which sits in the request path, does.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..infra.tracker import PROJECT_TAG
from .live import LiveBus
from .summary import UsageRow, window_start

DEFAULT_LEDGER = Path.home() / ".prompture" / "usage" / "usage.db"
#: How many recent rows are scanned for each model's latest rate-limit snapshot.
RATE_LIMIT_SCAN = 500


def _project(tags: str | None) -> str | None:
    for tag in json.loads(tags) if tags else []:
        if isinstance(tag, str) and tag.startswith(PROJECT_TAG):
            return tag[len(PROJECT_TAG) :] or None
    return None


def _meta(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw) if raw else {}
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


class LedgerSource:
    """Read-only view of a Prompture usage ledger."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_LEDGER

    def _connect(self) -> sqlite3.Connection | None:
        if not self.db_path.exists():
            return None
        conn = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        conn = self._connect()
        if conn is None:
            return []
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:  # table not created yet
            return []
        finally:
            conn.close()

    @staticmethod
    def _row(r: sqlite3.Row) -> UsageRow:
        meta = _meta(r["metadata"])
        route = meta.get("route") if isinstance(meta.get("route"), dict) else {}
        return UsageRow(
            model=r["model_name"] or "",
            cost_usd=float(r["cost"] or 0.0),
            tokens=int(r["total_tokens"] or 0),
            status="ok" if r["status"] == "success" else (r["status"] or "ok"),
            served_by=route.get("served_by"),
            project=_project(r["tags"]),
        )

    def rows(self, period: str = "day", now: datetime | None = None) -> list[UsageRow]:
        """Calls in the current UTC ``period`` window."""
        start = window_start(period, now).isoformat()
        return [
            self._row(r)
            for r in self._query(
                "SELECT model_name, cost, total_tokens, status, tags, metadata FROM usage_events WHERE timestamp >= ?",
                (start,),
            )
        ]

    def daily(self, since: datetime, offset_minutes: int = 0) -> dict[str, dict[str, Any]]:
        """Calls, tokens and cost per local day since ``since``.

        ``offset_minutes`` is the reader's UTC offset as JavaScript reports it
        (minutes *behind* UTC, e.g. 240 for UTC-4), so days break at local midnight.
        """
        rows = self._query(
            "SELECT date(timestamp, ?) AS day, COUNT(*) AS n, SUM(total_tokens) AS tokens, SUM(cost) AS cost "
            "FROM usage_events WHERE timestamp >= ? GROUP BY day",
            (f"{-offset_minutes:+d} minutes", since.isoformat()),
        )
        return {
            r["day"]: {"requests": int(r["n"]), "tokens": int(r["tokens"] or 0), "cost_usd": float(r["cost"] or 0.0)}
            for r in rows
            if r["day"]
        }

    def rate_limits(self) -> dict[str, dict[str, Any]]:
        """Latest rate-limit snapshot per model, from recent calls' metadata."""
        out: dict[str, dict[str, Any]] = {}
        for r in self._query(
            "SELECT model_name, metadata FROM usage_events WHERE metadata LIKE '%rate_limits%' "
            "ORDER BY rowid DESC LIMIT ?",
            (RATE_LIMIT_SCAN,),
        ):
            model = r["model_name"]
            data = _meta(r["metadata"]).get("rate_limits")
            if model and model not in out and isinstance(data, dict):
                out[model] = data
        return out

    def last_rowid(self) -> int:
        rows = self._query("SELECT MAX(rowid) AS last FROM usage_events")
        return int(rows[0]["last"] or 0) if rows else 0

    def events_after(self, rowid: int) -> list[tuple[int, dict[str, Any]]]:
        """``request.finished`` payloads for rows newer than *rowid*."""
        out = []
        for r in self._query(
            "SELECT rowid, id, timestamp, model_name, cost, prompt_tokens, completion_tokens, total_tokens, "
            "elapsed_ms, status, error_message, tags, metadata FROM usage_events WHERE rowid > ? ORDER BY rowid",
            (rowid,),
        ):
            meta = _meta(r["metadata"])
            route = meta.get("route") if isinstance(meta.get("route"), dict) else {}
            attempts = sum(1 for a in route.get("attempts", []) if a.get("outcome") in ("ok", "error")) or 1
            out.append(
                (
                    int(r["rowid"]),
                    {
                        "request_id": r["id"],
                        "ts": r["timestamp"],
                        "key_id": None,
                        "model": r["model_name"],
                        "served_by": route.get("served_by"),
                        "project": _project(r["tags"]),
                        "status": "ok" if r["status"] == "success" else r["status"],
                        "error": (r["error_message"] or "")[:200] or None,
                        "prompt_tokens": r["prompt_tokens"],
                        "completion_tokens": r["completion_tokens"],
                        "cost_usd": r["cost"],
                        "latency_ms": int(r["elapsed_ms"] or 0),
                        "attempts": attempts,
                        "fallback": attempts > 1,
                        "deprioritized": route.get("deprioritized") or [],
                    },
                )
            )
        return out

    def tail(self, bus: LiveBus, stop: threading.Event, interval: float = 1.0) -> None:
        """Publish a ``request.finished`` event for every row written from now on."""
        last = self.last_rowid()
        while not stop.wait(interval):
            for rowid, payload in self.events_after(last):
                bus.publish("request.finished", payload)
                last = rowid
