"""Aggregations shared by every companion API server.

Both prompture-hub (usage rows in its database) and the local companion
(Prompture's usage ledger) reduce their records to :class:`UsageRow` and let
this module produce the ``/v1/spend`` and ``/v1/limits`` documents, so the two
servers answer with the same shapes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

#: Bumped when a companion-facing endpoint changes incompatibly.
COMPANION_API_VERSION = 1

PERIODS = ("day", "week", "month")


def window_start(period: str, now: datetime | None = None, offset_minutes: int = 0) -> datetime:
    """First instant of the current ``day`` / ``week`` (Monday) / ``month``, as a UTC datetime.

    Windows break at midnight UTC by default. ``offset_minutes`` moves them to a
    reader's local midnight; it is minutes *behind* UTC, as JavaScript's
    ``getTimezoneOffset()`` reports it (240 for UTC-4).
    """
    shift = timedelta(minutes=offset_minutes)
    now = (now or datetime.now(timezone.utc)) - shift  # the reader's wall clock
    p = (period or "day").lower()
    if p == "week":
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    elif p == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start + shift


def window_end(period: str, now: datetime | None = None, offset_minutes: int = 0) -> datetime:
    """When the current window for ``period`` resets (exclusive end)."""
    shift = timedelta(minutes=offset_minutes)
    start = window_start(period, now, offset_minutes) - shift
    p = (period or "day").lower()
    if p == "week":
        return start + timedelta(days=7) + shift
    if p == "month":
        return (start + timedelta(days=32)).replace(day=1) + shift
    return start + timedelta(days=1) + shift


def iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class UsageRow:
    """One metered call, in the fields the companion views need."""

    model: str
    cost_usd: float = 0.0
    tokens: int = 0
    status: str = "ok"
    served_by: str | None = None
    project: str | None = None
    key_id: int | None = None


def _bucket() -> dict[str, Any]:
    return {"requests": 0, "cost_usd": 0.0, "tokens": 0, "errors": 0}


def _add(bucket: dict[str, Any], row: UsageRow) -> None:
    bucket["requests"] += 1
    bucket["cost_usd"] += row.cost_usd or 0.0
    bucket["tokens"] += row.tokens or 0
    if row.status == "error":
        bucket["errors"] += 1


def _ranked(buckets: Mapping[Any, dict[str, Any]], label: str) -> list[dict[str, Any]]:
    rows = [{label: k, **v, "cost_usd": round(v["cost_usd"], 6)} for k, v in buckets.items()]
    return sorted(rows, key=lambda r: (-r["cost_usd"], -r["requests"]))


def summarize_spend(
    rows: Iterable[UsageRow],
    period: str = "day",
    *,
    key_names: Mapping[int, str] | None = None,
    now: datetime | None = None,
    offset_minutes: int = 0,
) -> dict[str, Any]:
    """The ``/v1/spend`` document for rows already limited to the current window."""
    total = _bucket()
    by_project: dict[str | None, dict[str, Any]] = defaultdict(_bucket)
    by_key: dict[int, dict[str, Any]] = defaultdict(_bucket)
    by_model: dict[str, dict[str, Any]] = defaultdict(_bucket)
    for row in rows:
        _add(total, row)
        _add(by_project[row.project], row)
        if row.key_id is not None:
            _add(by_key[row.key_id], row)
        _add(by_model[row.served_by or row.model], row)
    keys = _ranked(by_key, "key_id")
    for item in keys:
        item["name"] = (key_names or {}).get(item["key_id"], f"key {item['key_id']}")
    total["cost_usd"] = round(total["cost_usd"], 6)
    return {
        "period": period,
        "start": iso_utc(window_start(period, now, offset_minutes)),
        "resets_at": iso_utc(window_end(period, now, offset_minutes)),
        "total": total,
        "by_project": _ranked(by_project, "project"),
        "by_key": keys,
        "by_model": _ranked(by_model, "model"),
    }


def provider_limits(snapshots: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``/v1/limits`` provider entries from ``{target: LimitSnapshot.to_dict()-like}``.

    Adds ``current_headroom`` / ``current_window`` (windows that already reset
    no longer count) when the input doesn't carry them.
    """
    from ..infra.rate_limits import LimitSnapshot

    out = []
    for target, data in sorted(snapshots.items()):
        entry = {"target": target, **data}
        if "current_headroom" not in entry:
            entry["current_headroom"], entry["current_window"] = LimitSnapshot.from_dict(data).current_headroom()
        out.append(entry)
    return out


def account_limits(max_age: float = 60.0) -> list[dict[str, Any]]:
    """Provider account balances / spend from documented endpoints (cached)."""
    from ..infra.accounts import get_account_snapshots

    return [snap.to_dict() for snap in get_account_snapshots(max_age=max_age).values()]
