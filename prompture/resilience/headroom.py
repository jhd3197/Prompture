"""Rate-limit headroom per route target, and ordering that avoids exhausted ones.

Drivers report ``meta["rate_limits"]`` (see :mod:`prompture.infra.rate_limits`)
on every response that carried rate-limit headers, and 429 errors carry the
same headers. :class:`HeadroomTracker` keeps the latest snapshot per target
label (``provider/model`` or ``provider/model#keyid``). :func:`partition_by_headroom`
moves targets that are nearly out of a still-open window to the back of the
candidate list, so a route prefers a target that can take the request instead
of spending an attempt on a predictable 429.

Nothing is ever skipped: a deprioritized target is still tried if everything
ahead of it fails, and when every target is low the order is left alone.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from ..infra.rate_limits import LimitSnapshot, parse_rate_limit_headers

T = TypeVar("T")


class HeadroomTracker:
    """Thread-safe latest :class:`LimitSnapshot` per target label."""

    def __init__(self, clock: Callable[[], float] = time.time, max_age: float = 120.0) -> None:
        self._clock = clock
        self.max_age = max_age
        self._lock = threading.Lock()
        self._snapshots: dict[str, LimitSnapshot] = {}

    def record(self, label: str, snapshot: LimitSnapshot | None) -> None:
        if snapshot is None:
            return
        with self._lock:
            self._snapshots[label] = snapshot

    def get(self, label: str) -> LimitSnapshot | None:
        with self._lock:
            return self._snapshots.get(label)

    def headroom(self, label: str) -> tuple[float | None, str | None]:
        """Current ``(headroom, tightest window)`` for *label*, or ``(None, None)`` if unknown."""
        snapshot = self.get(label)
        if snapshot is None:
            return None, None
        return snapshot.current_headroom(self._clock(), max_age=self.max_age)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Every known target's latest limits, JSON-ready (for dashboards and gateways)."""
        now = self._clock()
        with self._lock:
            items = list(self._snapshots.items())
        out: dict[str, dict[str, Any]] = {}
        for label, snap in items:
            data = snap.to_dict()
            data["current_headroom"], data["current_window"] = snap.current_headroom(now, max_age=self.max_age)
            out[label] = data
        return out

    def reset(self) -> None:
        with self._lock:
            self._snapshots.clear()


_default_tracker = HeadroomTracker()


def get_headroom_tracker() -> HeadroomTracker:
    """Process-wide tracker shared by resilient drivers by default."""
    return _default_tracker


def limits_in(obj: Any) -> LimitSnapshot | None:
    """Pull a rate-limit snapshot out of a driver result, stream event or exception."""
    if isinstance(obj, BaseException):
        from .errors import _headers_of

        for exc in _exception_chain(obj):
            snapshot = parse_rate_limit_headers(_headers_of(exc))
            if snapshot is not None:
                return snapshot
        return None
    data: Any = None
    if isinstance(obj, dict):
        meta = obj.get("meta")
        data = meta.get("rate_limits") if isinstance(meta, dict) else None
    else:
        usage = getattr(obj, "usage", None)
        if isinstance(usage, dict):
            data = usage.get("rate_limits")
    if isinstance(data, dict):
        return LimitSnapshot.from_dict(data)
    return None


def _exception_chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen and len(seen) < 8:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def partition_by_headroom(
    items: Sequence[T],
    *,
    label: Callable[[T], str],
    tracker: HeadroomTracker,
    min_headroom: float | None,
) -> tuple[list[T], list[dict[str, Any]]]:
    """Stable-move low-headroom items to the back.

    Returns ``(ordered, notes)`` where each note names a moved item's label,
    its headroom and the window that is running out.
    """
    items = list(items)
    if min_headroom is None or len(items) < 2:
        return items, []
    ready: list[T] = []
    low: list[T] = []
    notes: list[dict[str, Any]] = []
    for item in items:
        fraction, window = tracker.headroom(label(item))
        if fraction is not None and fraction < min_headroom:
            low.append(item)
            notes.append({"target": label(item), "headroom": round(fraction, 4), "window": window})
        else:
            ready.append(item)
    if not ready or not low:
        return items, []
    return ready + low, notes
