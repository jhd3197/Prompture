"""Provider rate-limit snapshots read from documented response headers.

Most providers report how much of each rate-limit window is left on *every*
response, not only on a 429. This module normalizes those headers into a
:class:`LimitSnapshot` so callers (the resilience router, a gateway, a
dashboard) can see remaining headroom before a request fails.

Recognized header families (verified against vendor docs, 2026-09):

- ``x-ratelimit-{limit,remaining,reset}-<window>``: OpenAI and Groq.
  Windows are ``requests``, ``tokens``, and OpenAI's ``project-tokens``.
  Reset values are durations such as ``"1s"``, ``"6m0s"`` or ``"2m59.56s"``.
- ``anthropic-ratelimit-<window>-{limit,remaining,reset}``: Anthropic.
  Windows are ``requests``, ``tokens``, ``input-tokens`` and ``output-tokens``.
  Resets are RFC 3339 timestamps.
- ``anthropic-priority-<window>-{limit,remaining,reset}``: Anthropic
  Priority Tier. These are reported as ``priority_<window>``.

Providers that send none of these yield ``None``, never an estimate.

SDK-based drivers expose headers in two ways. Streaming calls carry the
``httpx.Response`` on the stream object (:func:`limits_from_response`).
Non-streaming calls return a parsed model without headers, so the driver
attaches a response hook to the SDK's own httpx client
(:func:`attach_rate_limit_hook`) and wraps the call in
:func:`capture_rate_limits`.
"""

from __future__ import annotations

import contextvars
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

__all__ = [
    "LimitSnapshot",
    "LimitWindow",
    "RateLimitCapture",
    "add_rate_limits",
    "attach_rate_limit_hook",
    "capture_rate_limits",
    "limits_from_response",
    "parse_rate_limit_headers",
]

_OPENAI_STYLE_RE = re.compile(r"^x-ratelimit-(limit|remaining|reset)-([a-z0-9-]+)$")
_ANTHROPIC_STYLE_RE = re.compile(r"^anthropic-(ratelimit|priority)-([a-z0-9-]+)-(limit|remaining|reset)$")


@dataclass(frozen=True)
class LimitWindow:
    """One rate-limit window as reported by the provider."""

    limit: int | None = None
    remaining: int | None = None
    resets_at: float | None = None
    """Unix timestamp when the window is fully replenished, if reported."""

    @property
    def fraction_remaining(self) -> float | None:
        """``remaining / limit`` clamped to ``[0, 1]``, or ``None`` if either is missing."""
        if self.limit is None or self.remaining is None or self.limit <= 0:
            return None
        return max(0.0, min(1.0, self.remaining / self.limit))

    def to_dict(self) -> dict[str, Any]:
        return {"limit": self.limit, "remaining": self.remaining, "resets_at": self.resets_at}


@dataclass(frozen=True)
class LimitSnapshot:
    """Every rate-limit window one response reported, keyed by window name."""

    windows: dict[str, LimitWindow] = field(default_factory=dict)
    observed_at: float = field(default_factory=time.time)
    source: str = "headers"

    @property
    def headroom(self) -> float | None:
        """The tightest ``fraction_remaining`` across windows, or ``None`` if none is known."""
        fractions = [f for w in self.windows.values() if (f := w.fraction_remaining) is not None]
        return min(fractions) if fractions else None

    @property
    def tightest_window(self) -> str | None:
        """Name of the window with the least headroom."""
        known = {name: f for name, w in self.windows.items() if (f := w.fraction_remaining) is not None}
        return min(known, key=known.__getitem__) if known else None

    def current_headroom(self, now: float | None = None, *, max_age: float = 120.0) -> tuple[float | None, str | None]:
        """``(headroom, window)`` counting only windows that still apply at *now*.

        A window whose ``resets_at`` has passed has refilled, so it no longer
        constrains anything. A window without a reset time is trusted for
        *max_age* seconds after the snapshot was taken.
        """
        now = time.time() if now is None else now
        best: tuple[float | None, str | None] = (None, None)
        for name, window in self.windows.items():
            fraction = window.fraction_remaining
            if fraction is None:
                continue
            if window.resets_at is not None:
                if window.resets_at <= now:
                    continue
            elif now - self.observed_at > max_age:
                continue
            if best[0] is None or fraction < best[0]:
                best = (fraction, name)
        return best

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "observed_at": self.observed_at,
            "headroom": self.headroom,
            "tightest_window": self.tightest_window,
            "windows": {name: w.to_dict() for name, w in self.windows.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LimitSnapshot:
        windows = {
            name: LimitWindow(
                limit=w.get("limit"),
                remaining=w.get("remaining"),
                resets_at=w.get("resets_at"),
            )
            for name, w in (data.get("windows") or {}).items()
        }
        return cls(
            windows=windows,
            observed_at=float(data.get("observed_at") or time.time()),
            source=str(data.get("source") or "headers"),
        )


def _to_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_reset(value: str, now: float) -> float | None:
    """Parse a reset value into an absolute Unix timestamp."""
    value = value.strip()
    if not value:
        return None
    if "T" in value:  # RFC 3339 timestamp
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    try:
        return now + max(0.0, float(value))
    except ValueError:
        pass
    from ..resilience.errors import parse_duration

    seconds = parse_duration(value)
    return now + seconds if seconds is not None else None


def _iter_headers(headers: Any) -> Iterator[tuple[str, str]]:
    try:
        items = headers.items()
    except AttributeError:
        return
    for key, value in items:
        if isinstance(key, str) and isinstance(value, str):
            yield key.lower(), value


def parse_rate_limit_headers(headers: Any, *, now: float | None = None) -> LimitSnapshot | None:
    """Build a :class:`LimitSnapshot` from response headers.

    Accepts any mapping-like object with ``items()`` (``httpx.Headers``,
    ``requests`` case-insensitive dicts, plain dicts). Returns ``None`` when no
    recognized rate-limit header is present.
    """
    if headers is None:
        return None
    now = time.time() if now is None else now
    fields: dict[str, dict[str, Any]] = {}
    for key, value in _iter_headers(headers):
        if m := _OPENAI_STYLE_RE.match(key):
            kind, window = m.group(1), m.group(2)
        elif m := _ANTHROPIC_STYLE_RE.match(key):
            family, window, kind = m.group(1), m.group(2), m.group(3)
            if family == "priority":
                window = f"priority-{window}"
        else:
            continue
        parsed = _to_reset(value, now) if kind == "reset" else _to_int(value)
        if parsed is None:
            continue
        name = window.replace("-", "_")
        fields.setdefault(name, {})["resets_at" if kind == "reset" else kind] = parsed
    if not fields:
        return None
    return LimitSnapshot(windows={name: LimitWindow(**f) for name, f in fields.items()}, observed_at=now)


def limits_from_response(response: Any) -> LimitSnapshot | None:
    """Snapshot from an ``httpx.Response`` or ``requests.Response``; ``None`` for anything else."""
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "items"):
        return None
    try:
        return parse_rate_limit_headers(headers)
    except TypeError:  # a headers-like object whose items() is not iterable
        return None


def add_rate_limits(meta: dict[str, Any], snapshot: LimitSnapshot | None) -> dict[str, Any]:
    """Set ``meta["rate_limits"]`` when a snapshot exists; leave ``meta`` untouched otherwise."""
    if snapshot is not None:
        meta["rate_limits"] = snapshot.to_dict()
    return meta


# ---------------------------------------------------------------------------
# Capturing headers from SDK clients
# ---------------------------------------------------------------------------


@dataclass
class RateLimitCapture:
    """Holds the last snapshot seen while a :func:`capture_rate_limits` block is active."""

    snapshot: LimitSnapshot | None = None


_active_capture: contextvars.ContextVar[RateLimitCapture | None] = contextvars.ContextVar(
    "prompture_rate_limit_capture", default=None
)


@contextmanager
def capture_rate_limits() -> Iterator[RateLimitCapture]:
    """Collect rate-limit headers from hooked SDK calls made inside the block.

    If the SDK retries, the last response wins, which is the one that
    produced the result.
    """
    capture = RateLimitCapture()
    token = _active_capture.set(capture)
    try:
        yield capture
    finally:
        _active_capture.reset(token)


def _record(response: Any) -> None:
    capture = _active_capture.get()
    if capture is None:
        return
    snapshot = limits_from_response(response)
    if snapshot is not None:
        capture.snapshot = snapshot


def _sync_hook(response: Any) -> None:
    _record(response)


async def _async_hook(response: Any) -> None:
    _record(response)


def attach_rate_limit_hook(sdk_client: Any) -> None:
    """Add the header-capturing response hook to an OpenAI/Anthropic/Groq SDK client.

    These SDKs keep their ``httpx`` client on ``_client``. Hooking it (rather
    than passing a custom ``http_client``) keeps the SDK's own connection and
    timeout defaults and its cleanup on garbage collection. Anything that is
    not a real httpx client (a test double, a future SDK layout) is left alone.
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with every supported SDK
        return
    http = getattr(sdk_client, "_client", None)
    if isinstance(http, httpx.AsyncClient):
        hook: Any = _async_hook
    elif isinstance(http, httpx.Client):
        hook = _sync_hook
    else:
        return
    hooks = http.event_hooks
    responses = hooks.setdefault("response", [])
    if hook not in responses:
        responses.append(hook)
        http.event_hooks = hooks
