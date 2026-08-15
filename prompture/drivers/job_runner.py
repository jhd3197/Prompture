"""Shared submit→poll plumbing for long-running media job drivers.

Every aggregator / job-based media provider re-implements the same loop with a
slightly different status vocabulary (``COMPLETED`` vs ``succeeded`` vs
``success``; ``FAILED`` vs ``error`` vs ``canceled``). This module centralizes
status normalization and the poll loop so drivers only supply a ``fetch``
callable that returns the latest raw JSON.

Used by the Muapi aggregator driver; intended to back any other job-based
media driver (Replicate, Novita, …) and the gateway's async job endpoints.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ..jobs import JobStatus

__all__ = [
    "PENDING_STATES",
    "TERMINAL_DONE",
    "TERMINAL_FAIL",
    "JobFailedError",
    "async_poll_job",
    "normalize_status",
    "poll_job",
]

# Union of the terminal vocabularies seen across providers (lower-cased).
TERMINAL_DONE = {"completed", "succeeded", "success", "complete", "done", "ready"}
TERMINAL_FAIL = {"failed", "error", "errored", "canceled", "cancelled", "rejected"}
PENDING_STATES = {"pending", "queued", "in_queue", "not_started", "starting", "submitted"}


class JobFailedError(RuntimeError):
    """Raised when a polled job reaches a terminal failure state."""


def normalize_status(raw_status: Any) -> str:
    """Map a provider-specific status string onto a :class:`JobStatus` value."""
    s = str(raw_status or "").strip().lower()
    if s in TERMINAL_DONE:
        return JobStatus.COMPLETED.value
    if s in TERMINAL_FAIL:
        return JobStatus.FAILED.value
    if s in PENDING_STATES:
        return JobStatus.PENDING.value
    # in_progress / processing / running / generating / unknown → treat as running
    return JobStatus.RUNNING.value


def _status_of(raw: dict[str, Any], status_key: str) -> Any:
    return raw.get(status_key)


def poll_job(
    fetch: Callable[[], dict[str, Any]],
    *,
    interval: float = 2.0,
    timeout: float = 900.0,
    status_key: str = "status",
    status_of: Callable[[dict[str, Any]], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll ``fetch`` until the job completes; return the final raw payload.

    Raises :class:`JobFailedError` on terminal failure and :class:`TimeoutError`
    once ``timeout`` seconds elapse. ``sleep`` / ``monotonic`` are injectable so
    tests can drive the loop without real time.
    """
    get_status = status_of or (lambda raw: _status_of(raw, status_key))
    deadline = monotonic() + timeout
    while True:
        raw = fetch()
        norm = normalize_status(get_status(raw))
        if norm == JobStatus.COMPLETED.value:
            return raw
        if norm == JobStatus.FAILED.value:
            raise JobFailedError(f"Job failed: {raw.get('error') or get_status(raw) or raw}")
        if monotonic() >= deadline:
            raise TimeoutError(f"Job timed out after {timeout}s (status={get_status(raw)})")
        sleep(interval)


async def async_poll_job(
    fetch: Callable[[], Awaitable[dict[str, Any]]],
    *,
    interval: float = 2.0,
    timeout: float = 900.0,
    status_key: str = "status",
    status_of: Callable[[dict[str, Any]], Any] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Async counterpart of :func:`poll_job`."""
    get_status = status_of or (lambda raw: _status_of(raw, status_key))
    deadline = monotonic() + timeout
    while True:
        raw = await fetch()
        norm = normalize_status(get_status(raw))
        if norm == JobStatus.COMPLETED.value:
            return raw
        if norm == JobStatus.FAILED.value:
            raise JobFailedError(f"Job failed: {raw.get('error') or get_status(raw) or raw}")
        if monotonic() >= deadline:
            raise TimeoutError(f"Job timed out after {timeout}s (status={get_status(raw)})")
        await sleep(interval)
