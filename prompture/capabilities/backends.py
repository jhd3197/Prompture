"""Ordered backend chains with classified failover.

A capability (search, fetch, transcription, ...) holds an ordered list of
:class:`Backend` objects. :class:`BackendChain` tries them in order, skips the
unavailable ones, and on failure asks
:func:`prompture.resilience.classify_error` what to do:

* ``RETRY`` (timeouts, 5xx, connection errors) — retry the same backend once,
  then move on.
* ``FATAL`` (e.g. an unsafe URL) — re-raise the original error; every backend
  would get the same input. The attempts so far are attached as ``exc.attempts``.
* anything else (auth, quota, rate limit, challenge page, unsupported) — move
  on to the next backend immediately.

Every result carries a route record ``{served_by, fallback, attempts[]}``.
Reordering is a list change or an env override (``PROMPTURE_SEARCH_PROVIDERS=
"brave,exa_mcp"``); unknown names in an override are ignored and backends the
override does not mention keep their place after the named ones, so a stale
value can never hide a working backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from ..resilience.errors import ErrorAction, classify_error
from ..security.redaction import scrub_secrets
from .errors import AllBackendsFailedError, BackendUnavailableError
from .health import HealthStatus

T = TypeVar("T")


@runtime_checkable
class Backend(Protocol):
    """One way of providing a capability.

    Attributes:
        name: Stable identifier used in overrides and route records.
        requires: Human-readable requirements (env vars, extras, binaries).
    """

    name: str
    requires: tuple[str, ...]

    def available(self) -> bool:
        """Cheap, offline check that the backend is configured / installed."""
        ...

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Do the work. Raise on failure; the chain classifies the error."""
        ...

    def check(self, live: bool = False) -> HealthStatus:
        """Health row for doctor. Offline unless *live*."""
        ...


class BaseBackend:
    """Convenience base: subclass and implement :meth:`run` (and :meth:`available`)."""

    name: str = "backend"
    requires: tuple[str, ...] = ()
    keyless: bool = False
    category: str = "tools"

    def available(self) -> bool:
        return True

    def run(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def unavailable_hint(self) -> str | None:
        """Fix hint shown when :meth:`available` is false."""
        if self.requires:
            return "Requires " + ", ".join(self.requires)
        return None

    def live_check(self) -> None:
        """Run one real, cheap request; raise on failure. Override per backend."""

    def check(self, live: bool = False) -> HealthStatus:
        if not self.available():
            return HealthStatus(
                self.name,
                "unconfigured",
                category=self.category,
                message=f"{self.name} is not configured",
                fix_hint=self.unavailable_hint(),
            )
        if live:
            start = time.monotonic()
            try:
                self.live_check()
            except Exception as exc:
                info = classify_error(exc)
                return HealthStatus(
                    self.name,
                    "error",
                    category=self.category,
                    active_backend=self.name,
                    message=f"live check failed ({info.category}): {exc}",
                    details={"action": info.action.value},
                )
            return HealthStatus(
                self.name,
                "ok",
                category=self.category,
                active_backend=self.name,
                message=f"live check passed in {int((time.monotonic() - start) * 1000)} ms",
            )
        return HealthStatus(self.name, "ok", category=self.category, active_backend=self.name, message="configured")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


@dataclass
class ChainResult(Generic[T]):
    """Value produced by a chain plus the route that produced it."""

    value: T
    served_by: str
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def fallback(self) -> bool:
        """``True`` when a backend before the serving one was tried and failed."""
        return any(a["status"] == "error" for a in self.attempts)

    @property
    def route(self) -> dict[str, Any]:
        return {"served_by": self.served_by, "fallback": self.fallback, "attempts": list(self.attempts)}


def parse_override(value: str | None) -> list[str]:
    """Split a comma/space separated override into lowercase names."""
    if not value:
        return []
    return [part.strip().lower() for part in value.replace(";", ",").replace(" ", ",").split(",") if part.strip()]


def order_backends(backends: Sequence[Any], preferred: Iterable[str]) -> list[Any]:
    """Put *preferred* names first (in that order); keep the rest in place after them.

    Unknown names are ignored.
    """
    by_name = {b.name.lower(): b for b in backends}
    front: list[Any] = []
    for name in preferred:
        b = by_name.get(name.lower())
        if b is not None and b not in front:
            front.append(b)
    return front + [b for b in backends if b not in front]


class BackendChain(Generic[T]):
    """Try backends in order with classified failover.

    Args:
        backends: Candidate backends in default priority order.
        override_env: Env var holding a preferred order (read on every call).
        name: Capability name used in errors.
        retry_delay: Pause before the single retry of a transient failure.
    """

    def __init__(
        self,
        backends: Sequence[Any],
        *,
        override_env: str | None = None,
        name: str = "capability",
        retry_delay: float = 0.25,
    ) -> None:
        self.backends = list(backends)
        self.override_env = override_env
        self.name = name
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def ordered(self, only: Iterable[str] | None = None) -> list[Any]:
        """Backends in effective order.

        ``only`` restricts the chain to the named backends in that order
        (unknown names ignored); otherwise the env override reorders.
        """
        if only is not None:
            names = [n.lower() for n in only]
            by_name = {b.name.lower(): b for b in self.backends}
            return [by_name[n] for n in names if n in by_name]
        preferred: list[str] = []
        if self.override_env:
            from ..infra.credentials import get_config_value

            preferred = parse_override(get_config_value(self.override_env))
        return order_backends(self.backends, preferred)

    def get(self, name: str) -> Any | None:
        for b in self.backends:
            if b.name.lower() == name.lower():
                return b
        return None

    def active_backend(self, only: Iterable[str] | None = None) -> Any | None:
        """First available backend — what would serve a request right now."""
        for b in self.ordered(only):
            if _available(b):
                return b
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, *args: Any, only: Iterable[str] | None = None, **kwargs: Any) -> ChainResult[T]:
        """Run the chain synchronously."""
        attempts: list[dict[str, Any]] = []
        last_error: BaseException | None = None
        for backend in self.ordered(only):
            if not _available(backend):
                attempts.append(_skipped(backend))
                continue
            tries = 0
            while True:
                tries += 1
                start = time.monotonic()
                try:
                    value = backend.run(*args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    info = classify_error(exc)
                    attempts.append(_failed(backend, exc, info, start, tries))
                    if info.action == ErrorAction.FATAL:
                        _attach_attempts(exc, attempts)
                        raise
                    if info.action == ErrorAction.RETRY and tries == 1:
                        if self.retry_delay:
                            time.sleep(self.retry_delay)
                        continue
                    break
                attempts.append(_succeeded(backend, start, tries))
                return ChainResult(value, backend.name, attempts)
        raise _all_failed(self.name, attempts, last_error) from last_error

    async def arun(self, *args: Any, only: Iterable[str] | None = None, **kwargs: Any) -> ChainResult[T]:
        """Async variant: uses ``backend.arun`` when present, else runs ``run`` in a thread."""
        attempts: list[dict[str, Any]] = []
        last_error: BaseException | None = None
        for backend in self.ordered(only):
            if not _available(backend):
                attempts.append(_skipped(backend))
                continue
            tries = 0
            while True:
                tries += 1
                start = time.monotonic()
                try:
                    arun = getattr(backend, "arun", None)
                    if arun is not None and inspect.iscoroutinefunction(arun):
                        value = await arun(*args, **kwargs)
                    else:
                        value = await asyncio.to_thread(backend.run, *args, **kwargs)
                except Exception as exc:
                    last_error = exc
                    info = classify_error(exc)
                    attempts.append(_failed(backend, exc, info, start, tries))
                    if info.action == ErrorAction.FATAL:
                        _attach_attempts(exc, attempts)
                        raise
                    if info.action == ErrorAction.RETRY and tries == 1:
                        if self.retry_delay:
                            await asyncio.sleep(self.retry_delay)
                        continue
                    break
                attempts.append(_succeeded(backend, start, tries))
                return ChainResult(value, backend.name, attempts)
        raise _all_failed(self.name, attempts, last_error) from last_error

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def check(self, live: bool = False, *, category: str = "tools") -> HealthStatus:
        """Summarize the chain: ok when any backend is usable; lists each backend."""
        rows: list[dict[str, Any]] = []
        active: str | None = None
        for b in self.ordered():
            try:
                row = b.check(live) if hasattr(b, "check") else None
            except Exception as exc:
                row = HealthStatus(b.name, "error", message=str(exc))
            if row is None:
                row = HealthStatus(b.name, "ok" if _available(b) else "unconfigured")
            rows.append({"backend": b.name, "status": row.status, "message": row.message, "fix_hint": row.fix_hint})
            if active is None and row.status == "ok":
                active = b.name
        if active is not None:
            idle = [r for r in rows if r["status"] != "ok"]
            message = f"served by {active}"
            if idle:
                message += f" ({len(idle)} other backend(s) inactive)"
            hints = [r["fix_hint"] for r in idle if r["fix_hint"]]
            return HealthStatus(
                self.name,
                "ok",
                category=category,
                active_backend=active,
                message=message,
                fix_hint=("Optional: " + hints[0]) if hints else None,
                details={"backends": rows},
            )
        hints = [r["fix_hint"] for r in rows if r["fix_hint"]]
        return HealthStatus(
            self.name,
            "broken" if any(r["status"] in ("broken", "error") for r in rows) else "unconfigured",
            category=category,
            message="no usable backend",
            fix_hint=hints[0] if hints else None,
            details={"backends": rows},
        )

    def __repr__(self) -> str:
        return f"BackendChain({self.name!r}, {[b.name for b in self.backends]})"


def _available(backend: Any) -> bool:
    try:
        return bool(backend.available())
    except Exception:
        return False


def _skipped(backend: Any) -> dict[str, Any]:
    hint = getattr(backend, "unavailable_hint", None)
    return {
        "backend": backend.name,
        "status": "skipped",
        "reason": "unavailable",
        "hint": hint() if callable(hint) else None,
    }


def _failed(backend: Any, exc: BaseException, info: Any, start: float, tries: int) -> dict[str, Any]:
    return {
        "backend": backend.name,
        "status": "error",
        "error": scrub_secrets(f"{type(exc).__name__}: {exc}")[:500],
        "category": info.category,
        "action": info.action.value,
        "status_code": info.status_code,
        "try": tries,
        "elapsed_ms": int((time.monotonic() - start) * 1000),
    }


def _succeeded(backend: Any, start: float, tries: int) -> dict[str, Any]:
    return {
        "backend": backend.name,
        "status": "ok",
        "try": tries,
        "elapsed_ms": int((time.monotonic() - start) * 1000),
    }


def _attach_attempts(exc: BaseException, attempts: list[dict[str, Any]]) -> None:
    """Record the route on a fatal error so callers can still report it."""
    with contextlib.suppress(AttributeError):  # exotic exception types
        exc.attempts = list(attempts)  # type: ignore[attr-defined]


def _all_failed(name: str, attempts: list[dict[str, Any]], last: BaseException | None) -> AllBackendsFailedError:
    if not attempts:
        return AllBackendsFailedError(f"{name}: no backends configured", attempts=attempts, last_error=last)
    if all(a["status"] == "skipped" for a in attempts):
        hints = [a["hint"] for a in attempts if a.get("hint")]
        msg = f"{name}: no backend is available"
        if hints:
            msg += f" — {hints[0]}"
        return AllBackendsFailedError(msg, attempts=attempts, last_error=BackendUnavailableError(msg))
    tried = ", ".join(f"{a['backend']} ({a.get('category', a['status'])})" for a in attempts if a["status"] == "error")
    return AllBackendsFailedError(f"{name}: all backends failed: {tried}", attempts=attempts, last_error=last)
