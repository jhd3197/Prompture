"""I/O-free routing core shared by the sync and async resilient drivers.

:class:`RoutePlan` owns the target list, driver construction, breaker gating
and outcome bookkeeping. The drivers in :mod:`.driver` and
:mod:`.async_driver` only run the attempt loop and sleep.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .backoff import RetryPolicy
from .breaker import BreakerRegistry, get_breaker_registry, key_scope, model_scope, provider_scope
from .errors import ErrorAction, ErrorInfo
from .keys import get_key_pool, key_id

#: How long an unknown / retired model is skipped after a 404-style failure.
MODEL_NOT_FOUND_COOLDOWN = 600.0


@dataclass
class Target:
    """One concrete way to serve a request: a model plus (optionally) a specific key.

    Pass a prebuilt ``driver`` to route through an instance you configured
    yourself; otherwise the driver is built lazily from ``model``.
    """

    model: str
    api_key: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    driver: Any = None

    @property
    def provider(self) -> str:
        return self.model.split("/", 1)[0].lower()

    @property
    def key_id(self) -> str | None:
        return key_id(self.api_key) if self.api_key else None

    @property
    def label(self) -> str:
        return f"{self.model}#{self.key_id}" if self.key_id else self.model

    def scopes(self) -> list[str]:
        scopes = [provider_scope(self.provider), model_scope(self.model)]
        if self.key_id:
            scopes.append(key_scope(self.provider, self.key_id))
        return scopes


def _model_name_of(driver: Any) -> str:
    model = getattr(driver, "model", None) or type(driver).__name__
    if "/" in str(model):
        return str(model)
    provider = type(driver).__name__.removeprefix("Async").removesuffix("Driver").lower()
    return f"{provider}/{model}"


class _Group:
    """Targets expanded from one spec; rotated together for key round-robin."""

    def __init__(self, targets: list[Target]) -> None:
        self.targets = targets
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def ordered(self) -> list[Target]:
        if len(self.targets) < 2:
            return list(self.targets)
        with self._lock:
            start = next(self._counter) % len(self.targets)
        return self.targets[start:] + self.targets[:start]


class RoutePlan:
    """Targets + breakers + policy. Thread-safe; share one per resilient driver."""

    def __init__(
        self,
        targets: Sequence[Any],
        *,
        policy: RetryPolicy | None = None,
        breakers: BreakerRegistry | None = None,
        use_key_pools: bool = True,
        factory: Callable[..., Any],
    ) -> None:
        if not targets:
            raise ValueError("A resilient route needs at least one target")
        self.policy = policy or RetryPolicy()
        self.breakers = breakers or get_breaker_registry()
        self._factory = factory
        self._lock = threading.Lock()
        self.groups: list[_Group] = [_Group(self._expand(t, use_key_pools)) for t in targets]

    # -- construction --------------------------------------------------------

    @staticmethod
    def _expand(spec: Any, use_key_pools: bool) -> list[Target]:
        if isinstance(spec, Target):
            if spec.api_key is None and spec.driver is None and use_key_pools:
                pool = get_key_pool(spec.provider)
                if pool is not None:
                    return [Target(spec.model, k, dict(spec.overrides)) for k in pool.rotation()]
            return [spec]
        if isinstance(spec, str):
            return RoutePlan._expand(Target(spec), use_key_pools)
        # A ready driver instance.
        return [Target(_model_name_of(spec), getattr(spec, "api_key", None) or None, driver=spec)]

    @property
    def targets(self) -> list[Target]:
        return [t for g in self.groups for t in g.targets]

    @property
    def primary(self) -> Target:
        return self.groups[0].targets[0]

    def driver_for(self, target: Target) -> Any:
        if target.driver is None:
            with self._lock:
                if target.driver is None:
                    target.driver = self._factory(target.model, api_key=target.api_key, **target.overrides)
        return target.driver

    def peek_drivers(self) -> list[Any]:
        """Instantiate what can be instantiated (for capability flags); skip failures."""
        out = []
        for t in self.targets:
            try:
                out.append(self.driver_for(t))
            except Exception:
                continue
        return out

    # -- ordering + gating -----------------------------------------------------

    def candidates(self) -> list[Target]:
        return [t for g in self.groups for t in g.ordered()]

    def gate(self, target: Target) -> tuple[bool, float]:
        """``(allowed, seconds_until_available)`` for *target*."""
        breakers = [self.breakers.get(s) for s in target.scopes()]
        wait = max(b.available_in() for b in breakers)
        if wait > 0:
            return False, wait
        return all(b.allow() for b in breakers), 0.0

    # -- outcomes ----------------------------------------------------------------

    def succeed(self, target: Target) -> None:
        for s in target.scopes():
            self.breakers.get(s).record_success()

    def release(self, target: Target) -> None:
        for s in target.scopes():
            self.breakers.get(s).release()

    def penalize(self, target: Target, info: ErrorInfo) -> None:
        p = self.policy
        action = info.action
        if action is ErrorAction.COOLDOWN:
            scope = key_scope(target.provider, target.key_id) if target.key_id else model_scope(target.model)
            self.breakers.get(scope).cooldown(info.retry_after or p.default_cooldown, info.category)
        elif action is ErrorAction.DISABLE_KEY:
            seconds = info.retry_after if info.retry_after else p.disable_cooldown
            scope = key_scope(target.provider, target.key_id) if target.key_id else provider_scope(target.provider)
            self.breakers.get(scope).cooldown(seconds, info.category)
        elif action is ErrorAction.FAILOVER:
            if info.category == "model_not_found":
                self.breakers.get(model_scope(target.model)).cooldown(MODEL_NOT_FOUND_COOLDOWN, info.category)
            else:
                # Request-specific miss (context length, filter, feature): the
                # target itself is healthy, just hand back any probe slot.
                self.release(target)
        elif action is ErrorAction.RETRY:
            for s in target.scopes():
                self.breakers.get(s).record_failure(info.message[:200])


def attempt_record(
    target: Target,
    *,
    outcome: str,
    elapsed_ms: float | None = None,
    info: ErrorInfo | None = None,
    wait: float | None = None,
) -> dict[str, Any]:
    rec: dict[str, Any] = {"model": target.model, "outcome": outcome}
    if target.key_id:
        rec["key_id"] = target.key_id
    if elapsed_ms is not None:
        rec["elapsed_ms"] = round(elapsed_ms, 1)
    if info is not None:
        rec["error"] = info.to_dict()
    if wait is not None:
        rec["available_in"] = round(wait, 3)
    return rec


def route_summary(target: Target, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "served_by": target.model,
        "key_id": target.key_id,
        "fallback": any(a["outcome"] != "ok" for a in attempts),
        "attempts": attempts,
    }


def should_retry_same(policy: RetryPolicy, info: ErrorInfo, attempt: int, is_last: bool) -> float | None:
    """Delay before retrying the same target, or ``None`` to move on."""
    if attempt + 1 >= policy.max_attempts:
        return None
    if info.action is ErrorAction.RETRY:
        return policy.backoff(attempt, info.retry_after)
    if info.action is ErrorAction.COOLDOWN and is_last:
        # Nowhere else to go: wait out a short rate limit instead of failing.
        wait = info.retry_after if info.retry_after is not None else policy.backoff(attempt)
        if wait <= policy.max_wait:
            return wait
    return None


def failure_message(attempts: list[dict[str, Any]]) -> str:
    if not attempts:
        return "No route targets were available"
    parts = []
    for a in attempts:
        err = a.get("error") or {}
        detail = err.get("category") or a["outcome"]
        if err.get("status_code"):
            detail += f" {err['status_code']}"
        parts.append(f"{a['model']}: {detail}")
    return "All route targets failed (" + "; ".join(parts) + ")"
