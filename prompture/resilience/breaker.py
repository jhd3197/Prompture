"""Circuit breakers with lazy recovery and explicit cooldowns.

A breaker guards one *scope* — a provider (``provider:openai``), a model
(``model:openai/gpt-4o``) or a single credential (``key:openai#1a2b3c4d``).
State is recomputed when read, so there are no timers or background threads:
an OPEN breaker becomes HALF_OPEN the first time someone asks after its
recovery window has elapsed, and lets exactly one probe through.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class BreakerConfig:
    failure_threshold: int = 5
    """Consecutive failures that open the breaker."""

    recovery_timeout: float = 30.0
    """Seconds an OPEN breaker waits before allowing a probe."""


class CircuitBreaker:
    """Thread-safe breaker for a single scope."""

    def __init__(self, scope: str, config: BreakerConfig | None = None, clock: Callable[[], float] = time.monotonic):
        self.scope = scope
        self.config = config or BreakerConfig()
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False
        self._cooldown_until = 0.0
        self._cooldown_reason: str | None = None
        self.last_error: str | None = None

    # -- state -------------------------------------------------------------

    def _state_locked(self, now: float) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if now - self._opened_at >= self.config.recovery_timeout:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._state_locked(self._clock())

    def cooldown_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._cooldown_until - self._clock())

    def available_in(self) -> float:
        """Seconds until this scope would accept a call (0 = now)."""
        with self._lock:
            now = self._clock()
            wait = max(0.0, self._cooldown_until - now)
            if self._opened_at is not None:
                wait = max(wait, self._opened_at + self.config.recovery_timeout - now)
            return wait

    def allow(self) -> bool:
        """Whether a call may proceed now. Claims the probe slot when HALF_OPEN."""
        with self._lock:
            now = self._clock()
            if now < self._cooldown_until:
                return False
            state = self._state_locked(now)
            if state is BreakerState.CLOSED:
                return True
            if state is BreakerState.HALF_OPEN and not self._probe_in_flight:
                self._probe_in_flight = True
                return True
            return False

    # -- outcomes ----------------------------------------------------------

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self, error: str | None = None) -> None:
        with self._lock:
            now = self._clock()
            self.last_error = error
            if self._probe_in_flight or self._state_locked(now) is BreakerState.HALF_OPEN:
                # Failed probe: re-open for another full window.
                self._opened_at = now
                self._probe_in_flight = False
                return
            self._failures += 1
            if self._failures >= self.config.failure_threshold:
                self._opened_at = now

    def release(self) -> None:
        """Give back a HALF_OPEN probe slot without recording an outcome."""
        with self._lock:
            self._probe_in_flight = False

    def cooldown(self, seconds: float, reason: str | None = None) -> None:
        """Park this scope for *seconds* (extends, never shortens, an active cooldown)."""
        with self._lock:
            until = self._clock() + max(0.0, seconds)
            if until > self._cooldown_until:
                self._cooldown_until = until
                self._cooldown_reason = reason
            self._probe_in_flight = False

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False
            self._cooldown_until = 0.0
            self._cooldown_reason = None
            self.last_error = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            return {
                "scope": self.scope,
                "state": self._state_locked(now).value,
                "consecutive_failures": self._failures,
                "cooldown_remaining": round(max(0.0, self._cooldown_until - now), 3),
                "cooldown_reason": self._cooldown_reason if now < self._cooldown_until else None,
                "last_error": self.last_error,
            }


class BreakerRegistry:
    """Lazily creates one breaker per scope. Scope prefix picks the config."""

    def __init__(
        self,
        configs: dict[str, BreakerConfig] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._configs = {
            # A whole provider going down needs more evidence than one model.
            "provider": BreakerConfig(failure_threshold=10, recovery_timeout=30.0),
            "model": BreakerConfig(failure_threshold=5, recovery_timeout=30.0),
            "key": BreakerConfig(failure_threshold=5, recovery_timeout=60.0),
            **(configs or {}),
        }
        self._clock = clock
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, scope: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(scope)
            if breaker is None:
                kind = scope.split(":", 1)[0]
                breaker = CircuitBreaker(scope, self._configs.get(kind), clock=self._clock)
                self._breakers[scope] = breaker
            return breaker

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            breakers = list(self._breakers.values())
        return [b.snapshot() for b in breakers]

    def reset(self, scope: str | None = None) -> None:
        with self._lock:
            targets = [self._breakers[scope]] if scope and scope in self._breakers else list(self._breakers.values())
            if scope is None:
                self._breakers.clear()
        for b in targets:
            b.reset()


_default_registry = BreakerRegistry()


def get_breaker_registry() -> BreakerRegistry:
    """Process-wide registry shared by every resilient driver by default."""
    return _default_registry


def provider_scope(provider: str) -> str:
    return f"provider:{provider}"


def model_scope(model: str) -> str:
    return f"model:{model}"


def key_scope(provider: str, key_id: str) -> str:
    return f"key:{provider}#{key_id}"
