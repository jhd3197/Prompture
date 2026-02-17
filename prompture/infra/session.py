"""Usage session tracking for Prompture.

Provides :class:`UsageSession` which accumulates token counts, costs, and
errors across multiple driver calls.  A session instance is compatible as
both an ``on_response`` and ``on_error`` callback, so you can wire it
directly into :class:`~prompture.callbacks.DriverCallbacks`.

Usage::

    from prompture import UsageSession, DriverCallbacks

    session = UsageSession()
    callbacks = DriverCallbacks(
        on_response=session.record,
        on_error=session.record_error,
    )

    # ... pass *callbacks* to your driver / conversation ...

    print(session.summary()["formatted"])
"""

from __future__ import annotations

import logging
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("prompture.session")


@dataclass
class UsageSession:
    """Accumulates usage statistics across multiple driver calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    call_count: int = 0
    errors: int = 0
    total_elapsed_ms: float = 0.0
    _elapsed_samples: list[float] = field(default_factory=list, repr=False)
    _per_model: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_cost(self) -> float:
        """Deprecated: Use ``cost`` instead."""
        warnings.warn(
            "UsageSession.total_cost is deprecated, use .cost instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cost

    @total_cost.setter
    def total_cost(self, value: float) -> None:
        """Deprecated: Use ``cost`` instead."""
        warnings.warn(
            "UsageSession.total_cost is deprecated, use .cost instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.cost = value

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record(self, response_info: dict[str, Any]) -> None:
        """Record a successful driver response.

        Compatible as an ``on_response`` callback for
        :class:`~prompture.callbacks.DriverCallbacks`.

        Args:
            response_info: Payload dict with at least ``meta`` and
                optionally ``driver`` keys.
        """
        meta = response_info.get("meta", {})
        pt = meta.get("prompt_tokens", 0)
        ct = meta.get("completion_tokens", 0)
        tt = meta.get("total_tokens", 0)
        cost = meta.get("cost", 0.0)

        with self._lock:
            self.prompt_tokens += pt
            self.completion_tokens += ct
            self.total_tokens += tt
            self.cost += cost
            self.call_count += 1

            model = response_info.get("driver", "unknown")
            logger.debug(
                "[session] record driver=%s delta_tokens=%d delta_cost=%.6f | session total_tokens=%d cost=%.6f calls=%d",
                model,
                tt,
                cost,
                self.total_tokens,
                self.cost,
                self.call_count,
            )

            # Capture timing
            elapsed_ms = response_info.get("elapsed_ms", 0.0)
            if elapsed_ms > 0:
                self.total_elapsed_ms += elapsed_ms
                self._elapsed_samples.append(elapsed_ms)

            bucket = self._per_model.setdefault(
                model,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                    "elapsed_ms": 0.0,
                    "elapsed_samples": [],
                },
            )
            bucket["prompt_tokens"] += pt
            bucket["completion_tokens"] += ct
            bucket["total_tokens"] += tt
            bucket["cost"] += cost
            bucket["calls"] += 1
            if elapsed_ms > 0:
                bucket["elapsed_ms"] += elapsed_ms
                bucket["elapsed_samples"].append(elapsed_ms)

    def record_error(self, error_info: dict[str, Any]) -> None:
        """Record a driver error.

        Compatible as an ``on_error`` callback for
        :class:`~prompture.callbacks.DriverCallbacks`.
        """
        with self._lock:
            self.errors += 1

    # ------------------------------------------------------------------ #
    # Computed timing properties
    # ------------------------------------------------------------------ #

    @property
    def tokens_per_second(self) -> float:
        """Average output tokens per second across all calls."""
        if self.total_elapsed_ms <= 0:
            return 0.0
        return self.completion_tokens / (self.total_elapsed_ms / 1000)

    @property
    def latency_stats(self) -> dict[str, float]:
        """Return min/max/avg/p95 latency in milliseconds."""
        if not self._elapsed_samples:
            return {"min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}

        samples = sorted(self._elapsed_samples)
        p95_idx = int(len(samples) * 0.95)
        return {
            "min_ms": min(samples),
            "max_ms": max(samples),
            "avg_ms": sum(samples) / len(samples),
            "p95_ms": samples[min(p95_idx, len(samples) - 1)],
        }

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        """Return a machine-readable summary with a ``formatted`` string."""
        stats = self.latency_stats
        tps = self.tokens_per_second

        formatted = (
            f"Session: {self.total_tokens:,} tokens across {self.call_count} call(s) costing ${self.cost:.4f}"
        )
        if self.total_elapsed_ms > 0:
            formatted += f" | {tps:.1f} tok/s avg, {stats['avg_ms']:.0f}ms avg latency"
        if self.errors:
            formatted += f" ({self.errors} error(s))"

        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "total_cost": self.cost,  # Deprecated alias for backwards compatibility
            "call_count": self.call_count,
            "errors": self.errors,
            "total_elapsed_ms": self.total_elapsed_ms,
            "tokens_per_second": tps,
            "latency_stats": stats,
            "per_model": dict(self._per_model),
            "formatted": formatted,
        }

    def reset(self) -> None:
        """Clear all accumulated counters."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.cost = 0.0
            self.call_count = 0
            self.errors = 0
            self.total_elapsed_ms = 0.0
            self._elapsed_samples.clear()
            self._per_model.clear()
