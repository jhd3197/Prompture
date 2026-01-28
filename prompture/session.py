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

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UsageSession:
    """Accumulates usage statistics across multiple driver calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    errors: int = 0
    _per_model: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

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

        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt
        self.total_cost += cost
        self.call_count += 1

        model = response_info.get("driver", "unknown")
        bucket = self._per_model.setdefault(
            model,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0, "calls": 0},
        )
        bucket["prompt_tokens"] += pt
        bucket["completion_tokens"] += ct
        bucket["total_tokens"] += tt
        bucket["cost"] += cost
        bucket["calls"] += 1

    def record_error(self, error_info: dict[str, Any]) -> None:
        """Record a driver error.

        Compatible as an ``on_error`` callback for
        :class:`~prompture.callbacks.DriverCallbacks`.
        """
        self.errors += 1

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, Any]:
        """Return a machine-readable summary with a ``formatted`` string."""
        formatted = (
            f"Session: {self.total_tokens:,} tokens across {self.call_count} call(s) costing ${self.total_cost:.4f}"
        )
        if self.errors:
            formatted += f" ({self.errors} error(s))"

        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "errors": self.errors,
            "per_model": dict(self._per_model),
            "formatted": formatted,
        }

    def reset(self) -> None:
        """Clear all accumulated counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
        self.errors = 0
        self._per_model.clear()
