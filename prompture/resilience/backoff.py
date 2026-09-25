"""Retry timing: exponential backoff with jitter that honors server hints."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """How a resilient route retries and how long it is willing to wait.

    Attributes:
        max_attempts: Attempts per target for transient errors (1 = no retry).
        base_delay: First backoff delay in seconds.
        multiplier: Growth factor between attempts.
        max_delay: Ceiling for computed backoff delays.
        jitter: Random spread as a fraction of the delay (0.2 = ±20%).
        max_wait: Longest server-requested wait (``Retry-After``) the route
            will sleep through inline. Longer hints park the target instead.
        default_cooldown: How long to park a rate-limited target when the
            provider gave no ``Retry-After`` hint.
        disable_cooldown: How long to park a key after an auth / billing /
            exhausted-quota error.
        min_headroom: Targets whose reported rate-limit headroom (fraction of
            a still-open window left) is below this move to the back of the
            route. ``None`` turns headroom-aware ordering off.
    """

    max_attempts: int = 2
    base_delay: float = 0.5
    multiplier: float = 2.0
    max_delay: float = 20.0
    jitter: float = 0.2
    max_wait: float = 30.0
    default_cooldown: float = 30.0
    disable_cooldown: float = 3600.0
    min_headroom: float | None = 0.05

    def backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before retry number *attempt* (0-based) of the same target."""
        if retry_after is not None:
            return min(retry_after, self.max_wait)
        delay = min(self.base_delay * (self.multiplier**attempt), self.max_delay)
        if self.jitter:
            spread = delay * self.jitter
            delay += random.uniform(-spread, spread)  # nosec B311 - timing jitter, not crypto
        return max(0.0, delay)


NO_RETRY = RetryPolicy(max_attempts=1, max_wait=0.0)
"""Fail over immediately; never sleep."""
