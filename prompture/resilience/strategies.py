"""Target-ordering strategies and the per-target stats they read.

A strategy decides which *model* a resilient route tries first. Keys inside
a model keep their own round-robin (or sticky) order; the breakers and the
attempt loop still skip anything unhealthy, so a strategy only has to
express preference.
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

STRATEGIES = ("priority", "round_robin", "weighted", "latency", "p2c", "last_good", "cheapest")


@dataclass
class TargetStats:
    """Rolling view of one target's recent behavior."""

    ewma_latency_ms: float | None = None
    successes: int = 0
    failures: int = 0
    last_success: float | None = None
    last_failure: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ewma_latency_ms": None if self.ewma_latency_ms is None else round(self.ewma_latency_ms, 1),
            "successes": self.successes,
            "failures": self.failures,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
        }


class RouteStats:
    """Thread-safe stats keyed by target label (``provider/model`` or ``provider/model#keyid``)."""

    def __init__(self, alpha: float = 0.3, clock: Callable[[], float] = time.time) -> None:
        self.alpha = alpha
        self._clock = clock
        self._lock = threading.Lock()
        self._stats: dict[str, TargetStats] = {}

    def _get(self, label: str) -> TargetStats:
        stats = self._stats.get(label)
        if stats is None:
            stats = self._stats[label] = TargetStats()
        return stats

    def record_success(self, label: str, elapsed_ms: float | None = None) -> None:
        with self._lock:
            s = self._get(label)
            s.successes += 1
            s.last_success = self._clock()
            if elapsed_ms is not None:
                s.ewma_latency_ms = (
                    elapsed_ms
                    if s.ewma_latency_ms is None
                    else self.alpha * elapsed_ms + (1 - self.alpha) * s.ewma_latency_ms
                )

    def record_failure(self, label: str) -> None:
        with self._lock:
            s = self._get(label)
            s.failures += 1
            s.last_failure = self._clock()

    def get(self, label: str) -> TargetStats:
        with self._lock:
            s = self._stats.get(label)
            return TargetStats(**vars(s)) if s else TargetStats()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: v.to_dict() for k, v in self._stats.items()}

    def reset(self) -> None:
        with self._lock:
            self._stats.clear()


_default_stats = RouteStats()


def get_route_stats() -> RouteStats:
    """Process-wide stats shared by resilient drivers by default."""
    return _default_stats


def _price_per_mtok(model: str) -> float | None:
    """Blended input+output price per 1M tokens, or ``None`` if unknown."""
    if "/" not in model:
        return None
    provider, model_id = model.split("/", 1)
    try:
        from ..infra.model_rates import get_model_rates

        rates = get_model_rates(provider, model_id)
    except Exception:
        return None
    if not rates:
        return None
    inp, out = rates.get("input"), rates.get("output")
    if inp is None and out is None:
        return None
    return float(inp or 0.0) + float(out or 0.0)


class Orderer:
    """Applies one strategy to a sequence of items (model groups).

    ``key(item)`` gives the stats/pricing label for an item (its model string).
    """

    def __init__(
        self,
        strategy: str = "priority",
        *,
        weights: Sequence[float] | None = None,
        stats: RouteStats | None = None,
        price: Callable[[str], float | None] = _price_per_mtok,
        rng: random.Random | None = None,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown routing strategy '{strategy}'. Choose one of: {', '.join(STRATEGIES)}")
        self.strategy = strategy
        self.weights = list(weights) if weights is not None else None
        self.stats = stats or get_route_stats()
        self._price = price
        self._price_cache: dict[str, float | None] = {}
        self._rng = rng or random.Random()  # nosec B311 - load spreading, not crypto
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def _cost(self, model: str) -> float | None:
        if model not in self._price_cache:
            self._price_cache[model] = self._price(model)
        return self._price_cache[model]

    def order(self, items: Sequence[T], key: Callable[[T], str]) -> list[T]:
        items = list(items)
        if len(items) < 2 or self.strategy == "priority":
            return items
        s = self.strategy
        if s == "round_robin":
            with self._lock:
                start = next(self._counter) % len(items)
            return items[start:] + items[:start]
        if s == "weighted":
            weights = self.weights or [1.0] * len(items)
            if len(weights) != len(items):
                raise ValueError(f"weighted strategy needs {len(items)} weights, got {len(weights)}")
            pool = list(zip(items, weights, strict=True))
            out: list[T] = []
            while pool:
                total = sum(max(w, 0.0) for _, w in pool)
                if total <= 0:
                    out.extend(i for i, _ in pool)
                    break
                pick = self._rng.uniform(0, total)
                acc = 0.0
                for idx, (item, w) in enumerate(pool):
                    acc += max(w, 0.0)
                    if pick <= acc:
                        out.append(item)
                        pool.pop(idx)
                        break
            return out
        if s == "latency":
            # Unmeasured targets sort first so every target gets explored.
            return sorted(items, key=lambda i: self.stats.get(key(i)).ewma_latency_ms or 0.0)
        if s == "p2c":
            a, b = self._rng.sample(range(len(items)), 2)

            def lat(i: int) -> float:
                return self.stats.get(key(items[i])).ewma_latency_ms or 0.0

            first = a if lat(a) <= lat(b) else b
            return [items[first]] + [it for n, it in enumerate(items) if n != first]
        if s == "last_good":
            best = max(
                range(len(items)),
                key=lambda n: self.stats.get(key(items[n])).last_success or float("-inf"),
            )
            if self.stats.get(key(items[best])).last_success is None:
                return items
            return [items[best]] + [it for n, it in enumerate(items) if n != best]
        if s == "cheapest":
            indexed = list(enumerate(items))
            indexed.sort(key=lambda p: (self._cost(key(p[1])) is None, self._cost(key(p[1])) or 0.0, p[0]))
            return [it for _, it in indexed]
        return items  # pragma: no cover - guarded by STRATEGIES check
