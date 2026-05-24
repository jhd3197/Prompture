"""Scheduling policies for :class:`~prompture.swarm.Swarm`.

A scheduler decides which agents act in each tick.  Implementations
satisfy the simple :class:`Scheduler` protocol::

    def select(agents, env, tick) -> list[SwarmAgent]

The returned list is processed in order (left-to-right) within the
tick.  Returning an empty list is allowed — the swarm will still advance
ticks (useful when waiting for an external signal via callbacks).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .environment import Environment
    from .swarm_agent import SwarmAgent


@runtime_checkable
class Scheduler(Protocol):
    """Decides which agents act in each tick."""

    def select(
        self,
        agents: Sequence[SwarmAgent],
        env: Environment,
        tick: int,
    ) -> list[SwarmAgent]:
        """Return the subset of agents that should run this tick."""
        ...


class AllScheduler:
    """Every agent acts every tick.  Default.

    Use when every agent should observe and react each round — the
    classic broadcast simulation.  Order matches the swarm's agent
    list.
    """

    def select(
        self,
        agents: Sequence[SwarmAgent],
        env: Environment,
        tick: int,
    ) -> list[SwarmAgent]:
        return list(agents)


class SampleScheduler:
    """Sample *k* agents uniformly at random per tick.

    Use for gossip-style or stochastic simulations where only part of
    the population activates each round.

    Args:
        k: Number of agents to sample per tick.  Clamped to the
            population size.
        rng: Optional :class:`random.Random` instance for reproducible
            sampling.  When ``None`` the module-level ``random`` is used.
    """

    def __init__(self, k: int, *, rng: random.Random | None = None) -> None:
        if k < 0:
            raise ValueError("SampleScheduler k must be non-negative")
        self._k = k
        self._rng = rng or random.Random()  # nosec B311 - non-cryptographic agent sampling

    def select(
        self,
        agents: Sequence[SwarmAgent],
        env: Environment,
        tick: int,
    ) -> list[SwarmAgent]:
        if not agents:
            return []
        k = min(self._k, len(agents))
        return self._rng.sample(list(agents), k)


class RoundRobinScheduler:
    """Activate one agent per tick, rotating through the population.

    Args:
        batch_size: Number of agents to activate per tick (default 1).
            The window slides by ``batch_size`` each tick.
    """

    def __init__(self, batch_size: int = 1) -> None:
        if batch_size < 1:
            raise ValueError("RoundRobinScheduler batch_size must be >= 1")
        self._batch_size = batch_size
        self._cursor = 0

    def select(
        self,
        agents: Sequence[SwarmAgent],
        env: Environment,
        tick: int,
    ) -> list[SwarmAgent]:
        if not agents:
            return []
        n = len(agents)
        out: list[SwarmAgent] = []
        for i in range(self._batch_size):
            out.append(agents[(self._cursor + i) % n])
        self._cursor = (self._cursor + self._batch_size) % n
        return out


class PriorityScheduler:
    """Activate agents whose priority score exceeds a threshold.

    The scoring function is called once per agent per tick.  The
    threshold is checked with ``>=``.  Activated agents are returned in
    score-descending order so highest-priority actors run first.

    Args:
        score_fn: Callable ``(agent, env, tick) -> float`` producing a
            per-tick activation score.
        threshold: Minimum score required to activate.  Default ``1.0``.
        max_active: Optional hard cap on activations per tick.
    """

    def __init__(
        self,
        score_fn: Callable[[SwarmAgent, Environment, int], float],
        *,
        threshold: float = 1.0,
        max_active: int | None = None,
    ) -> None:
        self._score_fn = score_fn
        self._threshold = threshold
        self._max_active = max_active

    def select(
        self,
        agents: Sequence[SwarmAgent],
        env: Environment,
        tick: int,
    ) -> list[SwarmAgent]:
        scored: list[tuple[float, SwarmAgent]] = []
        for agent in agents:
            score = self._score_fn(agent, env, tick)
            if score >= self._threshold:
                scored.append((score, agent))
        scored.sort(key=lambda t: t[0], reverse=True)
        if self._max_active is not None:
            scored = scored[: self._max_active]
        return [a for _, a in scored]


class CallableScheduler:
    """Adapter that turns any callable into a :class:`Scheduler`.

    Useful for one-off scheduling logic without subclassing::

        scheduler = CallableScheduler(
            lambda agents, env, tick: [a for a in agents if a.name != "spectator"]
        )

    The wrapped callable must accept ``(agents, env, tick)`` and return
    a list of :class:`SwarmAgent`.
    """

    def __init__(
        self,
        fn: Callable[[Sequence[SwarmAgent], Environment, int], list[SwarmAgent]],
    ) -> None:
        self._fn = fn

    def select(
        self,
        agents: Sequence[SwarmAgent],
        env: Environment,
        tick: int,
    ) -> list[SwarmAgent]:
        return list(self._fn(agents, env, tick))


def make_scheduler(spec: Any) -> Scheduler:
    """Coerce *spec* to a :class:`Scheduler`.

    Accepts:
    - A :class:`Scheduler` instance (returned as-is).
    - The string ``"all"`` (→ :class:`AllScheduler`).
    - A callable matching :class:`CallableScheduler`'s signature.

    Used by :class:`~prompture.swarm.Swarm` to accept friendly inputs.
    """
    if spec is None:
        return AllScheduler()
    if isinstance(spec, Scheduler):
        return spec
    if isinstance(spec, str):
        if spec == "all":
            return AllScheduler()
        raise ValueError(f"Unknown scheduler spec: {spec!r}")
    if callable(spec):
        return CallableScheduler(spec)
    raise TypeError(f"Cannot coerce {type(spec).__name__} to Scheduler")
