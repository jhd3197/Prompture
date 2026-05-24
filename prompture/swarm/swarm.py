"""The :class:`Swarm` orchestrator.

A swarm is a population of :class:`SwarmAgent` instances acting against
a shared :class:`Environment` over multiple rounds (ticks).  Each tick
the active subset (chosen by a :class:`Scheduler`) takes one action,
writing observations back into the environment for the next round.

The runtime mirrors conventions established in
:mod:`prompture.groups`: an :class:`ErrorPolicy` enum reuse, a
:class:`SwarmCallbacks` dataclass, a ``stop()`` cooperative shutdown,
and a ``max_total_cost`` budget cap.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

from ..groups.types import ErrorPolicy
from .environment import Environment
from .memory import MemoryStore
from .scheduler import AllScheduler, Scheduler, make_scheduler
from .swarm_agent import SwarmAgent
from .types import SwarmCallbacks, SwarmResult, SwarmStep, aggregate_usage

logger = logging.getLogger("prompture.swarm")

_DEFAULT_MAX_TICKS = 10
_DEFAULT_ACTION_PROMPT = (
    "React to the current situation. Stay in character and respond concisely."
)


class Swarm:
    """A population of agents acting over rounds against a shared environment.

    Args:
        agents: The participating :class:`SwarmAgent` instances.
            Names must be unique within the swarm.
        env: Pre-built :class:`Environment` to use.  When ``None`` a
            fresh environment is created.
        memory: Default :class:`MemoryStore` applied to every agent
            that does not already have one.  Pass ``None`` to leave
            agents memory-less by default.
        scheduler: A :class:`Scheduler` instance, the string ``"all"``,
            or a callable accepted by
            :func:`~prompture.swarm.scheduler.make_scheduler`.
            Defaults to :class:`AllScheduler` — every agent acts every
            tick.
        max_ticks: Hard cap on rounds.  Default 10.
        action_prompt: Per-tick action directive passed to every agent
            (via :attr:`SwarmAgent._pending_prompt`).  Either a literal
            string (with ``{tick}`` placeholder) or a callable
            ``(env, tick) -> str``.
        max_total_cost: Optional USD budget cap.  When the aggregate
            cost crosses this threshold the run stops at the next tick
            boundary.
        error_policy: Reuses :class:`~prompture.groups.types.ErrorPolicy`.
            ``fail_fast`` (default) ends the run on the first error;
            ``continue_on_error`` records errors and keeps going;
            ``raise_on_error`` propagates the exception immediately.
            ``retry_failed`` is treated as ``continue_on_error`` here —
            scheduler-driven re-activation is the recommended retry
            mechanism.
        callbacks: Optional :class:`SwarmCallbacks` for observability.
        stop_when: Optional callable ``(env, tick) -> bool`` evaluated
            at the start of every tick.  Returning ``True`` ends the
            run before the tick executes.

    Example::

        swarm = Swarm(
            agents=[journalist, contrarian, moderator],
            scheduler=AllScheduler(),
            max_ticks=5,
        )
        swarm.environment.seed("Breaking: city council passes new tax.")
        result = swarm.run()
        for ev in result.events:
            print(ev.tick, ev.agent_id, ev.content)
    """

    def __init__(
        self,
        agents: Iterable[SwarmAgent],
        *,
        env: Environment | None = None,
        memory: MemoryStore | None = None,
        scheduler: Scheduler | str | Callable[..., Any] | None = None,
        max_ticks: int = _DEFAULT_MAX_TICKS,
        action_prompt: str | Callable[[Environment, int], str] = _DEFAULT_ACTION_PROMPT,
        max_total_cost: float | None = None,
        error_policy: ErrorPolicy = ErrorPolicy.fail_fast,
        callbacks: SwarmCallbacks | None = None,
        stop_when: Callable[[Environment, int], bool] | None = None,
    ) -> None:
        self._agents: list[SwarmAgent] = list(agents)
        if not self._agents:
            raise ValueError("Swarm requires at least one agent")

        seen: set[str] = set()
        for a in self._agents:
            if a.name in seen:
                raise ValueError(f"Duplicate swarm agent name: {a.name!r}")
            seen.add(a.name)

        if memory is not None:
            for a in self._agents:
                if a.memory is None:
                    a.memory = memory

        self._env = env if env is not None else Environment()
        self._memory_default = memory
        self._scheduler: Scheduler = make_scheduler(scheduler) if not isinstance(scheduler, Scheduler) else scheduler
        if self._scheduler is None:  # pragma: no cover — defensive
            self._scheduler = AllScheduler()
        if max_ticks < 0:
            raise ValueError("max_ticks must be non-negative")
        self._max_ticks = max_ticks
        self._action_prompt = action_prompt
        self._max_total_cost = max_total_cost
        self._error_policy = error_policy
        self._callbacks = callbacks or SwarmCallbacks()
        self._stop_when = stop_when

        self._stop_requested: bool = False
        self._stop_reason: str | None = None

        # Wire environment callbacks into swarm callbacks so emit/state
        # events bubble up through the SwarmCallbacks surface too.
        cb_event = self._callbacks.on_event
        if cb_event is not None and self._env._on_event is None:
            self._env._on_event = cb_event  # type: ignore[assignment]
        cb_state = self._callbacks.on_state_update
        if cb_state is not None and self._env._on_state_update is None:
            self._env._on_state_update = cb_state  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def environment(self) -> Environment:
        """The shared environment for this swarm."""
        return self._env

    @property
    def agents(self) -> list[SwarmAgent]:
        """Return a shallow copy of the agent list."""
        return list(self._agents)

    def stop(self, reason: str = "stop() called") -> None:
        """Request graceful shutdown after the current tick finishes."""
        self._stop_requested = True
        if self._stop_reason is None:
            self._stop_reason = reason

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, seed: str | None = None) -> SwarmResult:
        """Execute the swarm's tick loop to completion.

        Args:
            seed: Optional seed text broadcast through the environment
                before tick 1.  Equivalent to calling
                :meth:`Environment.seed` yourself.

        Returns:
            :class:`SwarmResult` with the full timeline, errors, final
            environment state, and aggregate usage.
        """
        if seed is not None:
            self._env.seed(seed)

        self._stop_requested = False
        self._stop_reason = None
        t0 = time.perf_counter()

        timeline: list[SwarmStep] = []
        agent_results: dict[str, Any] = {}
        errors: list[tuple[str, Exception]] = []
        usage_summaries: list[dict[str, Any]] = []
        ticks_run = 0

        for tick_idx in range(1, self._max_ticks + 1):
            if self._stop_requested:
                break

            if self._stop_when is not None:
                try:
                    if self._stop_when(self._env, tick_idx):
                        self._stop_reason = self._stop_reason or "stop_when returned True"
                        break
                except Exception:
                    logger.warning("stop_when raised; ending swarm run", exc_info=True)
                    self._stop_reason = "stop_when raised"
                    break

            if self._max_total_cost is not None:
                total_so_far = sum(s.get("total_cost", s.get("cost", 0.0)) for s in usage_summaries)
                if total_so_far >= self._max_total_cost:
                    self._stop_reason = (
                        f"budget exceeded ({total_so_far:.4f} >= {self._max_total_cost:.4f})"
                    )
                    break

            self._env.advance_tick()
            active = list(self._scheduler.select(self._agents, self._env, tick_idx))

            if self._callbacks.on_tick_start:
                self._callbacks.on_tick_start(tick_idx, [a.name for a in active])

            action_prompt = self._resolve_action_prompt(tick_idx)
            tick_had_error = False

            for swarm_agent in active:
                if self._stop_requested:
                    break

                step_t0 = time.perf_counter()
                if self._callbacks.on_agent_start:
                    self._callbacks.on_agent_start(
                        swarm_agent.name,
                        action_prompt,
                        tick_idx,
                    )

                try:
                    result = swarm_agent.run_step(self._env, tick_idx, action_prompt)
                    duration_ms = (time.perf_counter() - step_t0) * 1000

                    key = f"{swarm_agent.name}@tick{tick_idx}"
                    agent_results[key] = result
                    usage = getattr(result, "run_usage", {}) or {}
                    usage_summaries.append(usage)

                    timeline.append(
                        SwarmStep(
                            tick=tick_idx,
                            agent_id=swarm_agent.name,
                            step_type="agent_run",
                            timestamp=step_t0,
                            duration_ms=duration_ms,
                            usage_delta=usage,
                            observation=swarm_agent._pending_prompt,
                            output=getattr(result, "output_text", str(result)),
                        )
                    )

                    if self._callbacks.on_agent_complete:
                        self._callbacks.on_agent_complete(
                            swarm_agent.name, result, tick_idx
                        )

                except Exception as exc:
                    duration_ms = (time.perf_counter() - step_t0) * 1000
                    errors.append((swarm_agent.name, exc))
                    tick_had_error = True
                    timeline.append(
                        SwarmStep(
                            tick=tick_idx,
                            agent_id=swarm_agent.name,
                            step_type="agent_error",
                            timestamp=step_t0,
                            duration_ms=duration_ms,
                            observation=swarm_agent._pending_prompt,
                            error=str(exc),
                        )
                    )

                    if self._callbacks.on_agent_error:
                        self._callbacks.on_agent_error(
                            swarm_agent.name, exc, tick_idx
                        )

                    if self._error_policy == ErrorPolicy.raise_on_error:
                        self._stop_reason = f"raise_on_error: {swarm_agent.name}"
                        if self._callbacks.on_stop:
                            self._callbacks.on_stop(self._stop_reason)
                        raise
                    if self._error_policy == ErrorPolicy.fail_fast:
                        self._stop_reason = f"fail_fast on {swarm_agent.name}: {exc}"
                        break
                    # continue_on_error / retry_failed: continue with next agent.

            ticks_run = tick_idx
            if self._callbacks.on_tick_complete:
                self._callbacks.on_tick_complete(tick_idx)

            if tick_had_error and self._error_policy == ErrorPolicy.fail_fast:
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if self._stop_reason is None and ticks_run >= self._max_ticks:
            self._stop_reason = "max_ticks reached"
        if self._callbacks.on_stop and self._stop_reason is not None:
            self._callbacks.on_stop(self._stop_reason)

        return SwarmResult(
            agent_results=agent_results,
            aggregate_usage=aggregate_usage(*usage_summaries),
            environment_state=self._env.state,
            events=self._env.events,
            elapsed_ms=elapsed_ms,
            timeline=timeline,
            errors=errors,
            ticks_run=ticks_run,
            success=len(errors) == 0,
            stopped_reason=self._stop_reason,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_action_prompt(self, tick: int) -> str:
        spec = self._action_prompt
        if callable(spec):
            return spec(self._env, tick)
        return spec.format(tick=tick) if "{tick}" in spec else spec
