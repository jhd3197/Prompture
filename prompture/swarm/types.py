"""Shared types for the Prompture swarm runtime.

Defines event records, memory records, swarm-level callbacks, and the
result/step shapes returned by :class:`~prompture.swarm.Swarm`.
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agents.types import AgentResult


class EventKind(str, enum.Enum):
    """Classification for entries in the environment event log.

    Values are stored as plain strings so callers may freely add their
    own custom kinds without modifying the enum (kinds are compared by
    string equality everywhere in the swarm runtime).
    """

    say = "say"
    act = "act"
    observe = "observe"
    arrive = "arrive"
    leave = "leave"
    seed = "seed"
    system = "system"
    custom = "custom"


@dataclass
class Event:
    """A single record in the environment's append-only event log.

    Attributes:
        tick: Round index at which the event was emitted.
        ts: Wall-clock timestamp (seconds since epoch).
        agent_id: Identifier of the agent that emitted the event.  Use
            ``"__env__"`` for events authored by the environment itself
            (seed material, system messages, etc.).
        kind: Event classification.  See :class:`EventKind`.
        content: Free-form payload, typically a short natural-language
            string describing what happened.
        targets: Tuple of agent IDs the event is addressed to.  Empty
            tuple means broadcast — visible to every agent.
        topic: Optional topic tag for subscription-style filtering.
        metadata: Arbitrary structured payload (parsed JSON output,
            sentiment scores, etc.).
    """

    tick: int
    ts: float
    agent_id: str
    kind: str
    content: str
    targets: tuple[str, ...] = ()
    topic: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    """A single record stored in a :class:`MemoryStore`.

    Attributes:
        agent_id: Identifier of the agent that owns this memory.
        content: Free-form text.
        ts: Wall-clock timestamp (seconds since epoch).
        tick: Optional round index of authorship.
        metadata: Arbitrary structured payload.
    """

    agent_id: str
    content: str
    ts: float = field(default_factory=time.time)
    tick: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SwarmStep:
    """One agent invocation recorded during a swarm tick.

    Mirrors the shape of :class:`~prompture.groups.types.GroupStep` so
    consumers of group results can read swarm timelines with the same
    code path.

    Attributes:
        tick: Round index in which the step ran.
        agent_id: The acting agent's identifier.
        step_type: ``"agent_run"`` on success, ``"agent_error"`` on
            failure, ``"agent_skipped"`` when the scheduler omitted the
            agent.
        timestamp: ``time.perf_counter()`` reading at step start.
        duration_ms: Wall-clock duration of the agent's ``run()`` call.
        usage_delta: Per-step usage summary harvested from the agent
            result's ``run_usage``.
        observation: The text prompt the agent was given for this step.
        output: The agent's textual output (``output_text``).  ``None``
            for skipped or errored steps.
        error: String form of any raised exception, or ``None``.
    """

    tick: int
    agent_id: str
    step_type: str = "agent_run"
    timestamp: float = 0.0
    duration_ms: float = 0.0
    usage_delta: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    output: str | None = None
    error: str | None = None


@dataclass
class SwarmResult:
    """Outcome of a :meth:`Swarm.run` invocation.

    Attributes:
        agent_results: Mapping of ``"{agent_id}@tick{n}"`` to the
            corresponding :class:`~prompture.agents.types.AgentResult`.
        aggregate_usage: Combined token/cost totals across every agent
            invocation in the run.
        environment_state: Snapshot of the environment's world-state
            dict at the end of the run.
        events: Full event timeline from the environment.
        elapsed_ms: Wall-clock duration of the swarm run.
        timeline: Ordered list of :class:`SwarmStep` records, one per
            agent invocation (including skips and errors).
        errors: List of ``(agent_id, exception)`` for failed steps.
        ticks_run: Number of completed ticks (some may have been cut
            short by ``stop()`` or budget exhaustion).
        success: ``True`` when no errors were recorded.
        stopped_reason: Human-readable reason the run stopped, or
            ``None`` when the loop completed normally.
    """

    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    aggregate_usage: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    elapsed_ms: float = 0.0
    timeline: list[SwarmStep] = field(default_factory=list)
    errors: list[tuple[str, Exception]] = field(default_factory=list)
    ticks_run: int = 0
    success: bool = True
    stopped_reason: str | None = None

    def export(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {
            "ticks_run": self.ticks_run,
            "success": self.success,
            "stopped_reason": self.stopped_reason,
            "elapsed_ms": self.elapsed_ms,
            "aggregate_usage": self.aggregate_usage,
            "environment_state": self.environment_state,
            "agent_results": {
                k: {
                    "output_text": getattr(v, "output_text", str(v)),
                    "usage": getattr(v, "run_usage", {}),
                }
                for k, v in self.agent_results.items()
            },
            "timeline": [
                {
                    "tick": s.tick,
                    "agent_id": s.agent_id,
                    "step_type": s.step_type,
                    "duration_ms": s.duration_ms,
                    "usage_delta": s.usage_delta,
                    "observation": s.observation,
                    "output": s.output,
                    "error": s.error,
                }
                for s in self.timeline
            ],
            "events": [
                {
                    "tick": e.tick,
                    "ts": e.ts,
                    "agent_id": e.agent_id,
                    "kind": e.kind,
                    "content": e.content,
                    "targets": list(e.targets),
                    "topic": e.topic,
                    "metadata": e.metadata,
                }
                for e in self.events
            ],
            "errors": [
                {"agent_id": aid, "error": str(exc)}
                for aid, exc in self.errors
            ],
        }


@dataclass
class SwarmCallbacks:
    """Observability hooks fired during a swarm run.

    All callbacks are optional.  Set any subset to receive notifications;
    leave the rest as ``None`` to no-op.

    Attributes:
        on_tick_start: ``(tick, scheduled_agent_ids)`` at the start of
            each round, after scheduling has run.
        on_tick_complete: ``(tick,)`` at the end of each round.
        on_agent_start: ``(agent_id, observation, tick)`` before the
            agent's underlying ``Agent.run`` is invoked.
        on_agent_complete: ``(agent_id, result, tick)`` after a
            successful agent run.
        on_agent_error: ``(agent_id, exception, tick)`` on failure.
        on_event: ``(event,)`` for every event written to the
            environment.
        on_state_update: ``(key, value)`` whenever the environment's
            world state is mutated via :meth:`Environment.set`.
        on_stop: ``(reason,)`` when the loop ends (normal or otherwise).
    """

    on_tick_start: Callable[..., Any] | None = None
    on_tick_complete: Callable[..., Any] | None = None
    on_agent_start: Callable[..., Any] | None = None
    on_agent_complete: Callable[..., Any] | None = None
    on_agent_error: Callable[..., Any] | None = None
    on_event: Callable[..., Any] | None = None
    on_state_update: Callable[..., Any] | None = None
    on_stop: Callable[..., Any] | None = None


def aggregate_usage(*sessions: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple usage summary dicts into one.

    Tolerates both ``cost`` and ``total_cost`` keys (different parts of
    the codebase emit different names) by summing whichever is present
    into a single ``total_cost`` figure.
    """
    agg: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "call_count": 0,
        "errors": 0,
    }
    for s in sessions:
        if not s:
            continue
        agg["prompt_tokens"] += s.get("prompt_tokens", 0)
        agg["completion_tokens"] += s.get("completion_tokens", 0)
        agg["total_tokens"] += s.get("total_tokens", 0)
        agg["total_cost"] += s.get("total_cost", s.get("cost", 0.0))
        agg["call_count"] += s.get("call_count", 0)
        agg["errors"] += s.get("errors", 0)
    return agg
