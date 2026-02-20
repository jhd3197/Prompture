"""Shared types for multi-agent group coordination.

Defines enums, dataclasses, and callbacks used by
:class:`~prompture.groups.SequentialGroup`,
:class:`~prompture.async_groups.ParallelGroup`, and related classes.
"""

from __future__ import annotations

import enum
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ErrorPolicy(enum.Enum):
    """How a group handles agent failures."""

    fail_fast = "fail_fast"
    continue_on_error = "continue_on_error"
    retry_failed = "retry_failed"
    raise_on_error = "raise_on_error"


@dataclass
class GroupStep:
    """Record of a single agent execution within a group run."""

    agent_name: str
    step_type: str = "agent_run"
    timestamp: float = 0.0
    duration_ms: float = 0.0
    usage_delta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class AgentError:
    """Captures a failed agent execution within a group."""

    agent_name: str
    error: Exception
    error_message: str = ""
    output_key: str | None = None

    def __post_init__(self) -> None:
        if not self.error_message:
            self.error_message = str(self.error)


@dataclass
class GroupResult:
    """Outcome of a group execution.

    Attributes:
        agent_results: Mapping of agent name/key to their :class:`AgentResult`.
        aggregate_usage: Combined token/cost totals across all agent runs.
        shared_state: Final state dict after all agents have written outputs.
        elapsed_ms: Wall-clock duration of the group run.
        timeline: Ordered list of :class:`GroupStep` records.
        errors: List of :class:`AgentError` for any failed agents.
        success: ``True`` if no errors occurred.
    """

    agent_results: dict[str, Any] = field(default_factory=dict)
    aggregate_usage: dict[str, Any] = field(default_factory=dict)
    shared_state: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    timeline: list[GroupStep] = field(default_factory=list)
    errors: list[AgentError] = field(default_factory=list)
    success: bool = True

    def export(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "agent_results": {
                k: {
                    "output_text": getattr(v, "output_text", str(v)),
                    "usage": getattr(v, "run_usage", {}),
                }
                for k, v in self.agent_results.items()
            },
            "aggregate_usage": self.aggregate_usage,
            "shared_state": self.shared_state,
            "elapsed_ms": self.elapsed_ms,
            "timeline": [
                {
                    "agent_name": s.agent_name,
                    "step_type": s.step_type,
                    "timestamp": s.timestamp,
                    "duration_ms": s.duration_ms,
                    "usage_delta": s.usage_delta,
                    "error": s.error,
                }
                for s in self.timeline
            ],
            "errors": [
                {
                    "agent_name": e.agent_name,
                    "error_message": e.error_message,
                    "output_key": e.output_key,
                }
                for e in self.errors
            ],
            "success": self.success,
        }

    def save(self, path: str) -> None:
        """Write the exported dict to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.export(), f, indent=2, default=str)


@dataclass
class GroupCallbacks:
    """Observability callbacks for group execution.

    Attributes:
        on_agent_start: Called when an agent begins execution ``(name, prompt)``.
        on_agent_complete: Called when an agent finishes ``(name, result)``.
        on_agent_error: Called when an agent raises ``(name, exception)``.
        on_state_update: Called when shared state is updated ``(key, value)``.
        on_round_start: Called at the start of a debate/loop round ``(round_number,)``.
        on_round_complete: Called at the end of a debate/loop round ``(round_number,)``.
        on_step_skipped: Called when a waterfall step is skipped ``(agent_name, reason)``.
        on_route_decision: Called when a router selects an agent
            ``(agent_name, reason, confidence)``.
    """

    on_agent_start: Callable[[str, str], None] | None = None
    on_agent_complete: Callable[[str, Any], None] | None = None
    on_agent_error: Callable[[str, Exception], None] | None = None
    on_state_update: Callable[[str, Any], None] | None = None
    on_round_start: Callable[[int], None] | None = None
    on_round_complete: Callable[[int], None] | None = None
    on_step_skipped: Callable[[str, str], None] | None = None
    on_route_decision: Callable[[str, str, float], None] | None = None


def _aggregate_usage(*sessions: dict[str, Any]) -> dict[str, Any]:
    """Merge multiple usage summary dicts into one aggregate."""
    agg: dict[str, Any] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "call_count": 0,
        "errors": 0,
    }
    for s in sessions:
        agg["prompt_tokens"] += s.get("prompt_tokens", 0)
        agg["completion_tokens"] += s.get("completion_tokens", 0)
        agg["total_tokens"] += s.get("total_tokens", 0)
        agg["total_cost"] += s.get("total_cost", 0.0)
        agg["call_count"] += s.get("call_count", 0)
        agg["errors"] += s.get("errors", 0)
    return agg


def _now_ms() -> float:
    """Current time in milliseconds (perf_counter-based)."""
    return time.perf_counter() * 1000
