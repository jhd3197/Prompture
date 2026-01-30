"""Shared types for the Agent framework.

Defines enums, dataclasses, and exceptions used by :class:`~prompture.agent.Agent`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class AgentState(enum.Enum):
    """Lifecycle state of an Agent run."""

    idle = "idle"
    running = "running"
    stopped = "stopped"
    errored = "errored"


class StepType(enum.Enum):
    """Classification for individual steps within an Agent run."""

    think = "think"
    tool_call = "tool_call"
    tool_result = "tool_result"
    output = "output"


class ModelRetry(Exception):
    """Raised to feed an error message back to the LLM for retry.

    Used in Phase 3b guardrails; defined here so it can be imported
    without pulling in the full Agent module.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass
class AgentStep:
    """A single step recorded during an Agent run."""

    step_type: StepType
    timestamp: float
    content: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: Any = None
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """The outcome of an :meth:`Agent.run` invocation.

    Attributes:
        output: Parsed Pydantic model instance (if ``output_type`` is set)
            or the raw text response.
        output_text: The raw text from the final LLM response.
        messages: Full conversation message history from the run.
        usage: Accumulated token/cost totals.
        steps: Ordered list of :class:`AgentStep` recorded during the run.
        all_tool_calls: Flat list of tool-call dicts extracted from messages.
        state: Final :class:`AgentState` after the run completes.
    """

    output: Any
    output_text: str
    messages: list[dict[str, Any]]
    usage: dict[str, Any]
    steps: list[AgentStep] = field(default_factory=list)
    all_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state: AgentState = AgentState.idle
