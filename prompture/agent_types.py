"""Shared types for the Agent framework.

Defines enums, dataclasses, and exceptions used by :class:`~prompture.agent.Agent`.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

DepsType = TypeVar("DepsType")


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

    Tools raise this to return an error string to the LLM.
    Output guardrails raise this to re-prompt the LLM.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class GuardrailError(Exception):
    """Raised when an input guardrail rejects the prompt entirely."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass
class RunContext(Generic[DepsType]):
    """Dependency-injection context available to tools and guardrails.

    Built at the start of each :meth:`Agent.run` invocation and passed
    automatically to tools whose first parameter is annotated as
    ``RunContext``.

    Attributes:
        deps: User-supplied dependencies (database handles, API clients, etc.).
        model: The model string used for this run.
        usage: Snapshot of :class:`UsageSession.summary()` at context-build time.
        messages: Copy of conversation history at context-build time.
        iteration: Current iteration index (0 at the start of the run).
        prompt: The original user prompt for this run.
    """

    deps: DepsType
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    prompt: str = ""


@dataclass
class AgentCallbacks:
    """Agent-level observability callbacks.

    Fired at the logical agent layer (steps, tool invocations, output),
    separate from :class:`~prompture.callbacks.DriverCallbacks` which
    fires at the HTTP/driver layer.
    """

    on_step: Callable[[AgentStep], None] | None = None
    on_tool_start: Callable[[str, dict[str, Any]], None] | None = None
    on_tool_end: Callable[[str, Any], None] | None = None
    on_iteration: Callable[[int], None] | None = None
    on_output: Callable[[AgentResult], None] | None = None


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
        run_usage: Per-run :class:`UsageSession` summary dict.
    """

    output: Any
    output_text: str
    messages: list[dict[str, Any]]
    usage: dict[str, Any]
    steps: list[AgentStep] = field(default_factory=list)
    all_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state: AgentState = AgentState.idle
    run_usage: dict[str, Any] = field(default_factory=dict)


class StreamEventType(str, enum.Enum):
    """Classification for events emitted during streaming agent execution."""

    text_delta = "text_delta"
    tool_call = "tool_call"
    tool_result = "tool_result"
    output = "output"


@dataclass
class StreamEvent:
    """A single event emitted during a streaming agent run.

    Attributes:
        event_type: The kind of event.
        data: Payload — ``str`` for text_delta, ``dict`` for tool_call/result,
            :class:`AgentResult` for output.
        step: Optional associated :class:`AgentStep`.
    """

    event_type: StreamEventType
    data: Any
    step: AgentStep | None = None
