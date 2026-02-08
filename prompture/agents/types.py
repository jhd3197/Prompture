"""Shared types for the Agent framework.

Defines enums, dataclasses, and exceptions used by :class:`~prompture.agent.Agent`.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar, Union

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


class ApprovalRequired(Exception):
    """Raised by tools that require human approval before execution.

    When a tool raises this exception, the agent will invoke the
    ``on_approval_needed`` callback if configured. If the callback
    returns True, the tool will be executed; if False, the tool
    execution will be skipped and an error message returned to the LLM.

    Attributes:
        tool_name: Name of the tool requesting approval.
        action: Description of the action requiring approval.
        details: Additional details about what will be executed.
    """

    def __init__(
        self,
        tool_name: str,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.action = action
        self.details = details or {}
        message = f"Tool '{tool_name}' requires approval: {action}"
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

    Attributes:
        on_step: Called for each step during execution.
        on_tool_start: Called before a tool is invoked with (name, args).
        on_tool_end: Called after a tool completes with (name, result).
        on_iteration: Called at the start of each iteration with the index.
        on_output: Called when the agent produces final output.
        on_thinking: Called when the agent emits thinking/reasoning content.
            The callback receives the thinking text (e.g., content within
            <think> tags for models that support chain-of-thought).
        on_approval_needed: Called when a tool raises ApprovalRequired.
            The callback receives (tool_name, action, details) and should
            return True to approve execution or False to deny.
        on_message: Called with the final output text string after a run
            completes.  Useful for forwarding the response to a UI
            without processing the full :class:`AgentResult`.
    """

    on_step: Union[Callable[[AgentStep], None], Callable[[AgentStep], Awaitable[None]], None] = None
    on_tool_start: Union[Callable[[str, dict[str, Any]], None], Callable[[str, dict[str, Any]], Awaitable[None]], None] = None
    on_tool_end: Union[Callable[[str, Any], None], Callable[[str, Any], Awaitable[None]], None] = None
    on_iteration: Union[Callable[[int], None], Callable[[int], Awaitable[None]], None] = None
    on_output: Union[Callable[[AgentResult], None], Callable[[AgentResult], Awaitable[None]], None] = None
    on_thinking: Union[Callable[[str], None], Callable[[str], Awaitable[None]], None] = None
    on_approval_needed: Union[Callable[[str, str, dict[str, Any]], bool], Callable[[str, str, dict[str, Any]], Awaitable[bool]], None] = None
    on_message: Union[Callable[[str], None], Callable[[str], Awaitable[None]], None] = None


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
    usage: dict[str, Any] | None = None


@dataclass
class AgentResult:
    """The outcome of an :meth:`Agent.run` invocation.

    Attributes:
        output: Parsed Pydantic model instance (if ``output_type`` is set)
            or the raw text response.
        output_text: The raw text from the final LLM response.
        messages: Full conversation message history from the run.
        usage: Per-call usage from the conversation. Contains token counts
            and cost for this specific run, accumulated across tool rounds
            within a single ``ask()`` call.
            Keys: ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
            ``cost``, ``turns``.
        steps: Ordered list of :class:`AgentStep` recorded during the run.
        all_tool_calls: Flat list of tool-call dicts extracted from messages.
        state: Final :class:`AgentState` after the run completes.
        run_usage: Aggregated session usage from
            :meth:`~prompture.session.UsageSession.summary`. Contains totals
            across all API calls during the run, including timing metrics
            and per-model breakdowns.
            Keys: ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
            ``cost``, ``call_count``, ``errors``, ``total_elapsed_ms``,
            ``tokens_per_second``, ``latency_stats``, ``per_model``.
    """

    output: Any
    output_text: str
    messages: list[dict[str, Any]]
    usage: dict[str, Any]
    steps: list[AgentStep] = field(default_factory=list)
    all_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state: AgentState = AgentState.idle
    run_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_messages: bool = True) -> dict[str, Any]:
        """Convert this result to a dictionary for serialization.

        Args:
            include_messages: Whether to include the full message history.

        Returns:
            Dictionary representation of this result.
        """
        data: dict[str, Any] = {
            "output": str(self.output) if self.output is not None else None,
            "output_text": self.output_text,
            "state": self.state.value if hasattr(self.state, "value") else str(self.state),
            "usage": self.usage,
            "run_usage": self.run_usage,
            "steps": [
                {
                    "step_type": s.step_type.value if hasattr(s.step_type, "value") else str(s.step_type),
                    "timestamp": s.timestamp,
                    "content": s.content,
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "tool_result": s.tool_result,
                    "duration_ms": s.duration_ms,
                    "usage": s.usage,
                }
                for s in self.steps
            ],
            "all_tool_calls": self.all_tool_calls,
            "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }

        if include_messages:
            data["messages"] = self.messages

        return data

    def export_json(self, include_messages: bool = True) -> str:
        """Export this result to a JSON string.

        Args:
            include_messages: Whether to include the full message history.

        Returns:
            JSON string representation of this result.

        Example::

            result = agent.run("What is 2+2?")
            json_str = result.export_json()
            with open("agent_history.json", "w") as f:
                f.write(json_str)
        """
        return json.dumps(self.to_dict(include_messages=include_messages), indent=2, default=str)


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
