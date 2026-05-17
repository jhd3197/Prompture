"""State containers and configuration types for :class:`DeepAgent`.

The deep-agent capabilities (planning todos, virtual filesystem, sub-agent
delegation, automatic summarization) share a single :class:`DeepAgentState`
instance which is mutated in-place by built-in tools and the summarization
middleware. :class:`DeepAgent` owns the state, exposes it on the result via
:class:`DeepAgentResult`, and resets it on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from .persona import Persona
from .tools_schema import ToolDefinition
from .types import AgentResult

TodoStatus = Literal["pending", "in_progress", "completed"]


@dataclass
class Todo:
    """A single planning item produced by the ``write_todos`` tool.

    Attributes:
        content: The action statement (e.g. ``"Find regulatory bodies for X"``).
        status: One of ``pending``, ``in_progress``, ``completed``.
        active_form: Optional present-continuous label for UI display
            while ``status == "in_progress"`` (e.g. ``"Finding regulatory bodies"``).
    """

    content: str
    status: TodoStatus = "pending"
    active_form: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "status": self.status, "active_form": self.active_form}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Todo:
        return cls(
            content=str(data.get("content", "")),
            status=data.get("status", "pending"),  # type: ignore[arg-type]
            active_form=data.get("active_form"),
        )


@dataclass
class SubAgentCallRecord:
    """Bookkeeping for a single ``task`` tool invocation."""

    subagent_name: str
    description: str
    result_summary: str
    usage: dict[str, Any] = field(default_factory=dict)
    iterations: int = 0


@dataclass
class SummaryEvent:
    """Recorded when :class:`SummarizationMiddleware` collapses old messages."""

    triggered_at_message_index: int
    summary_text: str
    summarized_message_count: int
    summarizer_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentSpec:
    """Declarative configuration for a sub-agent the ``task`` tool can dispatch to.

    Attributes:
        name: Identifier used as the ``subagent_type`` argument to ``task``.
        description: Surface description in the ``task`` tool description —
            the parent model uses this to decide when to delegate.
        system_prompt: System prompt for the sub-agent (string or :class:`Persona`).
        model: Optional model override. Inherits parent's model when ``None``.
        tools: Optional explicit tool list. See ``inherit_tools``.
        inherit_tools: When ``True`` (default) and ``tools is None``, the
            sub-agent receives the parent's user-supplied tools. Built-in
            deep-agent tools (write_todos, task, vfs) are NEVER inherited.
        inherit_vfs: When ``True``, the sub-agent gets VFS tools that read
            and write the *parent's* file dict. Default ``False`` —
            sub-agents normally start fresh.
        max_iterations: Maximum tool rounds for the sub-agent.
        max_tool_result_length: Optional override for tool result truncation.
    """

    name: str
    description: str
    system_prompt: Union[str, Persona]
    model: str | None = None
    tools: list[ToolDefinition] | None = None
    inherit_tools: bool = True
    inherit_vfs: bool = False
    max_iterations: int = 20
    max_tool_result_length: int | None = None


@dataclass
class DeepAgentState:
    """Shared mutable state across one :class:`DeepAgent` lifetime.

    Built-in tool factories close over a single instance of this class.
    Each tool mutates the appropriate field directly. The deep agent
    snapshots this state into :class:`DeepAgentResult` after every run.

    Attributes:
        todos: Current planning list (replaced wholesale by ``write_todos``).
        files: Virtual filesystem mapping ``path -> content``.
        sub_agent_calls: One :class:`SubAgentCallRecord` per ``task`` invocation.
        summary_events: One :class:`SummaryEvent` per summarization trigger.
    """

    todos: list[Todo] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    sub_agent_calls: list[SubAgentCallRecord] = field(default_factory=list)
    summary_events: list[SummaryEvent] = field(default_factory=list)

    def reset(self) -> None:
        """Clear all state (todos, files, sub-agent calls, summaries)."""
        self.todos.clear()
        self.files.clear()
        self.sub_agent_calls.clear()
        self.summary_events.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "todos": [t.to_dict() for t in self.todos],
            "files": dict(self.files),
            "sub_agent_calls": [
                {
                    "subagent_name": r.subagent_name,
                    "description": r.description,
                    "result_summary": r.result_summary,
                    "usage": r.usage,
                    "iterations": r.iterations,
                }
                for r in self.sub_agent_calls
            ],
            "summary_events": [
                {
                    "triggered_at_message_index": e.triggered_at_message_index,
                    "summary_text": e.summary_text,
                    "summarized_message_count": e.summarized_message_count,
                    "summarizer_usage": e.summarizer_usage,
                }
                for e in self.summary_events
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeepAgentState:
        state = cls()
        state.todos = [Todo.from_dict(t) for t in data.get("todos", [])]
        state.files = dict(data.get("files", {}))
        for r in data.get("sub_agent_calls", []):
            state.sub_agent_calls.append(
                SubAgentCallRecord(
                    subagent_name=r.get("subagent_name", ""),
                    description=r.get("description", ""),
                    result_summary=r.get("result_summary", ""),
                    usage=r.get("usage", {}),
                    iterations=int(r.get("iterations", 0)),
                )
            )
        for e in data.get("summary_events", []):
            state.summary_events.append(
                SummaryEvent(
                    triggered_at_message_index=int(e.get("triggered_at_message_index", 0)),
                    summary_text=e.get("summary_text", ""),
                    summarized_message_count=int(e.get("summarized_message_count", 0)),
                    summarizer_usage=e.get("summarizer_usage", {}),
                )
            )
        return state


@dataclass
class DeepAgentResult(AgentResult):
    """Extends :class:`AgentResult` with deep-agent state snapshots.

    Attributes inherited from :class:`AgentResult`:
        output, output_text, messages, usage, steps, all_tool_calls,
        state (lifecycle), run_usage.

    Additional attributes:
        deep_state: Snapshot of the :class:`DeepAgentState` after the run.
        todos: Convenience alias for ``deep_state.todos``.
        files: Convenience alias for ``deep_state.files``.
        sub_agent_calls: Convenience alias for ``deep_state.sub_agent_calls``.
        summary_events: Convenience alias for ``deep_state.summary_events``.
    """

    deep_state: DeepAgentState = field(default_factory=DeepAgentState)

    @property
    def todos(self) -> list[Todo]:
        return self.deep_state.todos

    @property
    def files(self) -> dict[str, str]:
        return self.deep_state.files

    @property
    def sub_agent_calls(self) -> list[SubAgentCallRecord]:
        return self.deep_state.sub_agent_calls

    @property
    def summary_events(self) -> list[SummaryEvent]:
        return self.deep_state.summary_events
