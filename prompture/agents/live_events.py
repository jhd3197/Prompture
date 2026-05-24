"""Live event types for streaming agentic tool calling.

These events flow from a driver's :meth:`generate_messages_with_tools_stream`
up through :meth:`Conversation.ask_live` and out to the caller, giving the
"Claude Code feel" where the model narrates between tool calls instead of
buffering tools and text into a single turn.

Event ordering for a single tool-using turn::

    AssistantTurnStart
      [TextDelta | ThinkingDelta]*         # narration before any tool
      ToolUseStart(id, name)
        ToolInputDelta(id, fragment)*       # input JSON streamed in chunks
      ToolUseStop(id, input)                # final parsed input dict
      [TextDelta | ThinkingDelta]*         # optional narration between tools
      ... (more ToolUseStart / Stop pairs)
    MessageStop(usage, stop_reason)
    ToolResult(id, name, output, is_error)* # one per executed tool (yielded
                                            # by Conversation, not driver)
    # loop repeats with another AssistantTurnStart until the model stops
    # asking for tools, then:
    TurnComplete(usage)

The ``ToolResult`` event is emitted by the **conversation layer** after it
executes the tool — drivers never produce it. Everything else is produced
by the driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union


@dataclass(frozen=True)
class AssistantTurnStart:
    """Marks the start of an assistant turn (one LLM call)."""

    event_type: Literal["assistant_turn_start"] = field(default="assistant_turn_start", init=False)
    turn_index: int = 0


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant text being generated."""

    text: str
    event_type: Literal["text_delta"] = field(default="text_delta", init=False)


@dataclass(frozen=True)
class ThinkingDelta:
    """A fragment of assistant reasoning/thinking content."""

    text: str
    event_type: Literal["thinking_delta"] = field(default="thinking_delta", init=False)


@dataclass(frozen=True)
class ToolUseStart:
    """The model has decided to call a tool. Input JSON streams next."""

    id: str
    name: str
    event_type: Literal["tool_use_start"] = field(default="tool_use_start", init=False)


@dataclass(frozen=True)
class ToolInputDelta:
    """A fragment of the tool's input JSON as it's generated."""

    id: str
    fragment: str
    event_type: Literal["tool_input_delta"] = field(default="tool_input_delta", init=False)


@dataclass(frozen=True)
class ToolUseStop:
    """The tool's input is complete. ``input`` is the parsed dict."""

    id: str
    name: str
    input: dict[str, Any]
    event_type: Literal["tool_use_stop"] = field(default="tool_use_stop", init=False)


@dataclass(frozen=True)
class ToolResult:
    """The conversation layer executed the tool and got this output."""

    id: str
    name: str
    output: str
    is_error: bool = False
    event_type: Literal["tool_result"] = field(default="tool_result", init=False)


@dataclass(frozen=True)
class MessageStop:
    """End of one assistant turn. ``stop_reason`` matches provider semantics
    (``end_turn``, ``tool_use``, ``max_tokens``, ``stop``, ``length`` …)."""

    stop_reason: str
    usage: dict[str, Any] = field(default_factory=dict)
    event_type: Literal["message_stop"] = field(default="message_stop", init=False)


@dataclass(frozen=True)
class TurnComplete:
    """End of the whole interaction — no more tool calls pending."""

    usage: dict[str, Any] = field(default_factory=dict)
    event_type: Literal["turn_complete"] = field(default="turn_complete", init=False)


LiveEvent = Union[
    AssistantTurnStart,
    TextDelta,
    ThinkingDelta,
    ToolUseStart,
    ToolInputDelta,
    ToolUseStop,
    ToolResult,
    MessageStop,
    TurnComplete,
]
"""Discriminated union of every event the live tool-calling stream emits.

Use the ``event_type`` field for dispatch::

    for ev in agent.run_live("..."):
        match ev.event_type:
            case "text_delta":
                print(ev.text, end="", flush=True)
            case "tool_use_start":
                print(f"\\n[calling {ev.name}({ev.id})]")
            case "tool_result":
                print(f"[result: {ev.output[:80]}]")
            case "turn_complete":
                print(f"\\n[done, ${ev.usage.get('cost', 0):.4f}]")
"""


__all__ = [
    "AssistantTurnStart",
    "LiveEvent",
    "MessageStop",
    "TextDelta",
    "ThinkingDelta",
    "ToolInputDelta",
    "ToolResult",
    "ToolUseStart",
    "ToolUseStop",
    "TurnComplete",
]
