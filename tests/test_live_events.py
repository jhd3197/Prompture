"""Tests for the live event stream (interleaved tool calling)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from prompture.agents.agent import Agent
from prompture.agents.async_agent import AsyncAgent
from prompture.agents.async_conversation import AsyncConversation
from prompture.agents.conversation import Conversation
from prompture.agents.live_events import (
    AssistantTurnStart,
    LiveEvent,
    MessageStop,
    TextDelta,
    ThinkingDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
    ToolUseStop,
    TurnComplete,
)
from prompture.agents.tools_schema import ToolRegistry
from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver


# ---------------------------------------------------------------------------
# Mock drivers
# ---------------------------------------------------------------------------


class BufferedToolDriver(Driver):
    """Driver that only implements the buffered (non-streaming) tool method.

    Exercises the base class default ``generate_messages_with_tools_stream``
    fallback that wraps buffered responses into synthetic events.
    """

    supports_messages = True
    supports_tool_use = True
    supports_streaming = False
    supports_streaming_tool_use = False

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._idx = 0
        self.model = "mock-buffered"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._next()

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._next()

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return self._next()

    def _next(self) -> dict[str, Any]:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r


class NativeStreamingDriver(Driver):
    """Driver that natively emits LiveEvent sequences per turn.

    Used to verify the conversation layer correctly forwards events,
    executes tools, and loops until ``MessageStop`` reports no pending
    tool calls.
    """

    supports_messages = True
    supports_tool_use = True
    supports_streaming = True
    supports_streaming_tool_use = True

    def __init__(self, turn_event_sequences: list[list[LiveEvent]]):
        self._turns = list(turn_event_sequences)
        self._idx = 0
        self.model = "mock-native-stream"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "n/a", "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}}

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self.generate("", options)

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        raise AssertionError("ask_live should call generate_messages_with_tools_stream, not the buffered method")

    def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[LiveEvent]:
        events = self._turns[min(self._idx, len(self._turns) - 1)]
        self._idx += 1
        yield from events


# ---------------------------------------------------------------------------
# Default-impl fallback tests (base class wraps buffered)
# ---------------------------------------------------------------------------


def lookup_population(city: str) -> str:
    """Return population for a city."""
    return {"Tokyo": "13960000", "Paris": "2161000"}.get(city, "Unknown")


class TestBufferedFallback:
    def test_buffered_driver_wrapped_into_event_sequence(self):
        driver = BufferedToolDriver(
            [
                {
                    "text": "Looking up Tokyo.",
                    "tool_calls": [{"id": "call_1", "name": "lookup_population", "arguments": {"city": "Tokyo"}}],
                    "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                    "stop_reason": "tool_use",
                },
                {
                    "text": "Tokyo has 13,960,000 people.",
                    "tool_calls": [],
                    "meta": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28, "cost": 0.002},
                    "stop_reason": "end_turn",
                },
            ]
        )
        conv = Conversation(driver=driver, tools=self._tools())

        events = list(conv.ask_live("How many people live in Tokyo?"))

        kinds = [e.event_type for e in events]
        # Two turns expected: turn 1 has tool call, turn 2 is the final answer.
        assert kinds.count("assistant_turn_start") == 2
        assert kinds.count("tool_use_start") == 1
        assert kinds.count("tool_use_stop") == 1
        assert kinds.count("tool_result") == 1
        assert kinds.count("turn_complete") == 1

        # The tool result must be emitted between the two assistant turns.
        tool_result_idx = next(i for i, e in enumerate(events) if e.event_type == "tool_result")
        turn_starts = [i for i, e in enumerate(events) if e.event_type == "assistant_turn_start"]
        assert turn_starts[0] < tool_result_idx < turn_starts[1]

        # Tool result content must include the looked-up population.
        tool_result = next(e for e in events if isinstance(e, ToolResult))
        assert "13960000" in tool_result.output
        assert tool_result.is_error is False

    def test_buffered_driver_emits_text_before_tool_use_start(self):
        """In the fallback wrapper, narration text is emitted before the
        synthesized ToolUseStart so the consumer's UI still shows context
        before each tool fires."""
        driver = BufferedToolDriver(
            [
                {
                    "text": "Let me check.",
                    "tool_calls": [{"id": "c", "name": "lookup_population", "arguments": {"city": "Paris"}}],
                    "meta": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "cost": 0.0},
                    "stop_reason": "tool_use",
                },
                {
                    "text": "Done.",
                    "tool_calls": [],
                    "meta": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.0},
                    "stop_reason": "end_turn",
                },
            ]
        )
        conv = Conversation(driver=driver, tools=self._tools())
        events = list(conv.ask_live("Paris?"))

        # First TextDelta must come before the first ToolUseStart.
        text_idx = next(i for i, e in enumerate(events) if isinstance(e, TextDelta))
        tool_idx = next(i for i, e in enumerate(events) if isinstance(e, ToolUseStart))
        assert text_idx < tool_idx
        assert events[text_idx].text == "Let me check."

    def _tools(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(lookup_population)
        return reg


# ---------------------------------------------------------------------------
# Native-streaming conversation tests
# ---------------------------------------------------------------------------


class TestNativeStreaming:
    def test_interleaved_text_and_tool_in_one_turn(self):
        """A driver that streams `text → tool_use → text → tool_use → message_stop`
        within ONE turn should flow through ask_live untouched, with the
        conversation layer inserting one ToolResult after each tool_use_stop."""
        turn1 = [
            AssistantTurnStart(turn_index=0),
            TextDelta(text="Looking up "),
            TextDelta(text="Tokyo..."),
            ToolUseStart(id="t1", name="lookup_population"),
            ToolInputDelta(id="t1", fragment='{"city"'),
            ToolInputDelta(id="t1", fragment=': "Tokyo"}'),
            ToolUseStop(id="t1", name="lookup_population", input={"city": "Tokyo"}),
            TextDelta(text=" Now Paris..."),
            ToolUseStart(id="t2", name="lookup_population"),
            ToolUseStop(id="t2", name="lookup_population", input={"city": "Paris"}),
            MessageStop(
                stop_reason="tool_use",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
            ),
        ]
        turn2 = [
            AssistantTurnStart(turn_index=1),
            TextDelta(text="Tokyo has more people than Paris."),
            MessageStop(
                stop_reason="end_turn",
                usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28, "cost": 0.002},
            ),
        ]
        driver = NativeStreamingDriver([turn1, turn2])
        reg = ToolRegistry()
        reg.register(lookup_population)
        conv = Conversation(driver=driver, tools=reg)

        events = list(conv.ask_live("Compare Tokyo and Paris."))
        kinds = [e.event_type for e in events]

        # Both tool_uses must be followed by a tool_result (in order).
        assert kinds.count("tool_use_stop") == 2
        assert kinds.count("tool_result") == 2

        # Both tool_results carry the looked-up populations.
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        outputs = [r.output for r in tool_results]
        assert "13960000" in outputs[0]
        assert "2161000" in outputs[1]

        # Interleaved text BEFORE tools and BETWEEN tools is preserved.
        text_chunks = [e.text for e in events if isinstance(e, TextDelta)]
        joined_first_turn = "".join(text_chunks[:3])
        assert "Looking up Tokyo..." in joined_first_turn
        assert any("Now Paris" in t for t in text_chunks)

        # TurnComplete is fired exactly once at the end.
        assert kinds.count("turn_complete") == 1
        assert kinds[-1] == "turn_complete"

    def test_tool_input_deltas_forwarded(self):
        """Input-JSON fragments should reach the consumer in order so a UI
        can render the tool's arguments as they're generated."""
        turn1 = [
            AssistantTurnStart(turn_index=0),
            ToolUseStart(id="x", name="lookup_population"),
            ToolInputDelta(id="x", fragment='{"cit'),
            ToolInputDelta(id="x", fragment='y": "Paris"}'),
            ToolUseStop(id="x", name="lookup_population", input={"city": "Paris"}),
            MessageStop(stop_reason="tool_use", usage={"total_tokens": 5, "cost": 0.0}),
        ]
        turn2 = [
            AssistantTurnStart(turn_index=1),
            TextDelta(text="Paris has 2,161,000."),
            MessageStop(stop_reason="end_turn", usage={"total_tokens": 6, "cost": 0.0}),
        ]
        driver = NativeStreamingDriver([turn1, turn2])
        reg = ToolRegistry()
        reg.register(lookup_population)
        conv = Conversation(driver=driver, tools=reg)

        events = list(conv.ask_live("Paris?"))
        fragments = [e.fragment for e in events if isinstance(e, ToolInputDelta)]
        assert fragments == ['{"cit', 'y": "Paris"}']
        assert "".join(fragments) == '{"city": "Paris"}'

    def test_tool_error_marks_is_error(self):
        def broken(_x: str) -> str:
            """Always raises."""
            raise RuntimeError("kaboom")

        turn1 = [
            AssistantTurnStart(turn_index=0),
            ToolUseStart(id="b", name="broken"),
            ToolUseStop(id="b", name="broken", input={"_x": "foo"}),
            MessageStop(stop_reason="tool_use", usage={"total_tokens": 3, "cost": 0.0}),
        ]
        turn2 = [
            AssistantTurnStart(turn_index=1),
            TextDelta(text="Sorry, that tool failed."),
            MessageStop(stop_reason="end_turn", usage={"total_tokens": 4, "cost": 0.0}),
        ]
        driver = NativeStreamingDriver([turn1, turn2])
        reg = ToolRegistry()
        reg.register(broken)
        conv = Conversation(driver=driver, tools=reg)

        events = list(conv.ask_live("Try the broken tool."))
        tool_results = [e for e in events if isinstance(e, ToolResult)]
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "kaboom" in tool_results[0].output

    def test_no_tools_path_emits_text_only(self):
        """When the conversation has no tools registered, ask_live should
        delegate to streaming text and finish with TurnComplete."""

        class _TextStreamDriver(Driver):
            supports_messages = True
            supports_streaming = True
            model = "mock-text"

            def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
                return {
                    "text": "hello",
                    "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
                }

            def generate_messages(self, messages, options):
                return self.generate("", options)

            def generate_messages_stream(self, messages, options):
                yield {"type": "delta", "text": "hello "}
                yield {"type": "delta", "text": "world"}
                yield {
                    "type": "done",
                    "text": "hello world",
                    "meta": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                        "cost": 0.0,
                    },
                }

        conv = Conversation(driver=_TextStreamDriver())
        events = list(conv.ask_live("hi"))
        kinds = [e.event_type for e in events]
        assert kinds[0] == "text_delta"
        assert kinds[-1] == "turn_complete"
        joined = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert joined == "hello world"


# ---------------------------------------------------------------------------
# Agent.run_live tests
# ---------------------------------------------------------------------------


class TestAgentRunLive:
    def test_run_live_yields_events_and_returns_agent_result(self):
        turn1 = [
            AssistantTurnStart(turn_index=0),
            TextDelta(text="Checking."),
            ToolUseStart(id="a", name="lookup_population"),
            ToolUseStop(id="a", name="lookup_population", input={"city": "Tokyo"}),
            MessageStop(stop_reason="tool_use", usage={"total_tokens": 5, "cost": 0.001}),
        ]
        turn2 = [
            AssistantTurnStart(turn_index=1),
            TextDelta(text="Tokyo: 13,960,000."),
            MessageStop(stop_reason="end_turn", usage={"total_tokens": 6, "cost": 0.001}),
        ]
        driver = NativeStreamingDriver([turn1, turn2])
        agent = Agent("mock/mock", driver=driver, tools=[lookup_population])

        streamed = agent.run_live("Tokyo?")
        events = list(streamed)
        kinds = [e.event_type for e in events]
        assert kinds.count("text_delta") >= 2
        assert kinds.count("tool_use_stop") == 1
        assert kinds.count("tool_result") == 1
        assert kinds[-1] == "turn_complete"

        result = streamed.result
        assert result is not None
        # The final assistant turn's text becomes output_text.
        assert "Tokyo" in result.output_text

    def test_run_live_without_tools_works(self):
        class _Stream(Driver):
            supports_messages = True
            supports_streaming = True
            model = "mock-text"

            def generate(self, prompt, options):
                return {"text": "ok", "meta": {"total_tokens": 1, "cost": 0.0}}

            def generate_messages(self, messages, options):
                return self.generate("", options)

            def generate_messages_stream(self, messages, options):
                for piece in ("hel", "lo"):
                    yield {"type": "delta", "text": piece}
                yield {"type": "done", "text": "hello", "meta": {"total_tokens": 2, "cost": 0.0}}

        agent = Agent("mock/mock", driver=_Stream())
        events = list(agent.run_live("hi"))
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "hello"


# ---------------------------------------------------------------------------
# Smoke tests for event types
# ---------------------------------------------------------------------------


class TestEventTypes:
    def test_event_types_have_stable_discriminator(self):
        cases: list[tuple[type, str]] = [
            (AssistantTurnStart, "assistant_turn_start"),
            (TextDelta, "text_delta"),
            (ThinkingDelta, "thinking_delta"),
            (ToolUseStart, "tool_use_start"),
            (ToolInputDelta, "tool_input_delta"),
            (ToolUseStop, "tool_use_stop"),
            (ToolResult, "tool_result"),
            (MessageStop, "message_stop"),
            (TurnComplete, "turn_complete"),
        ]
        for cls, expected in cases:
            # All events carry their discriminator as a class default field.
            assert getattr(cls, "__dataclass_fields__")["event_type"].default == expected

    def test_event_types_are_frozen(self):
        e = TextDelta(text="x")
        with pytest.raises(Exception):  # noqa: B017 — frozen dataclass raises FrozenInstanceError
            e.text = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Async mocks
# ---------------------------------------------------------------------------


class AsyncBufferedToolDriver(AsyncDriver):
    """Async driver implementing only the buffered tool method, exercising
    the AsyncDriver base-class default that wraps it into synthetic events."""

    supports_messages = True
    supports_tool_use = True
    supports_streaming = False
    supports_streaming_tool_use = False

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._idx = 0
        self.model = "mock-async-buffered"

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._next()

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._next()

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return self._next()

    def _next(self) -> dict[str, Any]:
        r = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return r


class AsyncNativeStreamingDriver(AsyncDriver):
    """Async driver that natively emits a fixed LiveEvent sequence per turn."""

    supports_messages = True
    supports_tool_use = True
    supports_streaming = True
    supports_streaming_tool_use = True

    def __init__(self, turn_event_sequences: list[list[LiveEvent]]):
        self._turns = list(turn_event_sequences)
        self._idx = 0
        self.model = "mock-async-native"

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "n/a", "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}}

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return await self.generate("", options)

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        raise AssertionError("ask_live should use generate_messages_with_tools_stream, not the buffered method")

    async def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[LiveEvent]:
        events = self._turns[min(self._idx, len(self._turns) - 1)]
        self._idx += 1
        for ev in events:
            yield ev


# ---------------------------------------------------------------------------
# Async ask_live tests
# ---------------------------------------------------------------------------


class TestAsyncAskLive:
    @pytest.mark.asyncio
    async def test_async_buffered_fallback_yields_event_sequence(self):
        driver = AsyncBufferedToolDriver(
            [
                {
                    "text": "Looking up Tokyo.",
                    "tool_calls": [{"id": "call_1", "name": "lookup_population", "arguments": {"city": "Tokyo"}}],
                    "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
                    "stop_reason": "tool_use",
                },
                {
                    "text": "Tokyo has 13,960,000 people.",
                    "tool_calls": [],
                    "meta": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28, "cost": 0.002},
                    "stop_reason": "end_turn",
                },
            ]
        )
        reg = ToolRegistry()
        reg.register(lookup_population)
        conv = AsyncConversation(driver=driver, tools=reg)

        events = [ev async for ev in conv.ask_live("Tokyo?")]
        kinds = [e.event_type for e in events]
        assert kinds.count("assistant_turn_start") == 2
        assert kinds.count("tool_use_start") == 1
        assert kinds.count("tool_use_stop") == 1
        assert kinds.count("tool_result") == 1
        assert kinds[-1] == "turn_complete"
        tool_result = next(e for e in events if isinstance(e, ToolResult))
        assert "13960000" in tool_result.output

    @pytest.mark.asyncio
    async def test_async_native_interleaved_streaming(self):
        turn1 = [
            AssistantTurnStart(turn_index=0),
            TextDelta(text="Looking up "),
            TextDelta(text="Tokyo..."),
            ToolUseStart(id="t1", name="lookup_population"),
            ToolInputDelta(id="t1", fragment='{"city"'),
            ToolInputDelta(id="t1", fragment=': "Tokyo"}'),
            ToolUseStop(id="t1", name="lookup_population", input={"city": "Tokyo"}),
            TextDelta(text=" Now Paris..."),
            ToolUseStart(id="t2", name="lookup_population"),
            ToolUseStop(id="t2", name="lookup_population", input={"city": "Paris"}),
            MessageStop(
                stop_reason="tool_use",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001},
            ),
        ]
        turn2 = [
            AssistantTurnStart(turn_index=1),
            TextDelta(text="Tokyo has more people than Paris."),
            MessageStop(
                stop_reason="end_turn",
                usage={"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28, "cost": 0.002},
            ),
        ]
        driver = AsyncNativeStreamingDriver([turn1, turn2])
        reg = ToolRegistry()
        reg.register(lookup_population)
        conv = AsyncConversation(driver=driver, tools=reg)

        events = [ev async for ev in conv.ask_live("Compare Tokyo and Paris.")]
        kinds = [e.event_type for e in events]
        assert kinds.count("tool_use_stop") == 2
        assert kinds.count("tool_result") == 2
        assert kinds[-1] == "turn_complete"

        tool_results = [e for e in events if isinstance(e, ToolResult)]
        outputs = [r.output for r in tool_results]
        assert "13960000" in outputs[0]
        assert "2161000" in outputs[1]


# ---------------------------------------------------------------------------
# AsyncAgent.run_live tests
# ---------------------------------------------------------------------------


class TestAsyncAgentRunLive:
    @pytest.mark.asyncio
    async def test_async_run_live_yields_events_and_returns_result(self):
        turn1 = [
            AssistantTurnStart(turn_index=0),
            TextDelta(text="Checking."),
            ToolUseStart(id="a", name="lookup_population"),
            ToolUseStop(id="a", name="lookup_population", input={"city": "Tokyo"}),
            MessageStop(stop_reason="tool_use", usage={"total_tokens": 5, "cost": 0.001}),
        ]
        turn2 = [
            AssistantTurnStart(turn_index=1),
            TextDelta(text="Tokyo: 13,960,000."),
            MessageStop(stop_reason="end_turn", usage={"total_tokens": 6, "cost": 0.001}),
        ]
        driver = AsyncNativeStreamingDriver([turn1, turn2])
        agent = AsyncAgent("mock/mock", driver=driver, tools=[lookup_population])

        streamed = agent.run_live("Tokyo?")
        collected: list[Any] = []
        async for ev in streamed:
            collected.append(ev)

        kinds = [e.event_type for e in collected]
        assert kinds.count("text_delta") >= 2
        assert kinds.count("tool_use_stop") == 1
        assert kinds.count("tool_result") == 1
        assert kinds[-1] == "turn_complete"

        result = streamed.result
        assert result is not None
        assert "Tokyo" in result.output_text

    @pytest.mark.asyncio
    async def test_async_run_live_without_tools(self):
        class _AsyncTextStream(AsyncDriver):
            supports_messages = True
            supports_streaming = True
            model = "mock-text-async"

            async def generate(self, prompt, options):
                return {"text": "ok", "meta": {"total_tokens": 1, "cost": 0.0}}

            async def generate_messages(self, messages, options):
                return await self.generate("", options)

            async def generate_messages_stream(self, messages, options):
                for piece in ("hel", "lo"):
                    yield {"type": "delta", "text": piece}
                yield {"type": "done", "text": "hello", "meta": {"total_tokens": 2, "cost": 0.0}}

        agent = AsyncAgent("mock/mock", driver=_AsyncTextStream())
        events: list[Any] = []
        async for ev in agent.run_live("hi"):
            events.append(ev)
        text = "".join(e.text for e in events if isinstance(e, TextDelta))
        assert text == "hello"
