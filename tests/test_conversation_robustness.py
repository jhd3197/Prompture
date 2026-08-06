"""Robustness tests for Conversation/AsyncConversation tool loops.

Covers the tool-calling audit contracts: malformed/truncated argument
feedback (C1/C3), cooperative stop (C2), graceful max-rounds final
answer (C4), per-tool timeouts, full-result preservation, budget
enforcement in the event path, parallel async execution ordering, and
boundary-aware history trimming.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from prompture.agents.async_conversation import AsyncConversation
from prompture.agents.conversation import Conversation
from prompture.agents.tools_schema import ToolRegistry
from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver
from prompture.exceptions import BudgetExceededError

# ---------------------------------------------------------------------------
# Mock drivers
# ---------------------------------------------------------------------------


def _meta(cost: float = 0.0) -> dict[str, Any]:
    return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": cost}


def _tool_call_resp(*tool_calls: dict[str, Any], cost: float = 0.0) -> dict[str, Any]:
    return {"text": "", "meta": _meta(cost), "tool_calls": list(tool_calls), "stop_reason": "tool_use"}


def _text_resp(text: str, cost: float = 0.0) -> dict[str, Any]:
    return {"text": text, "meta": _meta(cost), "tool_calls": [], "stop_reason": "end_turn"}


class MockToolDriver(Driver):
    supports_messages = True
    supports_tool_use = True

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._call_idx = 0

    def generate(self, prompt, options):
        return self._get_next()

    def generate_messages(self, messages, options):
        return self._get_next()

    def generate_messages_with_tools(self, messages, tools, options):
        return self._get_next()

    def _get_next(self):
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


class MockAsyncToolDriver(AsyncDriver):
    supports_messages = True
    supports_tool_use = True

    def __init__(self, responses: list[dict[str, Any]]):
        self._responses = list(responses)
        self._call_idx = 0

    async def generate(self, prompt, options):
        return self._get_next()

    async def generate_messages(self, messages, options):
        return self._get_next()

    async def generate_messages_with_tools(self, messages, tools, options):
        return self._get_next()

    def _get_next(self):
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp


def _echo_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def echo(text: str) -> str:
        """Echo text back."""
        return text

    reg.register(echo)
    return reg


def _tool_messages(conv) -> list[dict[str, Any]]:
    return [m for m in conv.messages if m.get("role") == "tool"]


# ---------------------------------------------------------------------------
# Unknown tool / missing argument feedback
# ---------------------------------------------------------------------------


class TestErrorFeedback:
    def test_unknown_tool_feeds_error_and_completes(self):
        """Driver requests an unregistered tool -> error tool message -> loop continues."""
        responses = [
            _tool_call_resp({"id": "call_1", "name": "ghost_tool", "arguments": {}}),
            _text_resp("Sorry, that tool does not exist."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=_echo_registry())
        result = conv.ask("Call ghost_tool")

        assert result == "Sorry, that tool does not exist."
        tool_msgs = _tool_messages(conv)
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert "not registered" in tool_msgs[0]["content"]

    def test_missing_required_argument_feedback(self):
        """A call missing a required arg gets the validation error fed back, not a crash."""
        responses = [
            _tool_call_resp({"id": "call_1", "name": "echo", "arguments": {}}),
            _text_resp("I need the text argument."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=_echo_registry())
        result = conv.ask("Call echo with nothing")

        assert result == "I need the text argument."
        tool_msgs = _tool_messages(conv)
        assert len(tool_msgs) == 1
        assert "Missing required argument(s) for 'echo'" in tool_msgs[0]["content"]


# ---------------------------------------------------------------------------
# C1/C3: malformed / truncated arguments are not executed
# ---------------------------------------------------------------------------


class TestMalformedArguments:
    def test_arguments_error_skips_execution(self):
        called = False

        def echo(text: str) -> str:
            """Echo text back."""
            nonlocal called
            called = True
            return text

        reg = ToolRegistry()
        reg.register(echo)

        responses = [
            _tool_call_resp(
                {
                    "id": "call_1",
                    "name": "echo",
                    "arguments": {},
                    "arguments_error": "invalid JSON at offset 12",
                }
            ),
            _text_resp("Retrying with valid arguments."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=reg)
        result = conv.ask("Call echo")

        assert result == "Retrying with valid arguments."
        assert called is False
        tool_msgs = _tool_messages(conv)
        assert len(tool_msgs) == 1
        assert "could not be parsed" in tool_msgs[0]["content"]
        assert "invalid JSON at offset 12" in tool_msgs[0]["content"]
        # Full results still keyed by the tool call id
        assert conv._full_tool_results["call_1"] == tool_msgs[0]["content"]

    def test_truncated_arguments_skip_execution(self):
        called = False

        def echo(text: str) -> str:
            """Echo text back."""
            nonlocal called
            called = True
            return text

        reg = ToolRegistry()
        reg.register(echo)

        responses = [
            _tool_call_resp({"id": "call_1", "name": "echo", "arguments": {}, "truncated": True}),
            _text_resp("Recovered."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=reg)
        result = conv.ask("Call echo")

        assert result == "Recovered."
        assert called is False
        assert "were truncated" in _tool_messages(conv)[0]["content"]


# ---------------------------------------------------------------------------
# Tool result truncation vs. full-result preservation
# ---------------------------------------------------------------------------


class TestToolResultTruncation:
    def test_oversized_result_truncated_in_history_full_in_store(self):
        big = "x" * 500

        def big_tool() -> str:
            """Return a large payload."""
            return big

        reg = ToolRegistry()
        reg.register(big_tool)

        responses = [
            _tool_call_resp({"id": "call_1", "name": "big_tool", "arguments": {}}),
            _text_resp("Done."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=reg, max_tool_result_length=100)
        result = conv.ask("Call big_tool")

        assert result == "Done."
        tool_msg = _tool_messages(conv)[0]
        # History sees the truncated version
        assert len(tool_msg["content"]) < 200
        assert "result truncated" in tool_msg["content"]
        # The full result is preserved for step extraction
        assert conv._full_tool_results["call_1"] == big

    def test_default_max_tool_result_length(self):
        conv = Conversation(driver=MockToolDriver([_text_resp("hi")]))
        assert conv._max_tool_result_length == 16000


# ---------------------------------------------------------------------------
# clear() resets full tool results (L8)
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_full_tool_results(self):
        responses = [
            _tool_call_resp({"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}),
            _text_resp("Done."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=_echo_registry())
        conv.ask("Call echo")
        assert conv._full_tool_results

        conv.clear()
        assert conv.messages == []
        assert conv._full_tool_results == {}


# ---------------------------------------------------------------------------
# Budget enforcement in the event path (L2)
# ---------------------------------------------------------------------------


class TestBudgetInEventPath:
    def test_budget_enforced_in_ask_with_tool_events(self):
        responses = [
            _tool_call_resp({"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}, cost=0.02),
            _text_resp("should not get here"),
        ]
        conv = Conversation(
            driver=MockToolDriver(responses),
            tools=_echo_registry(),
            max_cost=0.01,
            budget_policy="hard_stop",
        )
        with pytest.raises(BudgetExceededError):
            for _event in conv.ask_with_tool_events("Call echo"):
                pass


# ---------------------------------------------------------------------------
# Cooperative stop mid-loop (C2)
# ---------------------------------------------------------------------------


class TestCooperativeStop:
    def test_request_stop_mid_loop(self):
        executions = 0

        def echo(text: str) -> str:
            """Echo text back."""
            nonlocal executions
            executions += 1
            conv.request_stop()
            return text

        # Round 1: tool call (tool requests the stop). Round 2 is the
        # graceful no-tools final answer.
        responses = [
            _tool_call_resp({"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}),
            _text_resp("Wrapping up without more tools."),
        ]
        conv = Conversation(driver=MockToolDriver(responses))
        conv.register_tool(echo)
        result = conv.ask("Call echo then keep going")

        assert result == "Wrapping up without more tools."
        assert executions == 1
        assert conv.max_rounds_reached is False

    async def test_request_stop_mid_loop_async(self):
        executions = 0

        async def echo(text: str) -> str:
            """Echo text back."""
            nonlocal executions
            executions += 1
            conv.request_stop()
            return text

        responses = [
            _tool_call_resp({"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}),
            _text_resp("Wrapping up without more tools."),
        ]
        conv = AsyncConversation(driver=MockAsyncToolDriver(responses))
        conv.register_tool(echo)
        result = await conv.ask("Call echo then keep going")

        assert result == "Wrapping up without more tools."
        assert executions == 1


# ---------------------------------------------------------------------------
# Per-tool timeout
# ---------------------------------------------------------------------------


class TestToolTimeout:
    def test_sync_tool_timeout_feeds_error(self):
        def slow() -> str:
            """Sleep too long."""
            time.sleep(5)
            return "never"

        reg = ToolRegistry()
        reg.register(slow)

        responses = [
            _tool_call_resp({"id": "call_1", "name": "slow", "arguments": {}}),
            _text_resp("Tool was too slow."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=reg, tool_timeout=0.1)
        result = conv.ask("Call slow")

        assert result == "Tool was too slow."
        assert "timed out after 0.1s" in _tool_messages(conv)[0]["content"]

    async def test_async_tool_timeout_feeds_error(self):
        async def slow() -> str:
            """Sleep too long."""
            await asyncio.sleep(5)
            return "never"

        reg = ToolRegistry()
        reg.register(slow)

        responses = [
            _tool_call_resp({"id": "call_1", "name": "slow", "arguments": {}}),
            _text_resp("Tool was too slow."),
        ]
        conv = AsyncConversation(driver=MockAsyncToolDriver(responses), tools=reg, tool_timeout=0.1)
        result = await conv.ask("Call slow")

        assert result == "Tool was too slow."
        assert "timed out after 0.1s" in _tool_messages(conv)[0]["content"]


# ---------------------------------------------------------------------------
# Graceful max-rounds final answer (C4)
# ---------------------------------------------------------------------------


class TestGracefulMaxRounds:
    def test_native_loop_graceful_final_answer(self):
        responses = [
            _tool_call_resp({"id": f"call_{i}", "name": "echo", "arguments": {"text": "hi"}}) for i in range(2)
        ]
        responses.append(_text_resp("Final answer from gathered results."))
        conv = Conversation(driver=MockToolDriver(responses), tools=_echo_registry(), max_tool_rounds=2)
        result = conv.ask("Loop forever")

        assert result == "Final answer from gathered results."
        assert conv.max_rounds_reached is True

    async def test_async_native_loop_graceful_final_answer(self):
        responses = [
            _tool_call_resp({"id": f"call_{i}", "name": "echo", "arguments": {"text": "hi"}}) for i in range(2)
        ]
        responses.append(_text_resp("Final answer from gathered results."))
        conv = AsyncConversation(driver=MockAsyncToolDriver(responses), tools=_echo_registry(), max_tool_rounds=2)
        result = await conv.ask("Loop forever")

        assert result == "Final answer from gathered results."
        assert conv.max_rounds_reached is True

    def test_event_path_graceful_final_answer(self):
        responses = [
            _tool_call_resp({"id": "call_0", "name": "echo", "arguments": {"text": "hi"}}),
            _text_resp("Final answer via events."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=_echo_registry(), max_tool_rounds=1)
        events = list(conv.ask_with_tool_events("Loop forever"))

        assert conv.max_rounds_reached is True
        assert events[-1] == {"type": "text_delta", "text": "Final answer via events."}


# ---------------------------------------------------------------------------
# Async parallel execution preserves call order
# ---------------------------------------------------------------------------


class TestAsyncParallelTools:
    async def test_gather_preserves_result_order(self):
        finished: list[str] = []

        async def slow_tool() -> str:
            """Slow tool."""
            await asyncio.sleep(0.2)
            finished.append("slow_tool")
            return "slow-result"

        async def fast_tool() -> str:
            """Fast tool."""
            finished.append("fast_tool")
            return "fast-result"

        reg = ToolRegistry()
        reg.register(slow_tool)
        reg.register(fast_tool)

        responses = [
            _tool_call_resp(
                {"id": "call_slow", "name": "slow_tool", "arguments": {}},
                {"id": "call_fast", "name": "fast_tool", "arguments": {}},
            ),
            _text_resp("Both done."),
        ]
        conv = AsyncConversation(driver=MockAsyncToolDriver(responses), tools=reg)
        result = await conv.ask("Call both")

        assert result == "Both done."
        # Completion order differs from call order (parallel execution)
        assert finished == ["fast_tool", "slow_tool"]
        # Results are recorded in call order so tool_call_id matching holds
        tool_msgs = _tool_messages(conv)
        assert [m["tool_call_id"] for m in tool_msgs] == ["call_slow", "call_fast"]
        assert [m["content"] for m in tool_msgs] == ["slow-result", "fast-result"]
        assert conv._full_tool_results == {"call_slow": "slow-result", "call_fast": "fast-result"}

    async def test_sequential_tools_opt_out(self):
        started: list[str] = []
        finished: list[str] = []

        async def slow_tool() -> str:
            """Slow tool."""
            started.append("slow_tool")
            await asyncio.sleep(0.1)
            finished.append("slow_tool")
            return "slow-result"

        async def fast_tool() -> str:
            """Fast tool."""
            started.append("fast_tool")
            finished.append("fast_tool")
            return "fast-result"

        reg = ToolRegistry()
        reg.register(slow_tool)
        reg.register(fast_tool)

        responses = [
            _tool_call_resp(
                {"id": "call_slow", "name": "slow_tool", "arguments": {}},
                {"id": "call_fast", "name": "fast_tool", "arguments": {}},
            ),
            _text_resp("Both done."),
        ]
        conv = AsyncConversation(driver=MockAsyncToolDriver(responses), tools=reg, sequential_tools=True)
        result = await conv.ask("Call both")

        assert result == "Both done."
        # Fully sequential: slow finishes before fast starts
        assert started == ["slow_tool", "fast_tool"]
        assert finished == ["slow_tool", "fast_tool"]
        tool_msgs = _tool_messages(conv)
        assert [m["tool_call_id"] for m in tool_msgs] == ["call_slow", "call_fast"]


# ---------------------------------------------------------------------------
# History trimming never orphans tool pairs (M11)
# ---------------------------------------------------------------------------


class TestHistoryTrimming:
    def test_trim_never_leaves_leading_tool_message(self):
        conv = Conversation(driver=MockToolDriver([_text_resp("hi")]), max_history_messages=3)
        conv._messages = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        conv._accumulate_usage({})

        # Naive tail-slice would start with the orphaned `tool` message;
        # boundary-aware trimming drops it.
        assert len(conv.messages) == 2
        assert conv.messages[0]["role"] == "user"
        assert all(m.get("role") != "tool" for m in conv.messages)


# ---------------------------------------------------------------------------
# Missing tool-call ids are generated before execution (L4)
# ---------------------------------------------------------------------------


class TestMissingToolCallId:
    def test_empty_id_gets_generated(self):
        responses = [
            _tool_call_resp({"id": "", "name": "echo", "arguments": {"text": "hi"}}),
            _text_resp("Done."),
        ]
        conv = Conversation(driver=MockToolDriver(responses), tools=_echo_registry())
        result = conv.ask("Call echo")

        assert result == "Done."
        tool_msg = _tool_messages(conv)[0]
        assert tool_msg["tool_call_id"].startswith("call_")
        assert len(tool_msg["tool_call_id"]) == len("call_") + 32
        # Assistant tool_calls entry matches the tool result id
        assistant_msg = next(m for m in conv.messages if m.get("tool_calls"))
        assert assistant_msg["tool_calls"][0]["id"] == tool_msg["tool_call_id"]
        assert tool_msg["tool_call_id"] in conv._full_tool_results
