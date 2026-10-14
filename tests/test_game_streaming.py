"""Game calls consume complete streams, retaining validation, retries and billing."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from prompture.drivers.base import Driver
from prompture.extraction.core import ask_for_json
from prompture.games import Connect4, Connect4Board, GameHarness
from prompture.infra.callbacks import DriverCallbacks


def answer(move="4"):
    return json.dumps({"reason": "center", "plan": "build a line", "move": move})


class StreamingDriver(Driver):
    """Scripted streaming backend: no provider, sockets, or engines."""

    supports_streaming = True
    supports_json_mode = True
    supports_json_schema = True
    model = "scripted"

    def __init__(self, replies=None, terminal=True, final_text=True):
        self.replies = list(replies or [answer()])
        self.terminal = terminal
        self.final_text = final_text
        self.requests = []
        self.closed = 0
        self.generate = Mock(side_effect=AssertionError("non-streaming call"))
        self._auto_record_usage = Mock()
        self.callbacks = DriverCallbacks(on_request=Mock(), on_response=Mock(), on_error=Mock(), on_stream_delta=Mock())

    def generate(self, prompt, options):
        raise AssertionError("non-streaming call")

    def generate_messages_stream(self, messages, options):
        self.requests.append((messages, options))
        text = self.replies.pop(0)
        try:
            yield {"type": "thinking_delta", "text": "not JSON"}
            if isinstance(text, Exception):
                raise text
            yield {"type": "delta", "text": text[:10]}
            yield {"type": "delta", "text": text[10:]}
            if self.terminal:
                yield {
                    "type": "done",
                    "text": text if self.final_text else "",
                    "meta": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost": 0.01,
                    },
                }
        finally:
            self.closed += 1


class TestGameStreaming:
    """Transport choice must not weaken move checking or lose paid usage."""

    @pytest.mark.parametrize("final_text", [True, False])
    def test_stream_preserves_schema_system_options_usage_and_callbacks(self, final_text):
        driver = StreamingDriver(final_text=final_text)
        harness = GameHarness(Connect4(), driver=driver, system_prompt="Play carefully", options={"effort": "high"})
        decision = harness.choose(Connect4Board())
        assert (decision.name, decision.reason, decision.plan, decision.rescued) == (
            "4",
            "center",
            "build a line",
            False,
        )
        assert decision.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}
        messages, options = driver.requests[0]
        assert messages[0] == {"role": "system", "content": "Play carefully"}
        assert options["effort"] == "high"
        assert options["json_schema"]["properties"]["move"]["enum"] == [str(n) for n in range(1, 8)]
        assert "plan" in options["json_schema"]["required"]
        driver.callbacks.on_request.assert_called_once()
        driver.callbacks.on_response.assert_called_once()
        driver.callbacks.on_error.assert_not_called()
        driver._auto_record_usage.assert_called_once()
        driver.generate.assert_not_called()
        assert driver.callbacks.on_stream_delta.call_count == 2
        assert driver.closed == 1

    @pytest.mark.parametrize("first", ["invalid JSON", answer("8")])
    def test_streamed_replies_are_retried_with_feedback_and_all_usage(self, first):
        driver = StreamingDriver([first, answer()])
        decision = GameHarness(Connect4(), driver=driver).choose(Connect4Board())
        assert not decision.rescued and len(decision.attempts) == 2
        assert "Your previous answer was rejected" in driver.requests[1][0][0]["content"]
        assert decision.usage["total_tokens"] == 30
        assert decision.usage["cost"] == 0.02
        assert driver.closed == 2

    def test_missing_done_is_a_failed_call_even_if_json_is_complete(self):
        driver = StreamingDriver(terminal=False)
        decision = GameHarness(Connect4(), driver=driver, max_retries=0).choose(Connect4Board())
        assert decision.rescued
        assert decision.attempts[0].reply is None
        assert "without a completion summary" in decision.attempts[0].error
        driver.callbacks.on_error.assert_called_once()
        driver.callbacks.on_response.assert_not_called()
        assert driver._auto_record_usage.call_args.args[0]["meta"]["usage_complete"] is False
        assert driver.closed == 1

    def test_broken_stream_retries_and_releases_the_connection(self):
        driver = StreamingDriver([TimeoutError("read timed out"), answer()])
        decision = GameHarness(Connect4(), driver=driver).choose(Connect4Board())
        assert not decision.rescued
        assert "read timed out" in decision.attempts[0].error
        assert decision.usage["total_tokens"] == 15
        driver.callbacks.on_error.assert_called_once()
        assert driver.closed == 2

    @pytest.mark.parametrize("supported,stream", [(False, True), (True, False)])
    def test_unsupported_or_disabled_stream_uses_existing_calls(self, supported, stream):
        driver = StreamingDriver()
        driver.supports_streaming = supported
        driver.generate = Mock(return_value={"text": answer(), "meta": {"total_tokens": 3}})
        decision = GameHarness(Connect4(), driver=driver, stream=stream).choose(Connect4Board())
        assert not decision.rescued
        driver.generate.assert_called_once()
        assert not driver.requests

    def test_tool_only_driver_streams_prompted_json(self):
        driver = StreamingDriver()
        driver.supports_json_mode = driver.supports_json_schema = False
        driver.supports_tool_use = True
        decision = GameHarness(Connect4(), driver=driver).choose(Connect4Board())
        assert not decision.rescued
        assert "JSON schema:" in driver.requests[0][0][0]["content"]
        assert "json_schema" not in driver.requests[0][1]

    def test_explicit_tool_strategy_is_preserved(self):
        driver = StreamingDriver()
        driver.supports_tool_use = True
        driver.generate_messages_with_tools = Mock(
            return_value={
                "text": "",
                "meta": {"total_tokens": 3},
                "tool_calls": [{"name": "extract_data", "arguments": json.loads(answer())}],
            }
        )
        result = ask_for_json(driver, "Pick", {"type": "object"}, strategy="tool_call", stream=True, cache=False)
        assert result["json_object"]["move"] == "4"
        driver.generate_messages_with_tools.assert_called_once()
        assert not driver.requests
