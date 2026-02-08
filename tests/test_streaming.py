"""Tests for streaming support."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from prompture.infra.callbacks import DriverCallbacks
from prompture.agents.conversation import Conversation
from prompture.drivers.base import Driver
from prompture.infra.session import UsageSession

# ---------------------------------------------------------------------------
# Mock streaming driver
# ---------------------------------------------------------------------------


class MockStreamDriver(Driver):
    supports_messages = True
    supports_streaming = True

    def __init__(self, chunks: list[dict[str, Any]]):
        self._chunks = chunks

    def generate_messages(self, messages, options):
        # Non-streaming fallback
        return {
            "text": "fallback response",
            "meta": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "cost": 0.0},
        }

    def generate_messages_stream(self, messages, options) -> Iterator[dict[str, Any]]:
        for chunk in self._chunks:
            yield chunk


class NonStreamDriver(Driver):
    supports_messages = True
    supports_streaming = False

    def generate_messages(self, messages, options):
        return {
            "text": "non-stream answer",
            "meta": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "cost": 0.0},
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_ask_stream_yields_chunks(self):
        chunks = [
            {"type": "delta", "text": "Hello"},
            {"type": "delta", "text": " world"},
            {"type": "done", "text": "Hello world", "meta": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001,
            }},
        ]

        driver = MockStreamDriver(chunks)
        conv = Conversation(driver=driver)

        collected = list(conv.ask_stream("Hi"))
        assert collected == ["Hello", " world"]

    def test_ask_stream_records_history(self):
        chunks = [
            {"type": "delta", "text": "Test"},
            {"type": "done", "text": "Test", "meta": {
                "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost": 0.0,
            }},
        ]

        driver = MockStreamDriver(chunks)
        conv = Conversation(driver=driver)

        # Consume the generator
        list(conv.ask_stream("Question"))

        msgs = conv.messages
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Question"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Test"

    def test_ask_stream_accumulates_usage(self):
        chunks = [
            {"type": "delta", "text": "X"},
            {"type": "done", "text": "X", "meta": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.002,
            }},
        ]

        driver = MockStreamDriver(chunks)
        conv = Conversation(driver=driver)
        list(conv.ask_stream("test"))

        usage = conv.usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15
        assert usage["cost"] == 0.002

    def test_ask_stream_fallback_no_streaming_support(self):
        """If driver doesn't support streaming, fall back to regular ask."""
        driver = NonStreamDriver()
        conv = Conversation(driver=driver)

        collected = list(conv.ask_stream("Hi"))
        assert collected == ["non-stream answer"]

    def test_on_stream_delta_callback(self):
        chunks = [
            {"type": "delta", "text": "A"},
            {"type": "delta", "text": "B"},
            {"type": "done", "text": "AB", "meta": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0,
            }},
        ]

        deltas = []
        callbacks = DriverCallbacks(on_stream_delta=lambda info: deltas.append(info["text"]))

        driver = MockStreamDriver(chunks)
        conv = Conversation(driver=driver, callbacks=callbacks)
        list(conv.ask_stream("test"))

        assert deltas == ["A", "B"]

    def test_driver_base_streaming_not_implemented(self):
        """Base Driver raises NotImplementedError for streaming."""
        driver = Driver()
        with pytest.raises(NotImplementedError):
            list(driver.generate_messages_stream([], {}))


class TestStreamingMultipleTurns:
    def test_stream_then_regular(self):
        """Can alternate between streaming and regular asks."""
        chunks = [
            {"type": "delta", "text": "streamed"},
            {"type": "done", "text": "streamed", "meta": {
                "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "cost": 0.0,
            }},
        ]

        class HybridDriver(Driver):
            supports_messages = True
            supports_streaming = True

            def generate_messages(self, messages, options):
                return {
                    "text": "regular response",
                    "meta": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "cost": 0.0},
                }

            def generate_messages_stream(self, messages, options):
                for c in chunks:
                    yield c

        conv = Conversation(driver=HybridDriver())

        # Stream first
        list(conv.ask_stream("stream this"))
        # Then regular
        result = conv.ask("regular question")

        assert result == "regular response"
        assert len(conv.messages) == 4  # 2 from stream + 2 from regular


class TestStreamingCallbacks:
    """Tests that on_request / on_response / on_error callbacks fire during streaming."""

    def _make_chunks(self) -> list[dict[str, Any]]:
        return [
            {"type": "delta", "text": "Hello"},
            {"type": "delta", "text": " world"},
            {"type": "done", "text": "Hello world", "meta": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001,
            }},
        ]

    def test_ask_stream_fires_on_response_callback(self):
        """on_response fires once with correct meta and elapsed_ms."""
        response_log: list[dict[str, Any]] = []
        callbacks = DriverCallbacks(on_response=lambda info: response_log.append(info))

        driver = MockStreamDriver(self._make_chunks())
        conv = Conversation(driver=driver, callbacks=callbacks)
        list(conv.ask_stream("Hi"))

        assert len(response_log) == 1
        info = response_log[0]
        assert info["meta"]["total_tokens"] == 15
        assert info["meta"]["cost"] == 0.001
        assert "elapsed_ms" in info
        assert info["elapsed_ms"] >= 0

    def test_ask_stream_fires_on_request_callback(self):
        """on_request fires once before streaming begins."""
        request_log: list[dict[str, Any]] = []
        callbacks = DriverCallbacks(on_request=lambda info: request_log.append(info))

        driver = MockStreamDriver(self._make_chunks())
        conv = Conversation(driver=driver, callbacks=callbacks)
        list(conv.ask_stream("Hi"))

        assert len(request_log) == 1
        info = request_log[0]
        assert info["prompt"] is None
        assert isinstance(info["messages"], list)
        assert "driver" in info

    def test_ask_stream_session_records_usage(self):
        """UsageSession wired as on_response records tokens/cost/call_count."""
        session = UsageSession()
        callbacks = DriverCallbacks(
            on_response=session.record,
            on_error=session.record_error,
        )

        driver = MockStreamDriver(self._make_chunks())
        conv = Conversation(driver=driver, callbacks=callbacks)
        list(conv.ask_stream("Hi"))

        assert session.total_tokens == 15
        assert session.prompt_tokens == 10
        assert session.completion_tokens == 5
        assert session.cost == 0.001
        assert session.call_count == 1

    def test_ask_stream_fires_on_error_callback(self):
        """on_error fires when the stream raises an exception."""

        class FailStreamDriver(Driver):
            supports_messages = True
            supports_streaming = True

            def generate_messages(self, messages, options):
                return {"text": "fallback", "meta": {}}

            def generate_messages_stream(self, messages, options) -> Iterator[dict[str, Any]]:
                yield {"type": "delta", "text": "partial"}
                raise RuntimeError("stream broke")

        error_log: list[dict[str, Any]] = []
        callbacks = DriverCallbacks(on_error=lambda info: error_log.append(info))

        driver = FailStreamDriver()
        conv = Conversation(driver=driver, callbacks=callbacks)

        with pytest.raises(RuntimeError, match="stream broke"):
            list(conv.ask_stream("Hi"))

        assert len(error_log) == 1
        assert isinstance(error_log[0]["error"], RuntimeError)
        assert "driver" in error_log[0]
