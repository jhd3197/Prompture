"""Tests for :mod:`prompture.drivers._prompted_tool_stream`.

These exercise the Tier-2 streaming-tool emulator that turns plain text
output from a non-tool-calling model into the same interleaved
:mod:`~prompture.agents.live_events` stream that native drivers produce.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from prompture.agents.live_events import (
    LiveEvent,
    MessageStop,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)
from prompture.agents.tool_grammars import (
    GRAMMARS,
    XML_TAGS_GRAMMAR,
    get_grammar,
    inject_tool_instructions,
)
from prompture.drivers._prompted_tool_stream import (
    PromptedToolStreamMixin,
    PromptedToolStreamParser,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_parser(chunks: list[str]) -> list[LiveEvent]:
    """Feed ``chunks`` into a fresh parser and return all events."""
    parser = PromptedToolStreamParser(XML_TAGS_GRAMMAR)
    out: list[LiveEvent] = []
    for c in chunks:
        out.extend(parser.feed(c))
    out.extend(parser.finish())
    return out


def _text_of(events: list[LiveEvent]) -> str:
    """Concatenate all ``TextDelta`` payloads (helper for assertions)."""
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def _args_text_of(events: list[LiveEvent]) -> str:
    """Concatenate all ``ToolInputDelta`` fragments."""
    return "".join(e.fragment for e in events if isinstance(e, ToolInputDelta))


# ---------------------------------------------------------------------------
# Grammar registry
# ---------------------------------------------------------------------------


class TestGrammarRegistry:
    def test_xml_tags_grammar_is_registered(self):
        assert "xml_tags" in GRAMMARS
        assert get_grammar("xml_tags") is XML_TAGS_GRAMMAR

    def test_unknown_grammar_raises(self):
        with pytest.raises(KeyError, match="Unknown tool grammar"):
            get_grammar("nope")

    def test_inject_tool_instructions_appends_to_existing_system(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        tools = [{"function": {"name": "search", "description": "Search", "parameters": {}}}]
        out = inject_tool_instructions(messages, tools, XML_TAGS_GRAMMAR)

        assert out[0]["role"] == "system"
        assert out[0]["content"].startswith("You are helpful.")
        assert "<tool_call" in out[0]["content"]
        assert "search" in out[0]["content"]
        # Original list not mutated
        assert messages[0]["content"] == "You are helpful."

    def test_inject_tool_instructions_inserts_system_if_missing(self):
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"function": {"name": "search", "description": "x", "parameters": {}}}]
        out = inject_tool_instructions(messages, tools, XML_TAGS_GRAMMAR)

        assert out[0]["role"] == "system"
        assert "<tool_call" in out[0]["content"]
        assert out[1]["role"] == "user"

    def test_inject_tool_instructions_empty_tools_is_passthrough(self):
        messages = [{"role": "user", "content": "hi"}]
        out = inject_tool_instructions(messages, [], XML_TAGS_GRAMMAR)
        assert out == messages
        # Returns a copy though
        assert out is not messages


# ---------------------------------------------------------------------------
# Parser — pure narration paths
# ---------------------------------------------------------------------------


class TestParserNarration:
    def test_plain_text_no_tools(self):
        events = _run_parser(["Hello, world!"])
        assert _text_of(events) == "Hello, world!"
        assert not any(isinstance(e, (ToolUseStart, ToolUseStop)) for e in events)

    def test_text_split_across_many_chunks(self):
        events = _run_parser(["Hel", "lo", ", ", "world", "!"])
        assert _text_of(events) == "Hello, world!"

    def test_empty_chunks_are_noop(self):
        events = _run_parser(["hi", "", "", " there"])
        assert _text_of(events) == "hi there"

    def test_empty_input_emits_nothing(self):
        events = _run_parser([])
        assert events == []

    def test_text_containing_lt_but_not_tool_call_is_emitted(self):
        events = _run_parser(["a < b and c > d"])
        assert _text_of(events) == "a < b and c > d"


# ---------------------------------------------------------------------------
# Parser — single tool call
# ---------------------------------------------------------------------------


class TestParserSingleToolCall:
    def test_tool_call_only(self):
        events = _run_parser(['<tool_call name="search">{"q": "cats"}</tool_call>'])
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        stops = [e for e in events if isinstance(e, ToolUseStop)]
        assert len(starts) == 1
        assert starts[0].name == "search"
        assert starts[0].id.startswith("call_")
        assert len(stops) == 1
        assert stops[0].input == {"q": "cats"}
        # Same id flows through start → stop
        assert starts[0].id == stops[0].id

    def test_narration_before_and_after_tool_call(self):
        events = _run_parser(['Let me search. <tool_call name="search">{"q": "cats"}</tool_call> Done.'])
        text_blocks = [e.text for e in events if isinstance(e, TextDelta)]
        assert "Let me search. " in text_blocks
        assert any(" Done." in t for t in text_blocks)
        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.input == {"q": "cats"}

    def test_tool_call_with_explicit_id(self):
        events = _run_parser(['<tool_call name="search" id="call_xyz">{"q":"a"}</tool_call>'])
        start = next(e for e in events if isinstance(e, ToolUseStart))
        assert start.id == "call_xyz"
        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.id == "call_xyz"

    def test_args_streamed_as_fragments(self):
        # Force the args to arrive in many small pieces.
        events = _run_parser(
            [
                '<tool_call name="search">',
                '{"q"',
                ': "ca',
                "ts",
                '"}',
                "</tool_call>",
            ]
        )
        assert _args_text_of(events) == '{"q": "cats"}'
        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.input == {"q": "cats"}

    def test_open_tag_split_across_chunks(self):
        events = _run_parser(
            [
                "Before ",
                "<tool_",
                'call name="x">',
                '{"a":1}',
                "</tool_call>",
                " after",
            ]
        )
        assert "Before " in _text_of(events)
        assert " after" in _text_of(events)
        # The "<tool_" prefix should NOT have leaked into text output.
        assert "<tool_" not in _text_of(events)
        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.name == "x"
        assert stop.input == {"a": 1}

    def test_close_marker_split_across_chunks(self):
        events = _run_parser(
            [
                '<tool_call name="x">{"a":1}</tool_',
                "call>",
                " post",
            ]
        )
        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.input == {"a": 1}
        # The "</tool_" prefix should NOT have leaked into args output.
        assert "</tool_" not in _args_text_of(events)
        assert " post" in _text_of(events)


# ---------------------------------------------------------------------------
# Parser — multi-tool & edge cases
# ---------------------------------------------------------------------------


class TestParserMultiTool:
    def test_two_tool_calls_in_one_turn(self):
        events = _run_parser(
            ['A <tool_call name="t1">{"x":1}</tool_call> B <tool_call name="t2">{"y":2}</tool_call> C']
        )
        starts = [e for e in events if isinstance(e, ToolUseStart)]
        stops = [e for e in events if isinstance(e, ToolUseStop)]
        assert [s.name for s in starts] == ["t1", "t2"]
        assert [s.input for s in stops] == [{"x": 1}, {"y": 2}]
        # All three text pieces present
        text = _text_of(events)
        for piece in ("A ", " B ", " C"):
            assert piece in text

    def test_two_tool_calls_each_in_their_own_chunk(self):
        events = _run_parser(
            [
                '<tool_call name="t1">{"x":1}</tool_call>',
                '<tool_call name="t2">{"y":2}</tool_call>',
            ]
        )
        stops = [e for e in events if isinstance(e, ToolUseStop)]
        assert len(stops) == 2
        assert stops[0].input == {"x": 1}
        assert stops[1].input == {"y": 2}


class TestParserErrorRecovery:
    def test_malformed_json_args_yields_empty_dict(self, caplog):
        events = _run_parser(['<tool_call name="bad">{this is not json}</tool_call>'])
        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.input == {}
        assert "Failed to parse tool args JSON" in caplog.text

    def test_args_parsed_as_non_dict_yields_empty_dict(self, caplog):
        events = _run_parser(['<tool_call name="bad">[1, 2, 3]</tool_call>'])
        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.input == {}
        assert "not dict" in caplog.text

    def test_unclosed_tool_call_at_end_emits_stop_with_warning(self, caplog):
        events = _run_parser(['<tool_call name="search">{"q": "cats"'])
        start = next(e for e in events if isinstance(e, ToolUseStart))
        stops = [e for e in events if isinstance(e, ToolUseStop)]
        assert start.name == "search"
        assert len(stops) == 1
        # Args won't be a complete JSON object, so input will be {} after
        # graceful parse failure.
        assert stops[0].input == {}
        assert "was not closed" in caplog.text

    def test_empty_args_block(self):
        events = _run_parser(['<tool_call name="n">{}</tool_call>'])
        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.input == {}


# ---------------------------------------------------------------------------
# Mixin integration with a fake driver
# ---------------------------------------------------------------------------


class _FakeStreamingDriver(PromptedToolStreamMixin):
    """Minimal stand-in driver that yields a scripted token stream."""

    prompted_tool_grammar = "xml_tags"

    def __init__(self, chunks: list[str], meta: dict[str, Any] | None = None) -> None:
        self._chunks = chunks
        self._meta = meta or {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.0,
            "model_name": "fake",
        }
        self.captured_messages: list[dict[str, Any]] | None = None

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        self.captured_messages = list(messages)
        full = ""
        for c in self._chunks:
            full += c
            yield {"type": "delta", "text": c}
        yield {"type": "done", "text": full, "meta": self._meta}


class TestMixinEmulation:
    def test_emulation_injects_tool_instructions_into_system(self):
        driver = _FakeStreamingDriver(chunks=["just narration"])
        tools = [{"function": {"name": "x", "description": "x", "parameters": {}}}]
        messages = [{"role": "user", "content": "hi"}]

        list(driver._stream_via_prompted_emulation(messages, tools, {}))

        assert driver.captured_messages is not None
        assert driver.captured_messages[0]["role"] == "system"
        assert "<tool_call" in driver.captured_messages[0]["content"]

    def test_emulation_yields_text_then_message_stop(self):
        driver = _FakeStreamingDriver(chunks=["hello", " world"])
        events = list(
            driver._stream_via_prompted_emulation(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"function": {"name": "x", "description": "x", "parameters": {}}}],
                options={},
            )
        )
        assert _text_of(events) == "hello world"
        last = events[-1]
        assert isinstance(last, MessageStop)
        assert last.usage["total_tokens"] == 15

    def test_emulation_detects_tool_call_in_stream(self):
        driver = _FakeStreamingDriver(
            chunks=[
                "Thinking. ",
                '<tool_call name="search">{"q":"cats"}</tool_call>',
                " done.",
            ],
        )
        events = list(
            driver._stream_via_prompted_emulation(
                messages=[{"role": "user", "content": "hi"}],
                tools=[{"function": {"name": "search", "description": "x", "parameters": {}}}],
                options={},
            )
        )

        # We should see text, then tool, then text, then message stop.
        kinds = [type(e).__name__ for e in events]
        assert "ToolUseStart" in kinds
        assert "ToolUseStop" in kinds
        assert kinds[-1] == "MessageStop"

        stop = next(e for e in events if isinstance(e, ToolUseStop))
        assert stop.input == {"q": "cats"}
