"""Tests for coding-agent structured event parsing."""

from __future__ import annotations

import json

from prompture.infra.coding_agent_events import (
    CodingAgentEvent,
    parse_claude_stream_json_lines,
    parse_codex_json_lines,
)


def _lines(*objs: dict) -> list[str]:
    return [json.dumps(o) for o in objs]


class TestClaudeStreamJSONParser:
    def test_emits_system_init_event(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines({"type": "system", "subtype": "init", "session_id": "abc"})
            )
        )
        assert len(events) == 1
        assert events[0].type == "system"
        assert events[0].raw == {"type": "system", "subtype": "init", "session_id": "abc"}

    def test_emits_assistant_text_message(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "assistant",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Hello world"}],
                        },
                    }
                )
            )
        )
        assert events == [
            CodingAgentEvent(type="message", text="Hello world", raw=events[0].raw),
        ]

    def test_emits_tool_call_with_input(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "id": "t1",
                                    "name": "Bash",
                                    "input": {"command": "ls"},
                                }
                            ]
                        },
                    }
                )
            )
        )
        assert len(events) == 1
        assert events[0].type == "tool_call"
        assert events[0].tool_name == "Bash"
        assert events[0].tool_input == {"command": "ls"}

    def test_emits_tool_result_flattened_text(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t1",
                                    "content": [
                                        {"type": "text", "text": "file1"},
                                        {"type": "text", "text": "file2"},
                                    ],
                                }
                            ]
                        },
                    }
                )
            )
        )
        assert len(events) == 1
        assert events[0].type == "tool_result"
        assert events[0].tool_output == "file1\nfile2"

    def test_tool_result_with_string_content(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "t1",
                                    "content": "raw stdout",
                                }
                            ]
                        },
                    }
                )
            )
        )
        assert events[0].tool_output == "raw stdout"

    def test_emits_done_with_cost_and_tokens(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "duration_ms": 12345,
                        "total_cost_usd": 0.0341,
                        "result": "Done",
                        "usage": {"input_tokens": 100, "output_tokens": 50},
                    }
                )
            )
        )
        assert events == [
            CodingAgentEvent(
                type="done",
                text="Done",
                cost_usd=0.0341,
                input_tokens=100,
                output_tokens=50,
                duration_ms=12345,
                raw=events[0].raw,
            )
        ]

    def test_done_with_error_flag_records_error_text(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "result",
                        "is_error": True,
                        "result": "rate limited",
                    }
                )
            )
        )
        assert events[0].type == "done"
        assert events[0].error == "rate limited"

    def test_skips_blank_lines_and_yields_error_on_bad_json(self):
        events = list(
            parse_claude_stream_json_lines(
                [
                    "",
                    "   ",
                    "not-json{",
                    json.dumps({"type": "system"}),
                ]
            )
        )
        assert [e.type for e in events] == ["error", "system"]
        assert "invalid JSON" in (events[0].error or "")

    def test_multi_part_assistant_content_emits_multiple_events(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "text", "text": "Let me check."},
                                {"type": "tool_use", "name": "Read", "input": {"path": "x"}},
                            ]
                        },
                    }
                )
            )
        )
        assert [e.type for e in events] == ["message", "tool_call"]
        assert events[0].text == "Let me check."
        assert events[1].tool_name == "Read"

    def test_unknown_event_type_is_silently_dropped(self):
        events = list(parse_claude_stream_json_lines(_lines({"type": "heartbeat"})))
        assert events == []

    def test_full_session_replay(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {"type": "system", "subtype": "init", "session_id": "s1"},
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "Reading..."}]},
                    },
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "tool_use", "name": "Read", "input": {"path": "a.py"}}
                            ]
                        },
                    },
                    {
                        "type": "user",
                        "message": {
                            "content": [
                                {"type": "tool_result", "tool_use_id": "1", "content": "print('hi')"}
                            ]
                        },
                    },
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "Done",
                        "total_cost_usd": 0.001,
                        "usage": {"input_tokens": 10, "output_tokens": 5},
                    },
                )
            )
        )
        assert [e.type for e in events] == [
            "system",
            "message",
            "tool_call",
            "tool_result",
            "done",
        ]
        done = events[-1]
        assert done.cost_usd == 0.001
        assert done.input_tokens == 10
        assert done.output_tokens == 5


class TestCodexJSONParser:
    def test_task_started_is_system_event(self):
        events = list(parse_codex_json_lines(_lines({"id": "1", "msg": {"type": "task_started", "model": "gpt-5"}})))
        assert [e.type for e in events] == ["system"]

    def test_agent_message_emits_message_event(self):
        events = list(
            parse_codex_json_lines(_lines({"msg": {"type": "agent_message", "message": "hello"}}))
        )
        assert events == [CodingAgentEvent(type="message", text="hello", raw=events[0].raw)]

    def test_agent_message_delta_emits_partial_message(self):
        events = list(
            parse_codex_json_lines(_lines({"msg": {"type": "agent_message_delta", "delta": "Hel"}}))
        )
        assert [e.type for e in events] == ["message"]
        assert events[0].text == "Hel"

    def test_empty_delta_is_skipped(self):
        events = list(
            parse_codex_json_lines(_lines({"msg": {"type": "agent_message_delta", "delta": ""}}))
        )
        assert events == []

    def test_exec_command_begin_is_tool_call(self):
        events = list(
            parse_codex_json_lines(
                _lines({"msg": {"type": "exec_command_begin", "call_id": "c1", "command": ["ls", "-la"]}})
            )
        )
        assert events[0].type == "tool_call"
        assert events[0].tool_name == "exec"
        assert events[0].tool_input == {"command": ["ls", "-la"]}

    def test_exec_command_end_merges_stdout_stderr(self):
        events = list(
            parse_codex_json_lines(
                _lines(
                    {
                        "msg": {
                            "type": "exec_command_end",
                            "call_id": "c1",
                            "exit_code": 0,
                            "stdout": "file1",
                            "stderr": "warn",
                        }
                    }
                )
            )
        )
        assert events[0].type == "tool_result"
        assert events[0].tool_output == "file1\nwarn"

    def test_task_complete_emits_done_with_tokens(self):
        events = list(
            parse_codex_json_lines(
                _lines(
                    {
                        "msg": {
                            "type": "task_complete",
                            "last_agent_message": "All done.",
                            "token_usage": {"input_tokens": 100, "output_tokens": 200},
                        }
                    }
                )
            )
        )
        assert events[0].type == "done"
        assert events[0].text == "All done."
        assert events[0].input_tokens == 100
        assert events[0].output_tokens == 200

    def test_explicit_error_event(self):
        events = list(
            parse_codex_json_lines(_lines({"msg": {"type": "error", "message": "rate limit"}}))
        )
        assert events[0].type == "error"
        assert events[0].error == "rate limit"

    def test_unknown_type_is_dropped(self):
        events = list(parse_codex_json_lines(_lines({"msg": {"type": "heartbeat"}})))
        assert events == []

    def test_envelope_without_msg_is_dropped(self):
        events = list(parse_codex_json_lines(_lines({"id": "1"})))
        assert events == []
