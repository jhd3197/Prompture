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
            parse_claude_stream_json_lines(_lines({"type": "system", "subtype": "init", "session_id": "abc"}))
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
                        "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"path": "a.py"}}]},
                    },
                    {
                        "type": "user",
                        "message": {"content": [{"type": "tool_result", "tool_use_id": "1", "content": "print('hi')"}]},
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
        events = list(parse_codex_json_lines(_lines({"msg": {"type": "agent_message", "message": "hello"}})))
        assert events == [CodingAgentEvent(type="message", text="hello", raw=events[0].raw)]

    def test_agent_message_delta_emits_partial_message(self):
        events = list(parse_codex_json_lines(_lines({"msg": {"type": "agent_message_delta", "delta": "Hel"}})))
        assert [e.type for e in events] == ["message"]
        assert events[0].text == "Hel"

    def test_empty_delta_is_skipped(self):
        events = list(parse_codex_json_lines(_lines({"msg": {"type": "agent_message_delta", "delta": ""}})))
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
        events = list(parse_codex_json_lines(_lines({"msg": {"type": "error", "message": "rate limit"}})))
        assert events[0].type == "error"
        assert events[0].error == "rate limit"

    def test_unknown_type_is_dropped(self):
        events = list(parse_codex_json_lines(_lines({"msg": {"type": "heartbeat"}})))
        assert events == []

    def test_envelope_without_msg_is_dropped(self):
        events = list(parse_codex_json_lines(_lines({"id": "1"})))
        assert events == []


class TestQuestionDetection:
    def test_simple_question_mark_triggers(self):
        from prompture.infra.coding_agent_events import detect_question

        is_q, choices = detect_question("Should we proceed with option A?")
        assert is_q is True
        assert choices is None

    def test_phrase_trigger_without_question_mark(self):
        from prompture.infra.coding_agent_events import detect_question

        is_q, _ = detect_question("Let me know which one you'd prefer.")
        assert is_q is True

    def test_numbered_choices_extracted(self):
        from prompture.infra.coding_agent_events import detect_question

        text = "Which approach do you want?\n1. Refactor the whole module\n2. Add a thin wrapper\n3. Leave it as-is"
        is_q, choices = detect_question(text)
        assert is_q is True
        assert choices == (
            "Refactor the whole module",
            "Add a thin wrapper",
            "Leave it as-is",
        )

    def test_bulleted_choices_extracted(self):
        from prompture.infra.coding_agent_events import detect_question

        text = "Do you want me to:\n- delete the file\n- archive it instead\n- skip it"
        is_q, choices = detect_question(text)
        assert is_q is True
        assert choices == ("delete the file", "archive it instead", "skip it")

    def test_no_question_when_just_a_statement(self):
        from prompture.infra.coding_agent_events import detect_question

        is_q, choices = detect_question("Updated the README and ran the tests.")
        assert is_q is False
        assert choices is None

    def test_single_bullet_is_not_choices(self):
        from prompture.infra.coding_agent_events import detect_question

        # One bullet alone doesn't count — needs at least two consecutive.
        is_q, choices = detect_question("Should I proceed?\n- yes")
        assert is_q is True
        assert choices is None

    def test_empty_text_returns_false(self):
        from prompture.infra.coding_agent_events import detect_question

        assert detect_question("") == (False, None)
        assert detect_question(None) == (False, None)

    def test_claude_message_with_question_emits_question_event(self):
        from prompture.infra.coding_agent_events import parse_claude_stream_json_lines

        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Which file would you like me to start with?",
                                }
                            ]
                        },
                    }
                )
            )
        )
        assert [e.type for e in events] == ["message", "question"]
        assert events[1].text == "Which file would you like me to start with?"

    def test_codex_message_with_choices_emits_question_with_choices(self):
        events = list(
            parse_codex_json_lines(
                _lines(
                    {
                        "msg": {
                            "type": "agent_message",
                            "message": ("Do you want me to:\n1. Update the schema\n2. Skip the migration"),
                        }
                    }
                )
            )
        )
        assert [e.type for e in events] == ["message", "question"]
        assert events[1].choices == ("Update the schema", "Skip the migration")

    def test_claude_done_text_question_emits_after_done(self):
        from prompture.infra.coding_agent_events import parse_claude_stream_json_lines

        events = list(
            parse_claude_stream_json_lines(
                _lines(
                    {
                        "type": "result",
                        "is_error": False,
                        "result": "Should I commit these changes?",
                        "total_cost_usd": 0.001,
                    }
                )
            )
        )
        assert [e.type for e in events] == ["done", "question"]


class TestSessionIdExtraction:
    def test_claude_system_event_extracts_session_id(self):
        events = list(
            parse_claude_stream_json_lines(_lines({"type": "system", "subtype": "init", "session_id": "abc-123"}))
        )
        assert events[0].session_id == "abc-123"

    def test_claude_result_event_extracts_session_id(self):
        events = list(
            parse_claude_stream_json_lines(
                _lines({"type": "result", "is_error": False, "result": "ok", "session_id": "abc-123"})
            )
        )
        assert events[0].type == "done"
        assert events[0].session_id == "abc-123"

    def test_codex_task_started_extracts_session_id_from_msg(self):
        events = list(
            parse_codex_json_lines(_lines({"msg": {"type": "task_started", "model": "gpt-5", "session_id": "codex-1"}}))
        )
        assert events[0].session_id == "codex-1"

    def test_codex_session_configured_extracts_envelope_session_id(self):
        events = list(parse_codex_json_lines(_lines({"session_id": "codex-2", "msg": {"type": "session_configured"}})))
        assert events[0].session_id == "codex-2"
