"""Coding-agent usage readers, on small logs shaped like each agent's real ones."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from prompture.infra.coding_agent_readers import (
    ClaudeCodeReader,
    ClineReader,
    CodexReader,
    ContinueReader,
    GeminiCliReader,
    KimiCodeReader,
    OpenCodeReader,
    QwenCodeReader,
)
from prompture.infra.coding_agent_usage import CodingAgentUsage, parse_ts

NOW = datetime.now(timezone.utc).replace(microsecond=0)
T = NOW - timedelta(minutes=5)
SINCE = NOW - timedelta(days=1)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _jsonl(path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _usage(readers) -> CodingAgentUsage:
    return CodingAgentUsage(readers, scan_interval=0)


def test_parse_ts_handles_iso_seconds_and_milliseconds():
    assert parse_ts("2026-09-24T12:00:00Z") == datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
    assert parse_ts(1790000000) == parse_ts(1790000000000)
    assert parse_ts("nope") is None and parse_ts(None) is None


def test_claude_keeps_the_final_count_of_a_streamed_message(tmp_path):
    def line(out: int) -> dict:
        return {
            "type": "assistant",
            "timestamp": _iso(T),
            "requestId": "r1",
            "cwd": "/src/alpha",
            "message": {"id": "m1", "model": "claude-opus-5", "usage": {"input_tokens": 5, "output_tokens": out}},
        }

    _jsonl(tmp_path / "projects" / "p" / "s.jsonl", [line(3), line(90)])
    calls = _usage([ClaudeCodeReader(tmp_path, plan_usage=False)]).calls()
    assert len(calls) == 1 and calls[0].output_tokens == 90 and calls[0].project == "alpha"


def test_codex_prefers_per_call_records_over_running_counts(tmp_path):
    usage = {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 50, "reasoning_output_tokens": 20}
    _jsonl(
        tmp_path / "sessions" / "2026" / "09" / "24" / "rollout-x.jsonl",
        [
            {"timestamp": _iso(T), "type": "turn_context", "payload": {"model": "gpt-6", "cwd": "/src/beta"}},
            {
                "timestamp": _iso(T),
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {"last_token_usage": usage, "total_token_usage": {"total_tokens": 1050}},
                },
            },
            {"timestamp": _iso(T), "type": "token_usage_record", "payload": {"response_id": "resp_1", "usage": usage}},
        ],
    )
    calls = _usage([CodexReader(tmp_path)]).calls()
    assert [c.id for c in calls] == ["codex:resp_1"]
    c = calls[0]
    assert (c.model, c.input_tokens, c.cache_read_tokens, c.reasoning_tokens, c.project) == (
        "openai/gpt-6",
        1000,
        800,
        20,
        "beta",
    )


def test_kimi_counts_turn_records_only(tmp_path):
    wire = tmp_path / "sessions" / "wd_mapanare_0123456789ab" / "session_abc" / "agents" / "main" / "wire.jsonl"
    record = {
        "type": "usage.record",
        "model": "kimi-code/k3",
        "usage": {"inputOther": 100, "output": 40, "inputCacheRead": 900, "inputCacheCreation": 0},
        "time": _ms(T),
    }
    _jsonl(wire, [{**record, "usageScope": "turn"}, {**record, "usageScope": "session"}, {"type": "step.end"}])
    calls = _usage([KimiCodeReader(tmp_path)]).calls()
    assert len(calls) == 1
    c = calls[0]
    assert (c.model, c.input_tokens, c.cache_read_tokens, c.output_tokens, c.project, c.session) == (
        "moonshot/k3",
        1000,
        900,
        40,
        "mapanare",
        "abc",
    )


def test_gemini_reads_rewritten_chat_files_once(tmp_path):
    chat = tmp_path / "tmp" / "cachibot" / "chats" / "session-1.json"
    chat.parent.mkdir(parents=True)
    msg = {
        "id": "g1",
        "type": "gemini",
        "timestamp": _iso(T),
        "model": "gemini-2.5-pro",
        "content": "secret",
        "tokens": {"input": 500, "output": 30, "cached": 200, "thoughts": 10},
    }
    chat.write_text(json.dumps({"sessionId": "s", "messages": [{"type": "user", "content": "hi"}, msg]}))
    usage = _usage([GeminiCliReader(tmp_path)])
    assert len(usage.refresh(force=True)) == 1
    chat.write_text(json.dumps({"sessionId": "s", "messages": [msg, {**msg, "id": "g2"}]}))  # grows, rewritten whole
    assert [c.id for c in usage.refresh(force=True)] == ["gemini:g2"]
    c = usage.calls()[0]
    assert (c.output_tokens, c.reasoning_tokens, c.project) == (40, 10, "cachibot")


def test_qwen_counts_api_responses(tmp_path):
    event = {
        "event.name": "qwen-code.api_response",
        "event.timestamp": _iso(T),
        "response_id": "q1",
        "model": "coder-model",
        "status_code": 200,
        "input_token_count": 300,
        "output_token_count": 20,
        "cached_content_token_count": 100,
        "thoughts_token_count": 5,
        "response_text": "secret",
    }
    line = {
        "type": "system",
        "subtype": "ui_telemetry",
        "timestamp": _iso(T),
        "cwd": "/src/gamma",
        "sessionId": "s",
        "systemPayload": {"uiEvent": event},
    }
    _jsonl(tmp_path / "projects" / "gamma" / "chats" / "s.jsonl", [line, line])
    calls = _usage([QwenCodeReader(tmp_path)]).calls()
    assert len(calls) == 1 and calls[0].output_tokens == 25 and calls[0].project == "gamma"


def test_opencode_reads_messages_as_they_complete(tmp_path):
    db = tmp_path / "opencode.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE message (id TEXT, session_id TEXT, time_created INT, time_updated INT, data TEXT)")

    def data(tokens_in: int) -> str:
        return json.dumps(
            {
                "role": "assistant",
                "providerID": "zai",
                "modelID": "glm-5",
                "cost": 0.25,
                "path": {"root": "/src/delta"},
                "time": {"created": _ms(T), "completed": _ms(T)},
                "tokens": {"input": tokens_in, "output": 10, "reasoning": 0, "cache": {"read": 50, "write": 0}},
            }
        )

    con.execute("INSERT INTO message VALUES ('msg_1', 'ses_1', ?, ?, ?)", (_ms(T), _ms(T), data(0)))
    con.commit()
    reader = OpenCodeReader(tmp_path)
    usage = _usage([reader])
    assert [c.input_tokens for c in usage.refresh(force=True)] == [50]
    con.execute("UPDATE message SET time_updated = ?, data = ?", (_ms(T) + 5, data(400)))  # finished streaming
    con.commit()
    con.close()
    usage.refresh(force=True)
    c = usage.calls()[0]
    assert (c.input_tokens, c.cost_usd, c.cost_source, c.project, c.model) == (
        450,
        0.25,
        "reported",
        "delta",
        "zai/glm-5",
    )


def test_cline_uses_reported_cost(tmp_path):
    task = tmp_path / "tasks" / "1700000000000"
    task.mkdir(parents=True)
    info = {"request": "secret", "tokensIn": 100, "tokensOut": 20, "cacheReads": 5, "cacheWrites": 0, "cost": 0.01}
    (task / "ui_messages.json").write_text(
        json.dumps(
            [
                {"ts": _ms(T), "type": "say", "say": "api_req_started", "text": json.dumps(info)},
                {"ts": _ms(T), "type": "say", "say": "text", "text": "secret"},
            ]
        )
    )
    calls = _usage([ClineReader([tmp_path])]).calls()
    assert len(calls) == 1
    assert (calls[0].input_tokens, calls[0].cost_usd, calls[0].cost_source) == (105, 0.01, "reported")


def test_continue_lines(tmp_path):
    _jsonl(
        tmp_path / "dev_data" / "0.2.0" / "tokensGenerated.jsonl",
        [
            {
                "timestamp": _iso(T),
                "model": "llama3.1:8b",
                "provider": "ollama",
                "promptTokens": 40,
                "generatedTokens": 8,
            }
        ],
    )
    calls = _usage([ContinueReader(tmp_path)]).calls()
    assert [(c.model, c.tokens) for c in calls] == [("ollama/llama3.1:8b", 48)]


def test_summary_groups_by_agent_model_and_project(tmp_path):
    _jsonl(
        tmp_path / "k" / "sessions" / "wd_app_0123456789ab" / "session_1" / "agents" / "main" / "wire.jsonl",
        [
            {
                "type": "usage.record",
                "usageScope": "turn",
                "model": "kimi-code/k3",
                "usage": {"inputOther": 10, "output": 5},
                "time": _ms(T),
            }
        ],
    )
    _jsonl(
        tmp_path / "c" / "dev_data" / "0.2.0" / "tokensGenerated.jsonl",
        [{"timestamp": _iso(T), "model": "m", "provider": "ollama", "promptTokens": 1, "generatedTokens": 1}],
    )
    summary = _usage([KimiCodeReader(tmp_path / "k"), ContinueReader(tmp_path / "c")]).summary(SINCE)
    assert [a["agent"] for a in summary] == ["kimi", "continue"]
    kimi = summary[0]
    assert kimi["name"] == "Kimi Code" and kimi["requests"] == 1 and kimi["tokens"] == 15
    assert kimi["models"][0]["model"] == "moonshot/k3" and kimi["projects"][0]["project"] == "app"


def test_one_broken_reader_does_not_hide_the_others(tmp_path):
    class Broken(ContinueReader):
        agent = "broken"

        def read(self, since):
            raise RuntimeError("bad log")

    _jsonl(
        tmp_path / "dev_data" / "0.2.0" / "tokensGenerated.jsonl",
        [{"timestamp": _iso(T), "model": "m", "provider": "p", "promptTokens": 1, "generatedTokens": 1}],
    )
    assert len(_usage([Broken(tmp_path), ContinueReader(tmp_path)]).calls()) == 1


@pytest.mark.parametrize(
    "reader",
    [ClaudeCodeReader, CodexReader, KimiCodeReader, GeminiCliReader, QwenCodeReader, OpenCodeReader, ContinueReader],
)
def test_missing_folders_mean_not_available(tmp_path, reader):
    r = reader(tmp_path / "nothing")
    assert r.available() is False
    assert list(r.read(SINCE)) == []
