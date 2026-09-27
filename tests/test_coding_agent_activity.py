"""Live coding-agent turns: read from the end of Claude Code and Codex logs, published on the companion's bus."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prompture.companion import CodingToolSource, LiveBus
from prompture.infra.coding_agent_activity import AgentActivity, agents_in, claude_turn, codex_turn, tail_entries

NOW = datetime.now(timezone.utc).replace(microsecond=0)
SESSION = "11111111-2222-3333-4444-555555555555"


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _write(path: Path, entries: list[dict], *, age: float = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    if age:
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    return path


def _user(text: str | None = "fix the tests", *, tool_result: bool = False, secs: int = 0) -> dict:
    content = [{"type": "tool_result", "content": "ok"}] if tool_result else text
    return {
        "type": "user",
        "timestamp": _iso(NOW + timedelta(seconds=secs)),
        "cwd": "C:\\code\\alpha",
        "message": {"role": "user", "content": content},
    }


def _assistant(stop: str | None, *, secs: int = 1, sidechain: bool = False) -> dict:
    return {
        "type": "assistant",
        "timestamp": _iso(NOW + timedelta(seconds=secs)),
        "cwd": "C:\\code\\alpha",
        "isSidechain": sidechain,
        "message": {"model": "claude-opus-5-5", "stop_reason": stop, "content": [{"type": "text", "text": "x"}]},
    }


def _claude_file(root: Path, entries: list[dict], **kw) -> Path:
    return _write(root / "projects" / "C--code-alpha" / f"{SESSION}.jsonl", entries, **kw)


def _codex(kind: str, secs: int = 0, **payload) -> dict:
    if kind == "turn_context":
        return {"timestamp": _iso(NOW), "type": "turn_context", "payload": {"model": "gpt-5.5-codex", "cwd": "/w/beta"}}
    return {"timestamp": _iso(NOW + timedelta(seconds=secs)), "type": "event_msg", "payload": {"type": kind, **payload}}


def _codex_file(root: Path, entries: list[dict], **kw) -> Path:
    name = f"rollout-2026-09-26T03-31-27-{SESSION}.jsonl"
    return _write(root / "sessions" / "2026" / "09" / "26" / name, entries, **kw)


# ------------------------------------------------------------------ Claude Code


def test_claude_turn_runs_until_the_model_ends_it(tmp_path):
    path = _claude_file(tmp_path, [_user(), _assistant("tool_use"), _user(tool_result=True, secs=2)])
    turn = claude_turn(path)
    assert turn is not None
    assert (turn.agent, turn.session, turn.model, turn.project) == (
        "claude",
        SESSION,
        "claude/claude-opus-5-5",
        "alpha",
    )
    assert turn.since == NOW  # the prompt, not the tool result

    _claude_file(tmp_path, [_user(), _assistant("tool_use"), _user(tool_result=True), _assistant("end_turn", secs=3)])
    assert claude_turn(path) is None


def test_claude_turn_ends_on_turn_duration_and_interrupts(tmp_path):
    done = {"type": "system", "subtype": "turn_duration", "timestamp": _iso(NOW)}
    noise = {"type": "ai-title", "title": "x"}
    path = _claude_file(tmp_path, [_user(), _assistant(None), done, noise])
    assert claude_turn(path) is None

    _claude_file(tmp_path, [_user(), _assistant("tool_use"), _user("[Request interrupted by user for tool use]")])
    assert claude_turn(path) is None


def test_claude_subagent_entries_dont_end_the_main_turn(tmp_path):
    path = _claude_file(tmp_path, [_user(), _assistant("tool_use"), _assistant("end_turn", secs=5, sidechain=True)])
    assert claude_turn(path) is not None


def test_tail_entries_skips_a_cut_first_line_and_a_partial_last_one(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"n": 1}\n{"n": 2}\n{"n": 3}\n{"n": 4', encoding="utf-8")
    assert [e["n"] for e in tail_entries(path, max_bytes=20)] == [3]  # "2}" was cut, "4" is unfinished


# ------------------------------------------------------------------ Codex


def test_codex_turn_between_task_started_and_task_complete(tmp_path):
    path = _codex_file(tmp_path, [_codex("turn_context"), _codex("task_started", 1), _codex("token_count", 2)])
    turn = codex_turn(path)
    assert turn is not None
    assert (turn.agent, turn.session, turn.model, turn.project) == ("codex", SESSION, "openai/gpt-5.5-codex", "beta")

    _codex_file(tmp_path, [_codex("turn_context"), _codex("task_started", 1), _codex("task_complete", 9)])
    assert codex_turn(path) is None
    _codex_file(tmp_path, [_codex("turn_context"), _codex("task_started", 1), _codex("turn_aborted", 9)])
    assert codex_turn(path) is None


# ------------------------------------------------------------------ scanner


def test_scan_ignores_quiet_logs_and_agents_that_arent_running(tmp_path):
    claude, codex = tmp_path / "claude", tmp_path / "codex"
    _claude_file(claude, [_user(), _assistant("tool_use")])
    _codex_file(codex, [_codex("turn_context"), _codex("task_started")], age=3600)  # a terminal closed mid-turn

    assert [t.agent for t in AgentActivity(claude, codex, processes=None).scan()] == ["claude"]
    assert AgentActivity(claude, codex, processes=lambda: {"codex"}).scan() == []
    assert len(AgentActivity(claude, codex, processes=lambda: {"claude"}).scan()) == 1


def test_process_command_lines_name_the_agent():
    assert agents_in([["C:\\Users\\me\\.local\\bin\\claude.exe"], ["explorer.exe"]]) == {"claude"}
    assert agents_in([["node", "/usr/lib/node_modules/@openai/codex/bin/codex.js"]]) == {"codex"}
    assert agents_in([["node", "/srv/app.js"], ["vim", "claude-notes.md"]]) == set()


# ------------------------------------------------------------------ companion


def test_companion_publishes_turns_starting_changing_and_ending(tmp_path):
    claude, codex = tmp_path / "claude", tmp_path / "codex"
    path = _claude_file(claude, [_user(), _assistant("tool_use")])
    source = CodingToolSource(claude, codex, prefs_file=None, activity=AgentActivity(claude, codex, processes=None))
    bus = LiveBus()

    source.publish_turns(bus)
    source.publish_turns(bus)  # unchanged: nothing new
    (started,) = bus.replay(0)
    assert started["type"] == "request.started"
    assert started["request_id"] == f"agent:claude:{SESSION}"
    assert (started["key_name"], started["tool"], started["state"]) == ("Claude Code", "claude", "working")
    assert [r["request_id"] for r in bus.running()] == [started["request_id"]]

    _claude_file(claude, [_user(), _assistant("tool_use"), _assistant("end_turn", secs=4)])
    source.publish_turns(bus)
    ended = bus.replay(started["id"])
    assert [e["type"] for e in ended] == ["request.ended"]
    assert bus.running() == []
    assert path.exists()
