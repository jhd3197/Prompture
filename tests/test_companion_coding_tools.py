"""The companion's coding-tool source: Claude Code and Codex usage read from their session logs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from prompture.companion import CodingToolSource, LiveBus, summarize_spend
from prompture.infra import coding_agent_readers


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


NOW = datetime.now(timezone.utc).replace(microsecond=0)
EARLIER = NOW - timedelta(seconds=10)  # stays inside today's window


def _claude(msg_id: str, *, ts: datetime = EARLIER, cwd: str = "C:\\code\\alpha", **usage) -> dict:
    return {
        "type": "assistant",
        "timestamp": _iso(ts),
        "requestId": f"req_{msg_id}",
        "cwd": cwd,
        "message": {
            "id": msg_id,
            "model": "claude-opus-5-5",
            "role": "assistant",
            "content": [{"type": "text", "text": "secret reply"}],
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 100,
                "output_tokens": 50,
                **usage,
            },
        },
    }


def _codex_count(total: int, *, ts: datetime = EARLIER, used: float = 42.0) -> dict:
    return {
        "timestamp": _iso(ts),
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {"total_tokens": total},
                "last_token_usage": {"input_tokens": 400, "cached_input_tokens": 300, "output_tokens": 20},
            },
            "rate_limits": {
                "primary": {"used_percent": used, "window_minutes": 10080, "resets_at": time.time() + 3600},
                "secondary": None,
                "plan_type": "plus",
            },
        },
    }


def _write(path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "claude", tmp_path / "codex"


@pytest.fixture
def source(dirs, tmp_path):
    claude, codex = dirs
    return CodingToolSource(claude, codex, plan_usage=False, cache_dir=tmp_path / "cache")


def test_claude_messages_are_counted_once_across_resumed_sessions(dirs, source):
    claude, _ = dirs
    first = claude / "projects" / "C--code-alpha" / "one.jsonl"
    _write(first, [_claude("m1"), _claude("m1"), {"type": "user", "message": {"content": "hi"}}])
    # A resumed session copies m1 into its own file.
    _write(claude / "projects" / "C--code-alpha" / "two.jsonl", [_claude("m1"), _claude("m2", output_tokens=5)])

    calls = source.calls("day")
    assert sorted(c.id for c in calls) == ["claude:m1:req_m1", "claude:m2:req_m2"]
    m1 = next(c for c in calls if c.id.endswith("m1"))
    assert (m1.model, m1.input_tokens, m1.output_tokens, m1.project) == (
        "claude/claude-opus-5-5",
        1110,
        50,
        "alpha",
    )
    assert not any("secret" in json.dumps(source.event(c)) for c in calls)


def test_codex_calls_take_the_turn_model_and_skip_repeated_counts(dirs, source):
    _, codex = dirs
    log = codex / "sessions" / "2026" / "09" / "24" / "rollout-a.jsonl"
    _write(
        log,
        [
            {"timestamp": _iso(EARLIER), "type": "session_meta", "payload": {"id": "s1", "cwd": "/home/me/beta"}},
            {"timestamp": _iso(EARLIER), "type": "turn_context", "payload": {"model": "gpt-6-astra"}},
            _codex_count(420),
            _codex_count(420),  # repeated without a new call
            _codex_count(840, ts=EARLIER + timedelta(seconds=5), used=50.0),
        ],
    )
    calls = source.calls("day")
    assert len(calls) == 2
    assert {c.model for c in calls} == {"openai/gpt-6-astra"}
    assert calls[0].project == "beta" and calls[0].tokens == 420

    limits = source.rate_limits()["openai/codex"]
    assert limits["source"] == "plan" and limits["plan"] == "plus"
    assert limits["windows"]["weekly"]["remaining"] == 50
    assert limits["headroom"] == pytest.approx(0.5)


def test_reads_only_new_complete_lines(dirs, source):
    claude, _ = dirs
    log = claude / "projects" / "p" / "s.jsonl"
    _write(log, [_claude("m1")])
    assert len(source.refresh(force=True)) == 1
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_claude("m2"))[:40])  # half-written line
    assert source.refresh(force=True) == []
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_claude("m2"))[40:] + "\n")
    assert [c.id for c in source.refresh(force=True)] == ["claude:m2:req_m2"]


def test_spend_and_live_events(dirs, source):
    claude, _ = dirs
    log = claude / "projects" / "p" / "s.jsonl"
    _write(log, [_claude("m1")])
    spend = summarize_spend(source.rows("day"), "day")
    assert spend["total"]["requests"] == 1 and spend["by_project"][0]["project"] == "alpha"

    bus = LiveBus()
    source.refresh(force=True)
    _write(log, [_claude("m2", ts=NOW)])
    new = source.refresh(force=True)
    for call in new:
        bus.publish("request.finished", source.event(call))
    event = bus.replay(0)[0]
    assert event["key_name"] == "Claude Code" and event["model"] == "claude/claude-opus-5-5"


def test_claude_plan_usage_is_cached_and_backs_off(dirs, tmp_path, monkeypatch):
    claude, codex = dirs
    claude.mkdir(parents=True)
    (claude / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "tok",
                    "expiresAt": (time.time() + 3600) * 1000,
                    "subscriptionType": "max",
                }
            }
        )
    )
    resets = _iso(NOW + timedelta(hours=2))
    body = {
        "five_hour": {"utilization": 10.0, "resets_at": resets},
        "seven_day": {"utilization": 41.0, "resets_at": resets},
        "seven_day_opus": {"utilization": 0.0, "resets_at": None},
        "seven_day_oauth_apps": None,
        "extra_usage": {"is_enabled": False},
    }
    requests = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(body).encode()

    def fake_urlopen(request, timeout):
        requests.append(request)
        return Resp()

    monkeypatch.setattr(coding_agent_readers.urllib.request, "urlopen", fake_urlopen)
    src = CodingToolSource(claude, codex, plan_usage=True, cache_dir=tmp_path / "cache")
    snap = src.rate_limits()["claude/claude-code"]
    assert snap["plan"] == "max"
    assert snap["windows"]["session_5h"]["remaining"] == 90
    assert snap["windows"]["weekly"]["remaining"] == 59
    assert snap["windows"]["weekly_opus"]["remaining"] == 100
    assert snap["tightest_window"] == "weekly"
    assert requests[0].get_header("Authorization") == "Bearer tok"

    # Within the interval, and from a new process via the cache file: no second request.
    src.rate_limits()
    again = CodingToolSource(claude, codex, plan_usage=True, cache_dir=tmp_path / "cache")
    assert again.rate_limits()["claude/claude-code"]["windows"]["weekly"]["remaining"] == 59
    assert len(requests) == 1


def test_expired_claude_token_is_never_used(dirs, tmp_path, monkeypatch):
    claude, codex = dirs
    claude.mkdir(parents=True)
    (claude / ".credentials.json").write_text(
        json.dumps({"claudeAiOauth": {"accessToken": "tok", "expiresAt": (time.time() - 60) * 1000}})
    )
    monkeypatch.setattr(coding_agent_readers.urllib.request, "urlopen", lambda *a, **k: pytest.fail("called"))
    src = CodingToolSource(claude, codex, plan_usage=True, cache_dir=tmp_path / "cache")
    assert src.rate_limits() == {}


def test_claude_plan_usage_is_off_unless_opted_in(dirs, tmp_path, monkeypatch):
    claude, codex = dirs
    monkeypatch.delenv("PROMPTURE_CLAUDE_PLAN_USAGE", raising=False)
    assert CodingToolSource(claude, codex).plan_usage is False
    monkeypatch.setenv("PROMPTURE_CLAUDE_PLAN_USAGE", "1")
    assert CodingToolSource(claude, codex).plan_usage is True
