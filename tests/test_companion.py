"""The companion API package: aggregations, ledger source and the local server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pytest

from prompture.companion import (
    CompanionServer,
    LedgerSource,
    LiveBus,
    UsageRow,
    provider_limits,
    read_state,
    running_instance,
    summarize_spend,
    window_end,
    window_start,
)
from prompture.infra.tracker import UsageEvent, UsageTracker

NOW = datetime(2026, 9, 24, 15, 30, tzinfo=timezone.utc)  # a Thursday


class TestWindows:
    def test_day_week_month(self):
        assert window_start("day", NOW) == datetime(2026, 9, 24, tzinfo=timezone.utc)
        assert window_start("week", NOW) == datetime(2026, 9, 21, tzinfo=timezone.utc)
        assert window_start("month", NOW) == datetime(2026, 9, 1, tzinfo=timezone.utc)
        assert window_end("month", NOW) == datetime(2026, 10, 1, tzinfo=timezone.utc)
        assert window_end("week", NOW) == datetime(2026, 9, 28, tzinfo=timezone.utc)


class TestSummaries:
    def test_spend_splits_by_project_key_and_serving_model(self):
        rows = [
            UsageRow(model="combo/cheap", served_by="groq/llama", cost_usd=0.5, tokens=100, project="shop", key_id=1),
            UsageRow(model="openai/gpt-4o", cost_usd=1.0, tokens=50, project="shop", key_id=2),
            UsageRow(model="openai/gpt-4o", cost_usd=0.0, tokens=0, status="error"),
        ]
        body = summarize_spend(rows, "day", key_names={1: "app"}, now=NOW)
        assert body["total"] == {"requests": 3, "cost_usd": 1.5, "tokens": 150, "errors": 1}
        assert body["by_project"][0] == {"project": "shop", "requests": 2, "cost_usd": 1.5, "tokens": 150, "errors": 0}
        assert [k["name"] for k in body["by_key"]] == ["key 2", "app"]
        assert {m["model"] for m in body["by_model"]} == {"groq/llama", "openai/gpt-4o"}
        assert body["resets_at"] == "2026-09-25T00:00:00+00:00"

    def test_provider_limits_compute_current_headroom(self):
        now = time.time()
        data = {"observed_at": now, "windows": {"tokens": {"limit": 100, "remaining": 20, "resets_at": now + 30}}}
        [entry] = provider_limits({"openai/gpt-4o": data})
        assert entry["target"] == "openai/gpt-4o"
        assert entry["current_headroom"] == pytest.approx(0.2)
        assert entry["current_window"] == "tokens"


def _ledger(tmp_path) -> tuple[UsageTracker, LedgerSource]:
    db = tmp_path / "usage.db"
    return UsageTracker(db_path=db, flush_threshold=1), LedgerSource(db)


def _record(tracker: UsageTracker, **kw) -> None:
    defaults = {"model_name": "openai/gpt-4o-mini", "provider": "openai", "cost": 0.25, "total_tokens": 40}
    tracker.record(UsageEvent(**{**defaults, **kw}))


class TestLedgerSource:
    def test_missing_ledger_is_empty(self, tmp_path):
        source = LedgerSource(tmp_path / "nope.db")
        assert source.rows() == []
        assert source.rate_limits() == {}
        assert source.last_rowid() == 0

    def test_rows_projects_routes_and_rate_limits(self, tmp_path):
        tracker, source = _ledger(tmp_path)
        _record(tracker, tags=["project:shop"])
        _record(
            tracker,
            model_name="combo/cheap",
            status="error",
            metadata={
                "route": {"served_by": "groq/llama", "attempts": [{"outcome": "error"}, {"outcome": "ok"}]},
                "rate_limits": {"observed_at": 1.0, "windows": {"requests": {"limit": 10, "remaining": 2}}},
            },
        )
        rows = source.rows("day")
        assert [(r.model, r.served_by, r.project, r.status) for r in rows] == [
            ("openai/gpt-4o-mini", None, "shop", "ok"),
            ("combo/cheap", "groq/llama", None, "error"),
        ]
        assert source.rate_limits()["combo/cheap"]["windows"]["requests"]["remaining"] == 2

        [(_, first), (_, second)] = source.events_after(0)
        assert first["project"] == "shop" and first["status"] == "ok"
        assert second["fallback"] is True and second["served_by"] == "groq/llama"

    def test_tail_publishes_new_rows(self, tmp_path):
        import threading

        tracker, source = _ledger(tmp_path)
        _record(tracker)  # existing rows are not replayed
        bus, stop = LiveBus(), threading.Event()
        thread = threading.Thread(target=source.tail, args=(bus, stop, 0.1), daemon=True)
        thread.start()
        time.sleep(0.3)
        _record(tracker, tags=["project:blog"])
        time.sleep(0.5)
        stop.set()
        thread.join(2)
        events = bus.replay(0)
        assert [(e["type"], e["project"]) for e in events] == [("request.finished", "blog")]


@pytest.fixture
def server(tmp_path):
    tracker, source = _ledger(tmp_path)
    srv = CompanionServer(source, token="t0ken", bus=LiveBus(), state_path=tmp_path / "companion.json")
    srv.start_background()
    yield srv, tracker
    srv.shutdown()
    srv.shutdown_companion()


def _get(url: str, token: str | None = "t0ken"):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, json.loads(resp.read())


class TestServer:
    def test_info_is_public_and_describes_local_mode(self, server):
        srv, _ = server
        status, body = _get(f"{srv.url}/v1/companion/info", token=None)
        assert status == 200
        assert body["service"] == "prompture" and body["mode"] == "local"
        assert body["api_version"] == 1
        assert body["capabilities"]["key_controls"] is False
        assert set(body["features"]) == {"live", "limits", "spend", "alerts", "tools", "activity"}
        assert body["capabilities"]["coding_tools"] is False and body["capabilities"]["activity"] is True

    def test_activity_merges_ledger_and_coding_tools_by_local_day(self, tmp_path):
        from prompture.companion import CodingToolSource
        from prompture.infra.coding_agent_readers import ContinueReader

        log = tmp_path / "continue" / "dev_data" / "0.2.0" / "tokensGenerated.jsonl"
        log.parent.mkdir(parents=True)
        ts = datetime.now(timezone.utc).isoformat()
        log.write_text(
            json.dumps({"timestamp": ts, "model": "m", "provider": "ollama", "promptTokens": 30, "generatedTokens": 12})
            + "\n"
        )
        tracker, source = _ledger(tmp_path)
        _record(tracker, cost=0.5)
        _record(tracker, cost=0.25)
        tools = CodingToolSource(readers=[ContinueReader(tmp_path / "continue")])
        srv = CompanionServer(source, token="t0ken", bus=LiveBus(), state_path=None, coding_tools=tools)
        srv.start_background()
        try:
            _, body = _get(f"{srv.url}/v1/activity?days=7&tz_offset=0")
            today = datetime.now(timezone.utc).date().isoformat()
            assert body["end"] == today and len(body["days"]) == 1
            day = body["days"][0]
            assert day["date"] == today and day["requests"] == 3
            assert day["cost_usd"] == pytest.approx(0.75)
            assert {s["name"] for s in day["sources"]} == {"Prompture", "Continue"}
            with pytest.raises(urllib.error.HTTPError) as err:
                _get(f"{srv.url}/v1/activity?days=soon")
            assert err.value.code == 422
        finally:
            srv.shutdown()
            srv.shutdown_companion()

    def test_tools_needs_coding_tools(self, server):
        srv, _ = server
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(f"{srv.url}/v1/tools")
        assert err.value.code == 404

    def test_coding_tool_calls_join_spend_and_tools(self, tmp_path, monkeypatch):
        from prompture.companion import CodingToolSource
        from prompture.companion import coding_tools as ct
        from prompture.infra.coding_agent_readers import ContinueReader

        monkeypatch.setattr(
            ct,
            "coding_agents_overview",
            lambda: [{"id": "continue", "name": "Continue", "installed": True, "runnable": False, "usage": True}],
        )
        log = tmp_path / "continue" / "dev_data" / "0.2.0" / "tokensGenerated.jsonl"
        log.parent.mkdir(parents=True)
        ts = datetime.now(timezone.utc).isoformat()
        log.write_text(
            json.dumps({"timestamp": ts, "model": "m", "provider": "ollama", "promptTokens": 30, "generatedTokens": 12})
            + "\n"
        )
        _, source = _ledger(tmp_path)
        tools = CodingToolSource(readers=[ContinueReader(tmp_path / "continue")], prefs_file=tmp_path / "prefs.json")
        srv = CompanionServer(source, token="t0ken", bus=LiveBus(), state_path=None, coding_tools=tools)
        srv.start_background()
        try:
            # Budgets measure API calls only: coding-tool usage stays out of sources=api.
            assert _get(f"{srv.url}/v1/spend?period=day&sources=api")[1]["total"]["tokens"] == 0
            req = urllib.request.Request(
                f"{srv.url}/v1/tools/claude-plan",
                data=json.dumps({"enabled": True}).encode(),
                headers={"Authorization": "Bearer t0ken", "Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert json.loads(resp.read()) == {"claude_plan_usage": False}  # no Claude reader in this source
            assert json.loads((tmp_path / "prefs.json").read_text()) == {"claude_plan_usage": True}
            _, info = _get(f"{srv.url}/v1/companion/info", token=None)
            assert info["capabilities"]["coding_tools"] is True
            _, body = _get(f"{srv.url}/v1/tools?period=day")
            assert body["period"] == "day" and body["installed"][0]["id"] == "continue"
            agent = body["agents"][0]
            assert (agent["agent"], agent["name"], agent["requests"], agent["tokens"]) == (
                "continue",
                "Continue",
                1,
                42,
            )
            _, spend = _get(f"{srv.url}/v1/spend?period=day")
            assert spend["total"]["tokens"] == 42
        finally:
            srv.shutdown()
            srv.shutdown_companion()

    def test_token_required(self, server):
        srv, _ = server
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(f"{srv.url}/v1/spend", token="wrong")
        assert err.value.code == 401

    def test_spend_limits_and_alerts(self, server):
        srv, tracker = server
        _record(tracker, tags=["project:shop"], cost=0.5)
        _record(tracker, cost=0.25)
        _, spend = _get(f"{srv.url}/v1/spend?period=week")
        assert spend["period"] == "week"
        assert spend["total"]["requests"] == 2
        assert spend["total"]["cost_usd"] == pytest.approx(0.75)
        assert {p["project"] for p in spend["by_project"]} == {"shop", None}
        _, limits = _get(f"{srv.url}/v1/limits?accounts=false")
        assert limits["keys"] == [] and limits["paused_providers"] == [] and limits["accounts"] is None
        assert _get(f"{srv.url}/v1/alerts")[1] == []
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(f"{srv.url}/v1/spend?period=year")
        assert err.value.code == 422

    def test_live_stream_delivers_new_ledger_rows(self, server):
        srv, tracker = server
        import threading

        def later():
            time.sleep(1.5)
            _record(tracker, tags=["project:live"])

        threading.Thread(target=later, daemon=True).start()
        req = urllib.request.Request(f"{srv.url}/v1/live?limit=2", headers={"Authorization": "Bearer t0ken"})
        events = []
        with urllib.request.urlopen(req, timeout=10) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        assert [e["type"] for e in events] == ["snapshot", "request.finished"]
        assert events[1]["project"] == "live"


def test_state_file_round_trip(tmp_path):
    _, source = _ledger(tmp_path)
    state_path = tmp_path / "companion.json"
    srv = CompanionServer(source, bus=LiveBus(), state_path=state_path)
    import threading

    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(50):
        if read_state(state_path):
            break
        time.sleep(0.05)
    state = read_state(state_path)
    assert state["url"] == srv.url and state["token"] == srv.token
    assert running_instance(state_path) == state
    srv.shutdown()
    thread.join(5)
    assert read_state(state_path) is None


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "https://127.0.0.1:5123", "http://example.com:80", "http://127.0.0.1"],
)
def test_running_instance_only_trusts_loopback_http(tmp_path, url):
    state_path = tmp_path / "companion.json"
    state_path.write_text(json.dumps({"url": url, "token": "t", "pid": 1}))
    assert running_instance(state_path) is None


def test_stops_when_owner_process_exits(tmp_path):
    import os
    import subprocess
    import sys
    import threading

    from prompture.companion.server import process_alive, stop_when_process_exits

    assert process_alive(os.getpid())
    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
    _, source = _ledger(tmp_path)
    srv = CompanionServer(source, bus=LiveBus(), state_path=tmp_path / "c.json")
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    stop_when_process_exits(srv, owner.pid, interval=0.2)
    owner.wait()
    thread.join(5)
    assert not thread.is_alive()
    assert not process_alive(owner.pid)


def test_module_entry_point_runs_the_cli():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "prompture", "companion", "--help"], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0
    assert "--exit-with-pid" in out.stdout
