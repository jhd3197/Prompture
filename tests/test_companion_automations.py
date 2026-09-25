"""Automations: queued coding-agent steps, run by the companion one after another."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request

import pytest

from prompture.companion import CompanionServer
from prompture.companion.automations import AutomationError, Automations, describe, roadmap_steps
from prompture.companion.live import LiveBus
from prompture.infra.coding_agent_events import CodingAgentEvent


def wait_until(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("timed out")


class FakeAgent:
    """Stands in for ``astream_coding_agent``: each task maps to a reply, recorded as it runs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.replies: dict[str, dict] = {}
        self.gate = threading.Event()
        self.gate.set()
        self.session = 0

    def __call__(self, agent, task, **kw):
        self.calls.append({"agent": agent, "task": task, **kw})
        reply = self.replies.get(task, {})
        gate = self.gate
        fake = self

        async def stream():
            fake.session += 1
            yield CodingAgentEvent(type="system", session_id=kw.get("session_id") or f"s{fake.session}")
            yield CodingAgentEvent(type="tool_call", tool_name="Edit", tool_input={"file_path": "app.py"})
            while not gate.is_set():
                await asyncio.sleep(0.01)
            if reply.get("error"):
                yield CodingAgentEvent(type="error", error=reply["error"])
                return
            yield CodingAgentEvent(type="done", text=reply.get("text", "Done."), cost_usd=reply.get("cost", 1.0))

        return stream()


@pytest.fixture
def agent():
    return FakeAgent()


@pytest.fixture
def auto(tmp_path, agent):
    a = Automations(tmp_path / "auto", bus=LiveBus(), runner=agent, installed=lambda _id: True)
    yield a
    if a.run and a.run.status not in ("finished", "failed", "stopped"):
        a.stop()


def start(auto, tmp_path, steps, **extra):
    return auto.start({"cwd": str(tmp_path), "agent": "claude", "steps": steps, **extra})


def status(auto):
    return auto.run.status if auto.run else None


def test_runs_steps_in_order_and_carries_sessions(auto, agent, tmp_path):
    start(auto, tmp_path, [{"text": "one"}, {"text": "two"}, {"text": "three", "session": "new"}])
    wait_until(lambda: status(auto) == "finished")
    assert [c["task"] for c in agent.calls] == ["one", "two", "three"]
    assert agent.calls[0]["session_id"] is None
    assert agent.calls[1]["session_id"] == "s1"  # "same": continues step one's session
    assert agent.calls[2]["session_id"] is None  # "new"
    assert all(c["approval_mode"] == "auto" for c in agent.calls)
    run = auto.state()["current"]
    assert [s["status"] for s in run["steps"]] == ["done", "done", "done"]
    assert run["cost_usd"] == 3.0
    assert auto.state()["history"][0]["id"] == run["id"]


def test_failure_pauses_and_resume_retries(auto, agent, tmp_path):
    agent.replies["build"] = {"error": "build exited with code 1"}
    start(auto, tmp_path, ["build", "ship"])
    wait_until(lambda: status(auto) == "paused")
    assert auto.run.reason == "fail"
    assert auto.run.steps[0].status == "failed"
    agent.replies["build"] = {}
    auto.resume()
    wait_until(lambda: status(auto) == "finished")
    assert [c["task"] for c in agent.calls] == ["build", "build", "ship"]


def test_failure_can_be_skipped_or_ignored(auto, agent, tmp_path):
    agent.replies["a"] = {"error": "nope"}
    start(auto, tmp_path, ["a", "b"], stop={"fail": False})
    wait_until(lambda: status(auto) == "finished")
    assert [s.status for s in auto.run.steps] == ["failed", "done"]


def test_question_pauses_and_answer_continues_the_session(auto, agent, tmp_path):
    agent.replies["plan"] = {"text": "Two ways to do this.\n\nShould I use SQLite or Postgres?"}
    start(auto, tmp_path, ["plan", "next"])
    wait_until(lambda: status(auto) == "paused")
    assert auto.run.reason == "ask"
    assert auto.run.question == "Should I use SQLite or Postgres?"
    auto.answer("SQLite")
    wait_until(lambda: status(auto) == "finished")
    assert [c["task"] for c in agent.calls] == ["plan", "SQLite", "next"]
    assert agent.calls[1]["session_id"] == "s1"
    assert any(line["text"] == "You: SQLite" for line in auto.log(auto.run.id, auto.run.steps[0].id))


def test_skip_and_stop_a_running_step(auto, agent, tmp_path):
    agent.gate.clear()
    start(auto, tmp_path, ["slow", "after", "never"])
    wait_until(lambda: auto.run.steps[0].status == "running" and auto.run.steps[0].action)
    assert auto.run.steps[0].action == "Editing app.py…"
    auto.skip()
    wait_until(lambda: auto.run.steps[1].status == "running")
    assert auto.run.steps[0].status == "skipped"
    auto.stop()
    assert status(auto) == "stopped"
    assert [s.status for s in auto.run.steps] == ["skipped", "stopped", "waiting"]
    with pytest.raises(AutomationError):
        auto.pause()


def test_pause_waits_for_the_running_step(auto, agent, tmp_path):
    agent.gate.clear()
    start(auto, tmp_path, ["a", "b"])
    wait_until(lambda: auto.run.steps[0].status == "running")
    auto.pause()
    agent.gate.set()
    wait_until(lambda: status(auto) == "paused")
    assert auto.run.reason == "manual"
    assert auto.run.steps[0].status == "done" and auto.run.steps[1].status == "waiting"
    auto.resume()
    wait_until(lambda: status(auto) == "finished")


def test_cost_cap_pauses_before_the_next_step(auto, agent, tmp_path):
    agent.replies["a"] = {"cost": 6.0}
    start(auto, tmp_path, ["a", "b"], stop={"cost_usd": 5})
    wait_until(lambda: status(auto) == "paused")
    assert auto.run.reason == "cost"
    auto.resume()  # "Continue": no cap for the rest of the run
    wait_until(lambda: status(auto) == "finished")


def test_plan_limit_pauses_until_the_window_resets(tmp_path, agent):
    reset = time.time() + 3600
    windows = {"session_5h": {"limit": 100, "remaining": 4, "resets_at": reset}}
    limits = {"claude/claude-code": {"windows": windows}}
    auto = Automations(tmp_path / "a", runner=agent, limits=lambda: limits, installed=lambda _id: True)
    start(auto, tmp_path, ["a"])
    wait_until(lambda: status(auto) == "paused")
    assert auto.run.reason == "limit" and auto.run.resumes_at == reset
    assert agent.calls == []
    auto.resume()  # "Resume now"
    wait_until(lambda: status(auto) == "finished")


def test_usage_limit_error_waits_and_continues_the_session(auto, agent, tmp_path):
    agent.replies["a"] = {"error": "Claude AI usage limit reached|1900000000"}
    start(auto, tmp_path, ["a"])
    wait_until(lambda: status(auto) == "paused")
    assert auto.run.reason == "limit" and auto.run.resumes_at == 1900000000
    auto.resume()
    wait_until(lambda: status(auto) == "finished")
    assert agent.calls[1]["task"] == "Continue where you left off."
    assert agent.calls[1]["session_id"] == "s1"


def test_waiting_steps_can_change_while_it_runs(auto, agent, tmp_path):
    agent.gate.clear()
    run = start(auto, tmp_path, ["a", "b", "c"])
    wait_until(lambda: auto.run.steps[0].status == "running")
    b, c = run["steps"][1], run["steps"][2]
    auto.set_steps([{"id": c["id"], "text": "c"}, {"text": "d", "session": "new"}, {"id": b["id"], "text": "b2"}])
    assert [s.text for s in auto.run.steps] == ["a", "c", "d", "b2"]
    assert auto.run.steps[1].id == c["id"] and auto.run.steps[3].id == b["id"]
    agent.gate.set()
    wait_until(lambda: status(auto) == "finished")
    assert [x["task"] for x in agent.calls] == ["a", "c", "d", "b2"]


def test_only_one_queue_at_a_time_and_bad_input(auto, agent, tmp_path):
    with pytest.raises(AutomationError, match="project folder"):
        auto.start({"cwd": str(tmp_path / "missing"), "steps": ["a"]})
    with pytest.raises(AutomationError, match="step"):
        start(auto, tmp_path, ["  "])
    agent.gate.clear()
    start(auto, tmp_path, ["a"])
    with pytest.raises(AutomationError) as exc:
        start(auto, tmp_path, ["b"])
    assert exc.value.status == 409


def test_a_run_cut_off_by_a_restart_is_kept_as_stopped(tmp_path, agent):
    agent.gate.clear()
    first = Automations(tmp_path / "a", runner=agent, installed=lambda _id: True)
    start(first, tmp_path, ["a", "b"])
    wait_until(lambda: first.run.steps[0].status == "running")
    again = Automations(tmp_path / "a", runner=agent, installed=lambda _id: True)
    assert again.run is None
    row = again.state()["history"][0]
    assert row["status"] == "stopped" and row["note"] == "Prompture stopped"
    assert [s["status"] for s in row["steps"]] == ["stopped", "waiting"]
    first.stop()


def test_roadmap_lists_unchecked_phases(tmp_path):
    (tmp_path / ".planning").mkdir()
    (tmp_path / ".planning" / "ROADMAP.md").write_text(
        "- [x] **Phase 1: Done**\n- [ ] **Phase 2: Next** - x\n- [ ] **Phase 2.1: Hotfix**\n  - [ ] 02-01-PLAN.md\n",
        encoding="utf-8",
    )
    assert roadmap_steps(tmp_path) == ["/gsd:execute-phase 2", "/gsd:execute-phase 2.1"]
    assert roadmap_steps(tmp_path / "nope") == []


def test_describe_tool_calls(tmp_path):
    edit = CodingAgentEvent(
        type="tool_call", tool_name="Edit", tool_input={"file_path": str(tmp_path / "src" / "a.py")}
    )
    assert describe(edit, str(tmp_path)) == ("Editing src/a.py…", "Edit src/a.py")
    bash = CodingAgentEvent(type="tool_call", tool_name="Bash", tool_input={"command": "npm test"})
    assert describe(bash, str(tmp_path)) == ("Running npm test…", "$ npm test")


def test_http_routes(tmp_path, agent):
    auto = Automations(tmp_path / "a", runner=agent, installed=lambda _id: True)
    server = CompanionServer(state_path=None, token="t", automations=auto)
    server.start_background()
    try:

        def call(method, path, body=None):
            data = json.dumps(body).encode() if body is not None else None
            req = urllib.request.Request(server.url + path, data=data, method=method)
            req.add_header("Authorization", "Bearer t")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())

        info = call("GET", "/v1/companion/info")
        assert info["capabilities"]["automations"] is True
        assert info["features"]["automations"] == "/v1/automations"
        run = call("POST", "/v1/automations", {"cwd": str(tmp_path), "agent": "claude", "steps": [{"text": "a"}]})
        wait_until(lambda: auto.run.status == "finished")
        state = call("GET", "/v1/automations")
        assert state["current"]["id"] == run["id"]
        assert state["agents"][0] == {"id": "claude", "name": "Claude Code", "installed": True}
        step = state["current"]["steps"][0]
        lines = call("GET", f"/v1/automations/runs/{run['id']}/steps/{step['id']}/log")["lines"]
        assert [line["kind"] for line in lines] == ["cmd", "tool", "ok"]
        assert call("GET", f"/v1/automations/runs/{run['id']}")["status"] == "finished"
        with pytest.raises(urllib.error.HTTPError) as exc:
            call("POST", "/v1/automations/current/pause", {})
        assert exc.value.code == 409
    finally:
        server.shutdown()
        server.shutdown_companion()
