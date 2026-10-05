"""``prompture delegate*`` — handing a task to the companion's automation queue."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from prompture.cli import delegate_cmd
from prompture.cli.cli import cli
from prompture.companion import CompanionServer, LedgerSource, LiveBus
from prompture.companion.automations import Automations
from prompture.infra.coding_agent_events import CodingAgentEvent


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
def comp(tmp_path, agent, monkeypatch):
    """A real companion with an automation queue, and the CLI pointed at it."""
    monkeypatch.setattr(delegate_cmd, "POLL_SECONDS", 0.02)
    auto = Automations(tmp_path / "auto", bus=LiveBus(), runner=agent, installed=lambda _id: True)
    srv = CompanionServer(
        LedgerSource(tmp_path / "usage.db"),
        token="t0ken",
        bus=LiveBus(),
        state_path=tmp_path / "companion.json",
        automations=auto,
    )
    srv.start_background()
    with patch.object(delegate_cmd, "_ensure_companion", return_value=delegate_cmd._Companion(srv.url, "t0ken")):
        yield srv, auto
    srv.shutdown()
    srv.shutdown_companion()
    if auto.run and auto.run.status not in delegate_cmd.ENDED:
        auto.stop()


def invoke(*args):
    return CliRunner().invoke(cli, list(args))


def test_delegate_streams_to_done(comp, agent, tmp_path):
    result = invoke("delegate", "--cwd", str(tmp_path), "translate", "README.md")

    assert result.exit_code == 0, result.output
    assert agent.calls[0]["task"] == "translate README.md"
    assert agent.calls[0]["approval_mode"] == "auto"  # unattended: no permission prompts
    assert "step 1/1: translate README.md" in result.output
    assert "finished" in result.output


def test_delegate_defaults_to_auto_agent(comp, agent, tmp_path):
    result = invoke("delegate", "--cwd", str(tmp_path), "tidy", "up")

    assert result.exit_code == 0, result.output
    assert agent.calls[0]["agent"] in ("claude", "codex")  # auto resolves to one of them


def test_delegate_no_wait_returns_the_run_id(comp, agent, tmp_path):
    agent.gate.clear()
    result = invoke("delegate", "--cwd", str(tmp_path), "--no-wait", "slow", "task")

    assert result.exit_code == 0, result.output
    assert "delegate-status" in result.output
    agent.gate.set()


def test_delegate_question_pauses_and_answer_continues(comp, agent, tmp_path):
    agent.replies["pick a db"] = {"text": "Two options.\n\nShould I use SQLite or Postgres?"}
    asked = invoke("delegate", "--cwd", str(tmp_path), "pick", "a", "db")

    assert asked.exit_code == delegate_cmd.EXIT_ASK, asked.output
    assert "Should I use SQLite or Postgres?" in asked.output
    assert "delegate-answer" in asked.output

    answered = invoke("delegate-answer", "SQLite")
    assert answered.exit_code == 0, answered.output
    assert agent.calls[1]["task"] == "SQLite"
    assert agent.calls[1]["session_id"] == "s1"  # the answer continues the step's session


def test_delegate_failure_exits_1_and_resume_retries(comp, agent, tmp_path):
    agent.replies["build"] = {"error": "build exited with code 1"}
    failed = invoke("delegate", "--cwd", str(tmp_path), "build")

    assert failed.exit_code == 1, failed.output
    assert "delegate-resume" in failed.output

    agent.replies["build"] = {}
    resumed = invoke("delegate-resume")
    assert resumed.exit_code == 0, resumed.output
    assert [c["task"] for c in agent.calls] == ["build", "build"]


def test_delegate_json_prints_the_run(comp, agent, tmp_path):
    result = invoke("delegate", "--cwd", str(tmp_path), "--json", "sum", "up")

    assert result.exit_code == 0, result.output
    import json

    run = json.loads(result.output[result.output.index("{") :])
    assert run["status"] == "finished"
    assert run["steps"][0]["text"] == "sum up"


def test_cost_cap_does_not_block_a_single_step(comp, agent, tmp_path):
    agent.replies["spend"] = {"cost": 6.0}
    result = invoke("delegate", "--cwd", str(tmp_path), "--cost-cap", "5", "spend")

    # caps bite between steps, so a one-step run still finishes
    assert result.exit_code == 0, result.output


def test_delegate_while_a_queue_is_running(comp, agent, tmp_path):
    agent.gate.clear()
    first = invoke("delegate", "--cwd", str(tmp_path), "--no-wait", "first")
    assert first.exit_code == 0, first.output

    second = invoke("delegate", "--cwd", str(tmp_path), "second")
    assert second.exit_code != 0
    assert "already running a queue" in second.output
    agent.gate.set()


def test_delegate_status_shows_the_queue(comp, agent, tmp_path):
    empty = invoke("delegate-status")
    assert empty.exit_code == 0
    assert "No queue running" in empty.output

    agent.gate.clear()
    invoke("delegate", "--cwd", str(tmp_path), "--no-wait", "running", "task")
    shown = invoke("delegate-status")
    assert shown.exit_code == 0
    assert "running task" in shown.output
    agent.gate.set()


def test_delegate_answer_without_a_question(comp, agent, tmp_path):
    result = invoke("delegate-answer", "hello")

    assert result.exit_code != 0
    assert "No queue is running" in result.output


def test_ensure_companion_reuses_the_running_one():
    state = {"url": "http://127.0.0.1:47811", "token": "abc"}
    with patch("prompture.companion.running_instance", return_value=state):
        client = delegate_cmd._ensure_companion()
    assert client.url == state["url"] and client.token == "abc"


def test_ensure_companion_starts_one_when_missing():
    state = {"url": "http://127.0.0.1:47811", "token": "abc"}
    calls = {"n": 0}

    def eventually_running(**_kw):
        calls["n"] += 1
        return state if calls["n"] > 1 else None

    with (
        patch("prompture.companion.running_instance", side_effect=eventually_running),
        patch.object(delegate_cmd.subprocess, "Popen") as popen,
    ):
        client = delegate_cmd._ensure_companion()

    assert popen.called
    argv = popen.call_args[0][0]
    assert argv[1:] == ["-m", "prompture", "companion"]
    assert client.url == state["url"]


def test_ensure_companion_reports_a_failed_start():
    import click

    with (
        patch("prompture.companion.running_instance", return_value=None),
        patch.object(delegate_cmd, "_start_companion", return_value=None),
        pytest.raises(click.ClickException, match="starting one failed"),
    ):
        delegate_cmd._ensure_companion()
