"""claude-code/<model> and codex/<model>: a signed-in coding-agent CLI used as a model."""

from __future__ import annotations

import asyncio

import pytest

from prompture.drivers import coding_agent_driver
from prompture.drivers.coding_agent_driver import ClaudeCodeDriver, CodexDriver, CodingAgentError
from prompture.infra.coding_agent_events import CodingAgentEvent
from prompture.infra.coding_agents import CodingAgentRunResult


def fake_run(calls, events=(), returncode=0, output="", timed_out=False, cost=None):
    def run(agent, task, **kwargs):
        calls.append({"agent": agent, "task": task, **kwargs})
        return CodingAgentRunResult(
            agent=agent,
            command=[agent],
            cwd=kwargs["cwd"],
            returncode=returncode,
            output=output,
            timed_out=timed_out,
            events=tuple(events),
            cost_usd=cost,
            input_tokens=12,
            output_tokens=3,
        )

    return run


def test_claude_code_answers_with_its_result_and_no_bill(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        coding_agent_driver,
        "run_coding_agent",
        fake_run(
            calls,
            [CodingAgentEvent(type="message", text="thinking"), CodingAgentEvent(type="done", text="Yellow.")],
            cost=0.0067,
        ),
    )
    result = ClaudeCodeDriver(model="sonnet").generate_messages(
        [{"role": "system", "content": "One word."}, {"role": "user", "content": "Banana colour?"}], {"timeout": 30}
    )
    assert result["text"] == "Yellow."
    assert result["meta"]["cost"] == 0.0 and result["meta"]["reported_cost"] == 0.0067
    assert result["meta"]["total_tokens"] == 15 and result["meta"]["model_name"] == "claude-code/sonnet"
    call = calls[0]
    assert call["agent"] == "claude" and call["model"] == "sonnet" and call["timeout"] == 30
    assert call["extra_args"] == ["--max-turns", "1", "--system-prompt", "One word."]
    assert "One word." not in call["task"] and "Banana colour?" in call["task"]
    assert call["output_format"] == "json"


def test_default_leaves_the_model_to_the_cli_and_codex_takes_effort(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        coding_agent_driver, "run_coding_agent", fake_run(calls, [CodingAgentEvent(type="message", text="Yellow")])
    )
    result = CodexDriver().generate_messages(
        [{"role": "system", "content": "One word."}, {"role": "user", "content": "Banana?"}], {"effort": "low"}
    )
    assert result["text"] == "Yellow"
    call = calls[0]
    assert call["agent"] == "codex" and call["model"] is None
    assert call["extra_args"] == ["--skip-git-repo-check", "--sandbox", "read-only", "-c", "model_reasoning_effort=low"]
    assert call["task"].startswith("One word.")  # Codex has no system-prompt flag


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timed_out": True, "returncode": -1}, "did not answer in time"),
        ({"returncode": 1, "events": [CodingAgentEvent(type="error", error="not signed in")]}, "not signed in"),
        ({"returncode": 1, "output": "boom\nUpgrade Codex"}, "Upgrade Codex"),
        ({"returncode": 0}, "no answer"),
    ],
)
def test_a_cli_that_does_not_answer_raises(monkeypatch, kwargs, message):
    monkeypatch.setattr(coding_agent_driver, "run_coding_agent", fake_run([], **kwargs))
    with pytest.raises(CodingAgentError, match=message):
        ClaudeCodeDriver().generate("hi", {})


def test_the_model_strings_resolve_and_the_async_driver_runs_in_a_thread(monkeypatch):
    from prompture.drivers import get_async_driver_for_model, get_driver_for_model

    assert isinstance(get_driver_for_model("claude-code/opus"), ClaudeCodeDriver)
    assert get_driver_for_model("codex/gpt-6-sol").model == "gpt-6-sol"
    assert ClaudeCodeDriver.list_models() == ["default", "opus", "sonnet", "haiku"]
    monkeypatch.setattr(
        coding_agent_driver, "run_coding_agent", fake_run([], [CodingAgentEvent(type="done", text="ok")])
    )
    driver = get_async_driver_for_model("claude-code/haiku")
    assert asyncio.run(driver.generate("hi", {}))["text"] == "ok"
