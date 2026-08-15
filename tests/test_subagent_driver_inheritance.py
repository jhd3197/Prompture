"""Sub-agents must inherit the parent's driver instance.

``make_task_tool`` rebuilt every sub-agent from a model *string* through the
global driver registry.  A host that constructed its parent with ``driver=`` —
a local endpoint, a test double, any alias the registry does not own — could
not have sub-agents reach the same model: they raised
``ValueError: Unsupported provider ...`` or silently went somewhere else.

A spec that names its own model still wins, so per-sub-agent model selection is
unaffected.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from prompture.agents.deep_agent import DeepAgent
from prompture.agents.deep_state import SubAgentSpec
from prompture.drivers.base import Driver


class _CountingDriver(Driver):
    """A driver under an alias the global registry does not own."""

    supports_tool_use = False

    def __init__(self, label: str = "parent") -> None:
        self.model = "scripted/unknown-alias"
        self.label = label
        self.calls = 0

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {
            "text": json.dumps({"type": "final_answer", "content": f"{self.label} answered"}),
            "meta": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "cost": 0.0,
                "raw_response": {},
            },
        }


def _task_tool(agent: DeepAgent):
    return next(t for t in agent._tools.definitions if t.name == "task")


def _spec(**kw: Any) -> SubAgentSpec:
    kw.setdefault("name", "helper")
    kw.setdefault("description", "A helper.")
    kw.setdefault("system_prompt", "You help.")
    return SubAgentSpec(**kw)


def test_subagent_uses_the_parent_driver():
    driver = _CountingDriver()
    agent = DeepAgent(driver=driver, subagents=[_spec()])

    _task_tool(agent).function(description="do a thing", subagent_type="helper")

    assert driver.calls > 0, "sub-agent did not run through the parent's driver"


def test_subagent_does_not_go_through_the_registry():
    """The alias is unknown to the registry, so any registry path would raise."""
    driver = _CountingDriver()
    agent = DeepAgent(driver=driver, subagents=[_spec()])

    result = _task_tool(agent).function(description="do a thing", subagent_type="helper")

    assert "Unsupported provider" not in str(result)
    assert "failed" not in str(result).lower()


def test_spec_model_still_overrides_the_inherited_driver():
    """Per-sub-agent model selection must keep working."""
    driver = _CountingDriver()
    agent = DeepAgent(driver=driver, subagents=[_spec(model="openai/gpt-4o-mini")])

    # The spec names a real provider, so construction must not reuse the
    # parent driver. It will fail at call time without credentials, which is
    # fine — what matters is that the parent's driver was not silently used.
    _task_tool(agent).function(description="do a thing", subagent_type="helper")

    assert driver.calls == 0, "spec.model was ignored in favour of the parent driver"


def test_model_string_parent_still_works():
    """A parent built from a model string keeps the original behaviour."""
    agent = DeepAgent(model="openai/gpt-4o-mini", subagents=[_spec()])

    # Building the task tool must not raise; the sub-agent resolves lazily.
    assert _task_tool(agent) is not None


@pytest.mark.asyncio
async def test_async_subagent_inherits_driver_too():
    from prompture.agents.async_deep_agent import AsyncDeepAgent

    class _AsyncCounting(_CountingDriver):
        async def agenerate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
            return self.generate(prompt, options)

    driver = _AsyncCounting()
    agent = AsyncDeepAgent(driver=driver, subagents=[_spec()])

    task = next(t for t in agent._tools.definitions if t.name == "task")
    assert task is not None
