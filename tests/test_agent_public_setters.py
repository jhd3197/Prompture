"""Public setters on Agent/AsyncAgent: driver, options, system_prompt.

Hosts (e.g. AgentSite) inject per-tenant drivers and per-agent option
overrides after construction; these used to be silent no-op attribute
writes. The properties must reach the real private state and take effect
on the next run.
"""

from typing import Any

import pytest

from prompture.agents.agent import Agent
from prompture.agents.async_agent import AsyncAgent
from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver


class RecordingDriver(Driver):
    supports_messages = True
    supports_tool_use = False

    def __init__(self, tag: str = "rec"):
        self.tag = tag
        self.model = f"mock-{tag}"
        self.seen_options: list[dict[str, Any]] = []
        self.seen_messages: list[list[dict[str, Any]]] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self.generate_messages([{"role": "user", "content": prompt}], options)

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        self.seen_options.append(dict(options))
        self.seen_messages.append(list(messages))
        return {
            "text": f"from-{self.tag}",
            "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
        }


class AsyncRecordingDriver(AsyncDriver):
    supports_messages = True
    supports_tool_use = False

    def __init__(self, tag: str = "rec"):
        self.tag = tag
        self.model = f"mock-{tag}"
        self.seen_options: list[dict[str, Any]] = []
        self.seen_messages: list[list[dict[str, Any]]] = []

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return await self.generate_messages([{"role": "user", "content": prompt}], options)

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        self.seen_options.append(dict(options))
        self.seen_messages.append(list(messages))
        return {
            "text": f"from-{self.tag}",
            "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0},
        }


def test_sync_driver_setter_swaps_backend():
    first, second = RecordingDriver("first"), RecordingDriver("second")
    agent = Agent(driver=first)
    assert agent.run("hi").output_text == "from-first"
    agent.driver = second
    assert agent.driver is second
    assert agent.run("hi").output_text == "from-second"


def test_sync_options_mutation_reaches_driver():
    drv = RecordingDriver()
    agent = Agent(driver=drv)
    agent.options["temperature"] = 0.123
    agent.run("hi")
    assert drv.seen_options[-1].get("temperature") == 0.123


def test_sync_options_setter_replaces_and_copies():
    agent = Agent(driver=RecordingDriver())
    src = {"temperature": 0.5}
    agent.options = src
    src["temperature"] = 0.9
    assert agent.options["temperature"] == 0.5
    agent.options = None
    assert agent.options == {}


def test_sync_system_prompt_setter_used_on_next_run():
    drv = RecordingDriver()
    agent = Agent(driver=drv, system_prompt="old prompt")
    agent.system_prompt = "new prompt"
    assert agent.system_prompt == "new prompt"
    agent.run("hi")
    system = [m for m in drv.seen_messages[-1] if m["role"] == "system"]
    assert system and "new prompt" in system[0]["content"]


@pytest.mark.asyncio
async def test_async_driver_setter_swaps_backend():
    first, second = AsyncRecordingDriver("first"), AsyncRecordingDriver("second")
    agent = AsyncAgent(driver=first)
    assert (await agent.run("hi")).output_text == "from-first"
    agent.driver = second
    assert (await agent.run("hi")).output_text == "from-second"


@pytest.mark.asyncio
async def test_async_driver_setter_rebuilds_persistent_conversation():
    first, second = AsyncRecordingDriver("first"), AsyncRecordingDriver("second")
    agent = AsyncAgent(driver=first, persistent_conversation=True)
    await agent.run("hi")
    agent.driver = second
    assert (await agent.run("hi")).output_text == "from-second"


@pytest.mark.asyncio
async def test_async_options_mutation_reaches_persistent_conversation():
    drv = AsyncRecordingDriver()
    agent = AsyncAgent(driver=drv, persistent_conversation=True)
    await agent.run("hi")
    agent.options["temperature"] = 0.42
    await agent.run("again")
    assert drv.seen_options[-1].get("temperature") == 0.42


@pytest.mark.asyncio
async def test_async_system_prompt_setter_used_on_next_run():
    drv = AsyncRecordingDriver()
    agent = AsyncAgent(driver=drv, system_prompt="old prompt", persistent_conversation=True)
    await agent.run("hi")
    agent.system_prompt = "new prompt"
    await agent.run("again")
    system = [m for m in drv.seen_messages[-1] if m["role"] == "system"]
    assert system and "new prompt" in system[0]["content"]
