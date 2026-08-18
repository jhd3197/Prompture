"""reasoning_content must land on assistant history messages on every path,
not only the with-tool-calls branch (hosts read it from result.messages)."""

from typing import Any

import pytest

from prompture.agents.async_conversation import AsyncConversation
from prompture.agents.conversation import Conversation
from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver

META = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0}


class ThinkingDriver(Driver):
    supports_messages = True
    supports_tool_use = False

    def __init__(self):
        self.model = "mock-think"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self.generate_messages([], options)

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "the answer", "reasoning_content": "let me think...", "meta": dict(META)}


class AsyncThinkingDriver(AsyncDriver):
    supports_messages = True
    supports_tool_use = False

    def __init__(self):
        self.model = "mock-think"

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return await self.generate_messages([], options)

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "the answer", "reasoning_content": "let me think...", "meta": dict(META)}


def test_sync_ask_attaches_reasoning_to_history():
    conv = Conversation(driver=ThinkingDriver())
    conv.ask("question")
    assistant = [m for m in conv.messages if m["role"] == "assistant"]
    assert assistant and assistant[-1].get("reasoning_content") == "let me think..."


@pytest.mark.asyncio
async def test_async_ask_attaches_reasoning_to_history():
    conv = AsyncConversation(driver=AsyncThinkingDriver())
    await conv.ask("question")
    assistant = [m for m in conv.messages if m["role"] == "assistant"]
    assert assistant and assistant[-1].get("reasoning_content") == "let me think..."


def test_sync_no_reasoning_leaves_message_bare():
    class Plain(ThinkingDriver):
        def generate_messages(self, messages, options):
            return {"text": "plain", "meta": dict(META)}

    conv = Conversation(driver=Plain())
    conv.ask("question")
    assistant = [m for m in conv.messages if m["role"] == "assistant"]
    assert assistant and "reasoning_content" not in assistant[-1]
