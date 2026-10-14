"""Async ``claude-code/<model>`` and ``codex/<model>``: the sync drivers run in a worker thread."""

from __future__ import annotations

import asyncio
from typing import Any

from .async_base import AsyncDriver
from .coding_agent_driver import DEFAULT_MODEL, ClaudeCodeDriver, CodexDriver, CodingAgentDriver


class _AsyncCodingAgentDriver(AsyncDriver):
    sync_class: type[CodingAgentDriver] = CodingAgentDriver
    supports_messages = True
    supports_json_mode = False
    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, model: str = DEFAULT_MODEL, **kwargs: Any) -> None:
        self.model = model or DEFAULT_MODEL
        self._sync = self.sync_class(model=self.model, **kwargs)

    @classmethod
    def list_models(cls, **kwargs: Any) -> list[str] | None:
        return cls.sync_class.list_models(**kwargs)

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.generate, prompt, options)

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.generate_messages, messages, options)


class AsyncClaudeCodeDriver(_AsyncCodingAgentDriver):
    sync_class = ClaudeCodeDriver


class AsyncCodexDriver(_AsyncCodingAgentDriver):
    sync_class = CodexDriver
