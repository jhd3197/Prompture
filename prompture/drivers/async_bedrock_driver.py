"""Async AWS Bedrock driver.

Uses the sync :class:`BedrockDriver` under ``asyncio.to_thread`` rather
than introducing an ``aioboto3`` dependency. boto3's invoke_model call
is short-lived and benefits little from native async I/O, and offloading
to a thread is a clean drop-in that keeps the async surface consistent
without an extra dependency.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..infra.cost_mixin import CostMixin
from .async_base import AsyncDriver
from .bedrock_driver import BedrockDriver

logger = logging.getLogger(__name__)


class AsyncBedrockDriver(CostMixin, AsyncDriver):
    """Async wrapper around :class:`BedrockDriver` using a thread executor."""

    supports_json_mode = False
    supports_json_schema = False
    supports_tool_use = True  # Anthropic only at runtime — see BedrockDriver
    supports_streaming = True
    supports_vision = True
    supports_messages = True

    MODEL_PRICING = BedrockDriver.MODEL_PRICING

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_region: str | None = None,
        model: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
    ):
        self._sync = BedrockDriver(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
            model=model,
        )
        self.model = self._sync.model

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.generate, prompt, options)

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.generate_messages, messages, options)

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync.generate_messages_with_tools, messages, tools, options)

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        # Drain the sync iterator on a worker thread, yielding chunks
        # without blocking the event loop on each event.
        loop = asyncio.get_running_loop()
        gen = await loop.run_in_executor(None, lambda: iter(self._sync.generate_messages_stream(messages, options)))
        sentinel = object()
        while True:
            chunk = await loop.run_in_executor(None, next, gen, sentinel)
            if chunk is sentinel:
                break
            yield chunk
