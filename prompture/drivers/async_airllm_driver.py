"""Async AirLLM driver — wraps the sync GPU-bound driver with asyncio.to_thread."""

from __future__ import annotations

import asyncio
from typing import Any

from .airllm_driver import AirLLMDriver
from .async_base import AsyncDriver


class AsyncAirLLMDriver(AsyncDriver):
    """Async wrapper around :class:`AirLLMDriver`.

    AirLLM is GPU-bound with no native async API, so we delegate to
    ``asyncio.to_thread()`` to avoid blocking the event loop.
    """

    MODEL_PRICING = AirLLMDriver.MODEL_PRICING

    def __init__(self, model: str = "meta-llama/Llama-2-7b-hf", compression: str | None = None):
        self.model = model
        self._sync_driver = AirLLMDriver(model=model, compression=compression)

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync_driver.generate, prompt, options)
