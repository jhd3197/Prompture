"""Async xAI Grok driver using httpx."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import CostMixin
from ._effort import apply_effort
from ._openai_chat import AsyncOpenAIChatTools, AsyncOpenAIChatToolStream, get_chat_profile
from .async_base import AsyncDriver
from .grok_driver import GrokDriver

logger = logging.getLogger(__name__)


class AsyncGrokDriver(AsyncOpenAIChatTools, AsyncOpenAIChatToolStream, CostMixin, AsyncDriver):
    chat_profile = get_chat_profile("grok")

    supports_json_mode = True
    supports_tool_use = True
    supports_streaming_tool_use = True

    supports_vision = True

    MODEL_PRICING = GrokDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "grok-4-fast-reasoning"):
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.model = model
        self.api_base = "https://api.x.ai/v1"

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(self._prepare_messages(messages), options)

    async def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GROK_API_KEY environment variable is required")

        model = options.get("model", self.model)

        model_config = self._get_model_config("grok", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        payload = {
            "model": model,
            "messages": messages,
        }
        payload[tokens_param] = opts.get("max_tokens", 512)
        apply_effort(payload, opts, "openai")

        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.api_base}/chat/completions", headers=headers, json=payload, timeout=120
                )
                response.raise_for_status()
                resp = response.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Grok API request failed: {e!s}") from e
            except Exception as e:
                raise RuntimeError(f"Grok API request failed: {e!s}") from e

        from .moonshot_driver import _extract_moonshot_cached_tokens

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        cached_prompt_tokens = _extract_moonshot_cached_tokens(usage)

        total_cost = self._calculate_cost(
            "grok",
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens=cached_prompt_tokens,
        )

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        message = resp["choices"][0]["message"]
        text = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")

        if not text and reasoning_content:
            text = reasoning_content

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result
