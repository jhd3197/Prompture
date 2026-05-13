"""Async DeepSeek driver using httpx."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .async_base import AsyncDriver
from .deepseek_driver import DeepSeekDriver, _is_reasoner

logger = logging.getLogger(__name__)


class AsyncDeepSeekDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = False
    supports_streaming = False
    supports_messages = True

    MODEL_PRICING = DeepSeekDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY env var.")
        self.model = model
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(messages, options)

    async def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        model_config = self._get_model_config("deepseek", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("deepseek", model, using_json_schema=bool(options.get("json_schema")))

        opts = {"temperature": 0.7, "max_tokens": 512, **options}
        data: dict[str, Any] = {"model": model, "messages": messages}
        data[tokens_param] = opts.get("max_tokens", 512)
        if supports_temperature and "temperature" in opts and not _is_reasoner(model):
            data["temperature"] = opts["temperature"]

        if options.get("json_mode") and not _is_reasoner(model):
            json_schema = options.get("json_schema")
            if json_schema:
                schema_copy = prepare_strict_schema(json_schema)
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": schema_copy,
                    },
                }
            else:
                data["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=120,
                )
                response.raise_for_status()
                resp = response.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"DeepSeek API request failed: {e!s}") from e
            except Exception as e:
                raise RuntimeError(f"DeepSeek API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        total_cost = self._calculate_cost(
            "deepseek",
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens=cached_tokens,
        )

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
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
