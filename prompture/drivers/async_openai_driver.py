"""Async OpenAI driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import os
from typing import Any

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin
from .openai_driver import OpenAIDriver


class AsyncOpenAIDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True

    MODEL_PRICING = OpenAIDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if AsyncOpenAI:
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None

    supports_messages = True

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(messages, options)

    async def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) is not installed")

        model = options.get("model", self.model)

        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        kwargs = {
            "model": model,
            "messages": messages,
        }
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": json_schema,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}

        resp = await self.client.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        total_cost = self._calculate_cost("openai", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost,
            "raw_response": resp.model_dump(),
            "model_name": model,
        }

        text = resp.choices[0].message.content
        return {"text": text, "meta": meta}
