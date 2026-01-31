"""Async Groq driver. Requires the ``groq`` package."""

from __future__ import annotations

import os
from typing import Any

try:
    import groq
except Exception:
    groq = None

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin
from .groq_driver import GroqDriver


class AsyncGroqDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_vision = True

    MODEL_PRICING = GroqDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "llama2-70b-4096"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        if groq:
            self.client = groq.AsyncClient(api_key=self.api_key)
        else:
            self.client = None

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
        if self.client is None:
            raise RuntimeError("groq package is not installed")

        model = options.get("model", self.model)

        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        opts = {"temperature": 0.7, "max_tokens": 512, **options}

        kwargs = {
            "model": model,
            "messages": messages,
        }
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self.client.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        total_cost = self._calculate_cost("groq", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp.model_dump(),
            "model_name": model,
        }

        text = resp.choices[0].message.content
        return {"text": text, "meta": meta}
