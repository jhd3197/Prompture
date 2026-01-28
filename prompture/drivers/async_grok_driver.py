"""Async xAI Grok driver using httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin
from .grok_driver import GrokDriver


class AsyncGrokDriver(CostMixin, AsyncDriver):
    MODEL_PRICING = GrokDriver.MODEL_PRICING
    _PRICING_UNIT = 1_000_000

    def __init__(self, api_key: str | None = None, model: str = "grok-4-fast-reasoning"):
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.model = model
        self.api_base = "https://api.x.ai/v1"

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GROK_API_KEY environment variable is required")

        model = options.get("model", self.model)

        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

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

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        total_cost = self._calculate_cost("grok", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost,
            "raw_response": resp,
            "model_name": model,
        }

        text = resp["choices"][0]["message"]["content"]
        return {"text": text, "meta": meta}
