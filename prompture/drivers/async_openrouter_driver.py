"""Async OpenRouter driver using httpx."""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin
from .openrouter_driver import OpenRouterDriver


class AsyncOpenRouterDriver(CostMixin, AsyncDriver):
    supports_json_mode = True

    MODEL_PRICING = OpenRouterDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY env var.")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/jhd3197/prompture",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)

        model_info = self.MODEL_PRICING.get(model, {})
        tokens_param = model_info.get("tokens_param", "max_tokens")
        supports_temperature = model_info.get("supports_temperature", True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
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
                error_msg = f"OpenRouter API request failed: {e!s}"
                raise RuntimeError(error_msg) from e
            except Exception as e:
                raise RuntimeError(f"OpenRouter API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        total_cost = self._calculate_cost("openrouter", model, prompt_tokens, completion_tokens)

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
