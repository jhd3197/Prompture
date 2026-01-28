"""Async LM Studio driver using httpx."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..async_driver import AsyncDriver

logger = logging.getLogger(__name__)


class AsyncLMStudioDriver(AsyncDriver):
    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, endpoint: str | None = None, model: str = "deepseek/deepseek-r1-0528-qwen3-8b"):
        self.endpoint = endpoint or os.getenv("LMSTUDIO_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions")
        self.model = model
        self.options: dict[str, Any] = {}

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        payload = {
            "model": merged_options.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": merged_options.get("temperature", 0.7),
        }

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(self.endpoint, json=payload, timeout=120)
                r.raise_for_status()
                response_data = r.json()
            except Exception as e:
                raise RuntimeError(f"AsyncLMStudioDriver request failed: {e}") from e

        if "choices" not in response_data or not response_data["choices"]:
            raise ValueError(f"Unexpected response format: {response_data}")

        text = response_data["choices"][0]["message"]["content"]

        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        return {"text": text, "meta": meta}
