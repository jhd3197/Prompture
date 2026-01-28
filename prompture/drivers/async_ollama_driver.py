"""Async Ollama driver using httpx."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..async_driver import AsyncDriver

logger = logging.getLogger(__name__)


class AsyncOllamaDriver(AsyncDriver):
    supports_json_mode = True

    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, endpoint: str | None = None, model: str = "llama3"):
        self.endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
        self.model = model
        self.options: dict[str, Any] = {}

    supports_messages = True

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        payload = {
            "prompt": prompt,
            "model": merged_options.get("model", self.model),
            "stream": False,
        }

        # Native JSON mode support
        if merged_options.get("json_mode"):
            payload["format"] = "json"

        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(self.endpoint, json=payload, timeout=120)
                r.raise_for_status()
                response_data = r.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama request failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Ollama request failed: {e}") from e

        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        return {"text": response_data.get("response", ""), "meta": meta}

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        """Use Ollama's /api/chat endpoint for multi-turn conversations."""
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        # Derive the chat endpoint from the generate endpoint
        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")

        payload: dict[str, Any] = {
            "model": merged_options.get("model", self.model),
            "messages": messages,
            "stream": False,
        }

        if merged_options.get("json_mode"):
            payload["format"] = "json"

        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(chat_endpoint, json=payload, timeout=120)
                r.raise_for_status()
                response_data = r.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama chat request failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Ollama chat request failed: {e}") from e

        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        message = response_data.get("message", {})
        text = message.get("content", "")
        return {"text": text, "meta": meta}
