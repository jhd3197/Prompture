"""Async LM Studio driver using httpx."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..async_driver import AsyncDriver

logger = logging.getLogger(__name__)


class AsyncLMStudioDriver(AsyncDriver):
    supports_json_mode = True
    supports_vision = True

    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(
        self,
        endpoint: str | None = None,
        model: str = "deepseek/deepseek-r1-0528-qwen3-8b",
        api_key: str | None = None,
    ):
        self.endpoint = endpoint or os.getenv("LMSTUDIO_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions")
        self.model = model
        self.options: dict[str, Any] = {}

        # Derive base_url once for reuse across management endpoints
        self.base_url = self.endpoint.split("/v1/")[0]

        # API key for LM Studio 0.4.0+ authentication
        self.api_key = api_key or os.getenv("LMSTUDIO_API_KEY")
        self._headers = self._build_headers()

    supports_messages = True

    def _build_headers(self) -> dict[str, str]:
        """Build request headers, including auth if an API key is configured."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(self._prepare_messages(messages), options)

    async def _do_generate(
        self, messages: list[dict[str, str]], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        payload = {
            "model": merged_options.get("model", self.model),
            "messages": messages,
            "temperature": merged_options.get("temperature", 0.7),
        }

        # Native JSON mode support
        if merged_options.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(self.endpoint, json=payload, headers=self._headers, timeout=120)
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

    # -- Model management (LM Studio 0.4.0+) ----------------------------------

    async def list_models(self) -> list[dict[str, Any]]:
        """List currently loaded models via GET /v1/models (OpenAI-compatible)."""
        url = f"{self.base_url}/v1/models"
        async with httpx.AsyncClient() as client:
            r = await client.get(url, headers=self._headers, timeout=10)
            r.raise_for_status()
            data = r.json()
        return data.get("data", [])

    async def load_model(self, model: str, context_length: int | None = None) -> dict[str, Any]:
        """Load a model into LM Studio via POST /api/v1/models/load."""
        url = f"{self.base_url}/api/v1/models/load"
        payload: dict[str, Any] = {"model": model}
        if context_length is not None:
            payload["context_length"] = context_length
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=self._headers, timeout=120)
            r.raise_for_status()
        return r.json()

    async def unload_model(self, model: str) -> dict[str, Any]:
        """Unload a model from LM Studio via POST /api/v1/models/unload."""
        url = f"{self.base_url}/api/v1/models/unload"
        payload = {"model": model}
        async with httpx.AsyncClient() as client:
            r = await client.post(url, json=payload, headers=self._headers, timeout=30)
            r.raise_for_status()
        return r.json()
