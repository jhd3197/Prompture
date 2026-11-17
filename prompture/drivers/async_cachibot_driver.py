"""Async CachiBot.ai proxy driver using httpx.

Routes requests through the CachiBot hosted API (OpenAI-compatible).
Uses CACHIBOT_API_KEY env var for authentication.

Model IDs are presented without the upstream provider prefix (e.g. ``gpt-4o``
instead of ``openai/gpt-4o``).  The driver uses a mapping populated by
``list_models()`` to reconstruct the full API model ID for requests.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..infra.cost_mixin import CostMixin
from ._openai_chat import AsyncOpenAIChatTools, AsyncOpenAIChatToolStream, ChatRoute, get_chat_profile
from .async_base import AsyncDriver
from .cachibot_driver import CachiBotDriver, _resolve_model

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://cachibot.ai/api/v1"


class AsyncCachiBotDriver(AsyncOpenAIChatTools, AsyncOpenAIChatToolStream, CostMixin, AsyncDriver):
    chat_profile = get_chat_profile("cachibot")

    def _chat_route(self, model: str, options: dict[str, Any]) -> ChatRoute:
        api_model, upstream_provider, upstream_model = _resolve_model(model)
        return ChatRoute(api_model, upstream_provider, upstream_model)

    supports_json_mode = True
    supports_tool_use = True
    supports_streaming_tool_use = True
    supports_streaming = True
    supports_vision = True

    MODEL_PRICING = CachiBotDriver.MODEL_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4o-mini",
        endpoint: str = _DEFAULT_ENDPOINT,
    ):
        self.api_key = api_key or os.getenv("CACHIBOT_API_KEY")
        self.model = model
        self.base_url = endpoint.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(self._prepare_messages(messages), options)

    async def _do_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("CACHIBOT_API_KEY is required")

        model = options.get("model", self.model)
        api_model, upstream_provider, upstream_model = _resolve_model(model)

        model_config = self._get_model_config(upstream_provider, upstream_model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        payload: dict[str, Any] = {
            "model": api_model,
            "messages": messages,
        }
        payload[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        if options.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        total_cost = self._calculate_cost(upstream_provider, upstream_model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": data,
            "model_name": model,
        }

        text = data["choices"][0]["message"].get("content") or ""
        return {"text": text, "meta": meta}

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield response chunks via CachiBot streaming API (SSE)."""
        messages = self._prepare_messages(messages)
        if not self.api_key:
            raise RuntimeError("CACHIBOT_API_KEY is required")

        model = options.get("model", self.model)
        api_model, upstream_provider, upstream_model = _resolve_model(model)

        model_config = self._get_model_config(upstream_provider, upstream_model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        payload: dict[str, Any] = {
            "model": api_model,
            "messages": messages,
            "stream": True,
        }
        payload[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[len("data: ") :]
                if payload_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or ""
                    if content:
                        full_text += content
                        yield {"type": "delta", "text": content}

        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost(upstream_provider, upstream_model, prompt_tokens, completion_tokens)

        yield {
            "type": "done",
            "text": full_text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": round(total_cost, 6),
                "raw_response": {},
                "model_name": model,
            },
        }
