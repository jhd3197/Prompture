"""Async ModelScope (Alibaba Cloud) driver using httpx.

No hardcoded pricing — ModelScope's free tier has no per-token cost.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin
from .modelscope_driver import ModelScopeDriver

logger = logging.getLogger(__name__)


class AsyncModelScopeDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = False
    supports_tool_use = True
    supports_streaming = True
    supports_vision = False

    MODEL_PRICING = ModelScopeDriver.MODEL_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        endpoint: str = "https://api-inference.modelscope.cn/v1",
    ):
        self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("ModelScope API key not found. Set MODELSCOPE_API_KEY env var.")
        self.model = model
        self.base_url = endpoint.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    supports_messages = True

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(messages, options)

    async def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)

        model_config = self._get_model_config("modelscope", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities(
            "modelscope",
            model,
            using_json_schema=bool(options.get("json_schema")),
        )

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

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
                error_msg = f"ModelScope API request failed: {e!s}"
                raise RuntimeError(error_msg) from e
            except Exception as e:
                raise RuntimeError(f"ModelScope API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        total_cost = self._calculate_cost("modelscope", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        text = resp["choices"][0]["message"]["content"]
        return {"text": text, "meta": meta}

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls."""
        model = options.get("model", self.model)
        model_config = self._get_model_config("modelscope", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("modelscope", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        if "tool_choice" in options:
            data["tool_choice"] = options["tool_choice"]

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
                error_msg = f"ModelScope API request failed: {e!s}"
                raise RuntimeError(error_msg) from e
            except Exception as e:
                raise RuntimeError(f"ModelScope API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        total_cost = self._calculate_cost("modelscope", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        choice = resp["choices"][0]
        text = choice["message"].get("content") or ""
        stop_reason = choice.get("finish_reason")

        tool_calls_out: list[dict[str, Any]] = []
        for tc in choice["message"].get("tool_calls", []):
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                raw = tc["function"].get("arguments")
                logger.warning(
                    "Failed to parse tool arguments for %s: %r",
                    tc["function"]["name"], raw,
                )
                args = {}
            tool_calls_out.append(
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": args,
                }
            )

        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield response chunks via ModelScope streaming API."""
        model = options.get("model", self.model)
        model_config = self._get_model_config("modelscope", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=120,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_text += content
                        yield {"type": "delta", "text": content}

        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("modelscope", model, prompt_tokens, completion_tokens)

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
