"""Async MiniMax LLM driver."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .async_base import AsyncDriver
from .base import _parse_tool_arguments
from .minimax_driver import _DEFAULT_ENDPOINT, MiniMaxDriver

logger = logging.getLogger(__name__)


class AsyncMiniMaxDriver(CostMixin, AsyncDriver):
    supports_json_mode = MiniMaxDriver.supports_json_mode
    supports_json_schema = MiniMaxDriver.supports_json_schema
    supports_tool_use = MiniMaxDriver.supports_tool_use
    supports_streaming = False
    supports_vision = False
    supports_messages = True

    MODEL_PRICING = MiniMaxDriver.MODEL_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "MiniMax-Text-01",
        endpoint: str | None = None,
    ):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY") or os.getenv("HAILUO_API_KEY")
        if not self.api_key:
            raise ValueError("MiniMax API key not found. Set MINIMAX_API_KEY env var.")
        self.model = model
        self.base_url = (endpoint or os.getenv("MINIMAX_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate([{"role": "user", "content": prompt}], options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(list(messages), options)

    async def _do_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        opts = {"temperature": 1.0, "max_tokens": 1024, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": opts["max_tokens"],
            "temperature": opts["temperature"],
        }
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": prepare_strict_schema(json_schema),
                    },
                }
            else:
                data["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=opts.get("timeout", 300)) as client:
            response = await client.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers=self.headers,
                json=data,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"MiniMax API error {response.status_code}: {response.text}")
            resp = response.json()

        base_resp = resp.get("base_resp")
        if isinstance(base_resp, dict) and base_resp.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax API error: {base_resp.get('status_msg')}")

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") or usage.get("total_input_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("total_output_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        total_cost = self._calculate_cost("minimax", model, prompt_tokens, completion_tokens)

        text = ""
        choices = resp.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""

        return {
            "text": text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": round(total_cost, 6),
                "raw_response": resp,
                "model_name": model,
            },
        }

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        model = options.get("model", self.model)
        opts = {"temperature": 1.0, "max_tokens": 4096, **options}
        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": opts["max_tokens"],
            "temperature": opts["temperature"],
        }
        if "tool_choice" in options:
            data["tool_choice"] = options["tool_choice"]

        async with httpx.AsyncClient(timeout=opts.get("timeout", 300)) as client:
            response = await client.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers=self.headers,
                json=data,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"MiniMax API error {response.status_code}: {response.text}")
            resp = response.json()

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        total_cost = self._calculate_cost("minimax", model, prompt_tokens, completion_tokens)

        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        stop_reason = choice.get("finish_reason")

        tool_calls_out: list[dict[str, Any]] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = _parse_tool_arguments(fn.get("arguments"), fn.get("name", ""), stop_reason)
            tool_calls_out.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})

        return {
            "text": text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": round(total_cost, 6),
                "raw_response": resp,
                "model_name": model,
            },
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }
