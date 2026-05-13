"""Async Cohere v2 chat driver (httpx)."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import CostMixin
from .async_base import AsyncDriver
from .base import _parse_tool_arguments
from .cohere_driver import CohereDriver

logger = logging.getLogger(__name__)


class AsyncCohereDriver(CostMixin, AsyncDriver):
    """Async Cohere v2 chat driver."""

    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = False
    supports_vision = False
    supports_messages = True

    MODEL_PRICING = CohereDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "command-r-plus"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not found. Set COHERE_API_KEY env var.")
        self.model = model
        self.base_url = "https://api.cohere.com/v2"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(list(messages), options)

    async def _do_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        model_config = self._get_model_config("cohere", model)
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("cohere", model, using_json_schema=bool(options.get("json_schema")))

        opts = {"temperature": 0.3, "max_tokens": 512, **options}
        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": opts.get("max_tokens", 512),
        }
        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                data["response_format"] = {"type": "json_object", "schema": json_schema}
            else:
                data["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat",
                    headers=self.headers,
                    json=data,
                    timeout=120,
                )
                response.raise_for_status()
                resp = response.json()
            except httpx.HTTPStatusError as e:
                body = ""
                with contextlib.suppress(Exception):
                    body = e.response.text
                error_msg = f"Cohere API request failed: {e!s}"
                if body:
                    error_msg += f"\nResponse: {body}"
                raise RuntimeError(error_msg) from e
            except httpx.HTTPError as e:
                raise RuntimeError(f"Cohere API request failed: {e!s}") from e

        message = resp.get("message", {}) or {}
        text = CohereDriver._extract_text(message)

        usage = resp.get("usage", {}) or {}
        tokens = (usage.get("billed_units") or usage.get("tokens") or {}) if isinstance(usage, dict) else {}
        prompt_tokens = int(tokens.get("input_tokens", 0) or 0)
        completion_tokens = int(tokens.get("output_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("cohere", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }
        return {"text": text, "meta": meta}

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        model = options.get("model", self.model)
        model_config = self._get_model_config("cohere", model)
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("cohere", model, using_tool_use=True)

        opts = {"temperature": 0.3, "max_tokens": 4096, **options}
        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "max_tokens": opts.get("max_tokens", 4096),
        }
        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat",
                    headers=self.headers,
                    json=data,
                    timeout=120,
                )
                response.raise_for_status()
                resp = response.json()
            except httpx.HTTPStatusError as e:
                body = ""
                with contextlib.suppress(Exception):
                    body = e.response.text
                error_msg = f"Cohere API request failed: {e!s}"
                if body:
                    error_msg += f"\nResponse: {body}"
                raise RuntimeError(error_msg) from e
            except httpx.HTTPError as e:
                raise RuntimeError(f"Cohere API request failed: {e!s}") from e

        message = resp.get("message", {}) or {}
        text = CohereDriver._extract_text(message)
        stop_reason = resp.get("finish_reason")

        tool_calls_out: list[dict[str, Any]] = []
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            args = _parse_tool_arguments(raw_args, name, stop_reason)
            tool_calls_out.append({"id": tc.get("id", ""), "name": name, "arguments": args})

        usage = resp.get("usage", {}) or {}
        tokens = (usage.get("billed_units") or usage.get("tokens") or {}) if isinstance(usage, dict) else {}
        prompt_tokens = int(tokens.get("input_tokens", 0) or 0)
        completion_tokens = int(tokens.get("output_tokens", 0) or 0)
        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("cohere", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }
        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }
