"""MiniMax LLM driver (OpenAI-compatible).

MiniMax exposes an OpenAI-style chat completion endpoint at
``POST /v1/text/chatcompletion_v2``. This driver supports text generation,
``generate_messages``, JSON mode, and tool use via the standard
OpenAI-compatible request shape.

Streaming and reasoning-content handling are intentionally omitted from the
initial version — they can be added later if MiniMax usage in Prompture
calls for it.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .base import Driver, _parse_tool_arguments

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://api.minimax.io/v1"


class MiniMaxDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = False
    supports_vision = False
    supports_messages = True

    MODEL_PRICING: dict[str, dict[str, Any]] = {}

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

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate([{"role": "user", "content": prompt}], options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(list(messages), options)

    def _do_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
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

        try:
            response = requests.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers=self.headers,
                json=data,
                timeout=opts.get("timeout", 300),  # nosec B113
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"MiniMax API request failed: {e!s}") from e

        base_resp = resp.get("base_resp")
        if isinstance(base_resp, dict) and base_resp.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax API error: {base_resp.get('status_msg')}")

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens") or usage.get("total_input_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("total_output_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        total_cost = self._calculate_cost("minimax", model, prompt_tokens, completion_tokens)

        choices = resp.get("choices") or []
        text = ""
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content") or ""

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

    def generate_messages_with_tools(
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

        try:
            response = requests.post(
                f"{self.base_url}/text/chatcompletion_v2",
                headers=self.headers,
                json=data,
                timeout=opts.get("timeout", 300),  # nosec B113
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"MiniMax API request failed: {e!s}") from e

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
