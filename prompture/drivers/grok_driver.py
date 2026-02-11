"""xAI Grok driver.
Requires the `requests` package. Uses GROK_API_KEY env var.
"""

import json
import logging
import os
from typing import Any

import requests

from ..infra.cost_mixin import CostMixin
from .base import Driver

logger = logging.getLogger(__name__)


class GrokDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_tool_use = True
    supports_vision = True

    # Pricing per 1M tokens based on xAI's documentation
    _PRICING_UNIT = 1_000_000
    MODEL_PRICING = {
        "grok-code-fast-1": {
            "prompt": 0.20,
            "completion": 1.50,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-4-fast-reasoning": {
            "prompt": 0.20,
            "completion": 0.50,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-4-fast-non-reasoning": {
            "prompt": 0.20,
            "completion": 0.50,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-4-0709": {
            "prompt": 3.00,
            "completion": 15.00,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-3-mini": {
            "prompt": 0.30,
            "completion": 0.50,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-3": {
            "prompt": 3.00,
            "completion": 15.00,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-2-vision-1212us-east-1": {
            "prompt": 2.00,
            "completion": 10.00,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "grok-2-vision-1212eu-west-1": {
            "prompt": 2.00,
            "completion": 10.00,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "grok-4-fast-reasoning"):
        """Initialize Grok driver.

        Args:
            api_key: xAI API key. If not provided, reads from GROK_API_KEY env var
            model: Model to use. Defaults to grok-4-fast-reasoning
        """
        self.api_key = api_key or os.getenv("GROK_API_KEY")
        self.model = model
        self.api_base = "https://api.x.ai/v1"

    @classmethod
    def list_models(cls, *, api_key: str | None = None, timeout: int = 10, **kw: object) -> list[str] | None:
        """List models available via the xAI API."""
        from .base import _fetch_openai_compatible_models

        key = api_key or os.getenv("GROK_API_KEY")
        if not key:
            return None
        return _fetch_openai_compatible_models("https://api.x.ai/v1", api_key=key, timeout=timeout)

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(self._prepare_messages(messages), options)

    def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GROK_API_KEY environment variable is required")

        model = options.get("model", self.model)

        # Lookup model-specific config (live models.dev data + hardcoded fallback)
        model_config = self._get_model_config("grok", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        # Defaults
        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        # Base request payload
        payload = {
            "model": model,
            "messages": messages,
        }

        # Add token limit with correct parameter name
        payload[tokens_param] = opts.get("max_tokens", 512)

        # Add temperature if supported
        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Grok API request failed: {e!s}") from e

        # Extract usage info
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Calculate cost via shared mixin
        total_cost = self._calculate_cost("grok", model, prompt_tokens, completion_tokens)

        # Standardized meta object
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        message = resp["choices"][0]["message"]
        text = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")

        if not text and reasoning_content:
            text = reasoning_content

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls."""
        if not self.api_key:
            raise RuntimeError("GROK_API_KEY environment variable is required")

        model = options.get("model", self.model)
        model_config = self._get_model_config("grok", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("grok", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 4096, **options}

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        payload[tokens_param] = opts.get("max_tokens", 4096)

        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        if "tool_choice" in options:
            payload["tool_choice"] = options["tool_choice"]

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        try:
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.RequestException as e:
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
                if stop_reason == "length":
                    logger.warning(
                        "Tool arguments for %s were truncated due to max_tokens limit. "
                        "Increase max_tokens in options to allow longer tool outputs. "
                        "Truncated arguments: %r",
                        tc["function"]["name"],
                        raw[:200] if raw else raw,
                    )
                else:
                    logger.warning(
                        "Failed to parse tool arguments for %s: %r",
                        tc["function"]["name"],
                        raw,
                    )
                args = {}
            tool_calls_out.append(
                {
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": args,
                }
            )

        result: dict[str, Any] = {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }
        if choice["message"].get("reasoning_content") is not None:
            result["reasoning_content"] = choice["message"]["reasoning_content"]
        return result
