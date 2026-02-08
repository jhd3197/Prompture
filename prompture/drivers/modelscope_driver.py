"""ModelScope (Alibaba Cloud) driver implementation.
Requires the `requests` package. Uses MODELSCOPE_API_KEY env var.

The ModelScope API-Inference endpoint is fully OpenAI-compatible (/v1/chat/completions).
No hardcoded pricing — ModelScope's free tier has no per-token cost.

Model IDs are namespace-prefixed (e.g. Qwen/Qwen3-235B-A22B-Instruct-2507).
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import requests

from ..infra.cost_mixin import CostMixin
from .base import Driver

logger = logging.getLogger(__name__)


class ModelScopeDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = False
    supports_tool_use = True
    supports_streaming = True
    supports_vision = False

    # No pricing data available — ModelScope free tier has no per-token cost
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
        endpoint: str = "https://api-inference.modelscope.cn/v1",
    ):
        """Initialize ModelScope driver.

        Args:
            api_key: ModelScope API key. If not provided, will look for MODELSCOPE_API_KEY env var.
            model: Model to use. Defaults to Qwen/Qwen3-235B-A22B-Instruct-2507.
            endpoint: API base URL. Defaults to https://api-inference.modelscope.cn/v1.
        """
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

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(messages, options)

    def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("ModelScope API key not found")

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

        # Native JSON mode support
        if options.get("json_mode"):
            data["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"ModelScope API request failed: {e!s}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
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

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls."""
        if not self.api_key:
            raise RuntimeError("ModelScope API key not found")

        model = options.get("model", self.model)
        model_config = self._get_model_config("modelscope", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("modelscope", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 4096, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        data[tokens_param] = opts.get("max_tokens", 4096)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        if "tool_choice" in options:
            data["tool_choice"] = options["tool_choice"]

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"ModelScope API request failed: {e!s}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
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
                if stop_reason == "length":
                    logger.warning(
                        "Tool arguments for %s were truncated due to max_tokens limit. "
                        "Increase max_tokens in options to allow longer tool outputs. "
                        "Truncated arguments: %r",
                        tc["function"]["name"], raw[:200] if raw else raw,
                    )
                else:
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

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks via ModelScope streaming API."""
        if not self.api_key:
            raise RuntimeError("ModelScope API key not found")

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

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        for line in response.iter_lines(decode_unicode=True):
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
