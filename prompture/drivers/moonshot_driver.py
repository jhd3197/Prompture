"""Moonshot AI (Kimi) driver implementation.
Requires the `requests` package. Uses MOONSHOT_API_KEY env var.

The Moonshot API is fully OpenAI-compatible (/v1/chat/completions).
All pricing comes from models.dev (provider: "moonshotai") — no hardcoded pricing.

Moonshot-specific constraints:
- Temperature clamped to [0, 1] (OpenAI allows [0, 2])
- tool_choice: "required" not supported — only "auto" or "none"
"""

import json
import os
from collections.abc import Iterator
from typing import Any

import requests

from ..cost_mixin import CostMixin, prepare_strict_schema
from ..driver import Driver


class MoonshotDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = True

    # All pricing resolved live from models.dev (provider: "moonshotai")
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "kimi-k2-0905-preview",
        endpoint: str = "https://api.moonshot.ai/v1",
    ):
        """Initialize Moonshot driver.

        Args:
            api_key: Moonshot API key. If not provided, will look for MOONSHOT_API_KEY env var.
            model: Model to use. Defaults to kimi-k2-0905-preview.
            endpoint: API base URL. Defaults to https://api.moonshot.ai/v1.
                      Use https://api.moonshot.cn/v1 for the China endpoint.
        """
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError("Moonshot API key not found. Set MOONSHOT_API_KEY env var.")

        self.model = model
        self.base_url = endpoint.rstrip("/")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    supports_messages = True

    @staticmethod
    def _clamp_temperature(opts: dict[str, Any]) -> dict[str, Any]:
        """Clamp temperature to Moonshot's supported range [0, 1]."""
        if "temperature" in opts:
            opts["temperature"] = max(0.0, min(1.0, float(opts["temperature"])))
        return opts

    @staticmethod
    def _sanitize_tool_choice(data: dict[str, Any]) -> dict[str, Any]:
        """Downgrade tool_choice='required' to 'auto' (unsupported by Moonshot)."""
        if data.get("tool_choice") == "required":
            data["tool_choice"] = "auto"
        return data

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
            raise RuntimeError("Moonshot API key not found")

        model = options.get("model", self.model)

        model_config = self._get_model_config("moonshot", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities(
            "moonshot",
            model,
            using_json_schema=bool(options.get("json_schema")),
        )

        opts = {"temperature": 1.0, "max_tokens": 512, **options}
        opts = self._clamp_temperature(opts)

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        # Native JSON mode support — skip for reasoning models where
        # Moonshot's API does not reliably support response_format.
        if options.get("json_mode"):
            from ..model_rates import get_model_capabilities

            caps = get_model_capabilities("moonshot", model)
            is_reasoning = caps is not None and caps.is_reasoning is True
            model_supports_structured = (
                caps is None or caps.supports_structured_output is not False
            ) and not is_reasoning

            if model_supports_structured:
                json_schema = options.get("json_schema")
                if json_schema:
                    schema_copy = prepare_strict_schema(json_schema)
                    data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "extraction",
                            "strict": True,
                            "schema": schema_copy,
                        },
                    }
                else:
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
            error_msg = f"Moonshot API request failed: {e!s}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Moonshot API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        total_cost = self._calculate_cost("moonshot", model, prompt_tokens, completion_tokens)

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

        # Reasoning models may return content in reasoning_content when content is empty
        if not text and message.get("reasoning_content"):
            text = message["reasoning_content"]

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
            raise RuntimeError("Moonshot API key not found")

        model = options.get("model", self.model)
        model_config = self._get_model_config("moonshot", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("moonshot", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}
        opts = self._clamp_temperature(opts)

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

        data = self._sanitize_tool_choice(data)

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
            error_msg = f"Moonshot API request failed: {e!s}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Moonshot API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        total_cost = self._calculate_cost("moonshot", model, prompt_tokens, completion_tokens)

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
        """Yield response chunks via Moonshot streaming API."""
        if not self.api_key:
            raise RuntimeError("Moonshot API key not found")

        model = options.get("model", self.model)
        model_config = self._get_model_config("moonshot", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}
        opts = self._clamp_temperature(opts)

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
        total_cost = self._calculate_cost("moonshot", model, prompt_tokens, completion_tokens)

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
