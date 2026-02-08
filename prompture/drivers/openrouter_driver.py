"""OpenRouter driver implementation.
Requires the `requests` package. Uses OPENROUTER_API_KEY env var.
"""

import contextlib
import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import requests

from ..cost_mixin import CostMixin, prepare_strict_schema
from ..driver import Driver

logger = logging.getLogger(__name__)


class OpenRouterDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = True

    # Approximate pricing per 1K tokens based on OpenRouter's pricing
    # https://openrouter.ai/docs#pricing
    MODEL_PRICING = {
        "openai/gpt-4o": {
            "prompt": 0.005,
            "completion": 0.015,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "openai/gpt-4o-mini": {
            "prompt": 0.00015,
            "completion": 0.0006,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "anthropic/claude-sonnet-4-20250514": {
            "prompt": 0.003,
            "completion": 0.015,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "google/gemini-2.0-flash-001": {
            "prompt": 0.0001,
            "completion": 0.0004,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
        "meta-llama/llama-3.1-70b-instruct": {
            "prompt": 0.0004,
            "completion": 0.0004,
            "tokens_param": "max_tokens",
            "supports_temperature": True,
        },
    }

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-4o-mini"):
        """Initialize OpenRouter driver.

        Args:
            api_key: OpenRouter API key. If not provided, will look for OPENROUTER_API_KEY env var
            model: Model to use. Defaults to openai/gpt-4o-mini
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY env var.")

        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

        # Required headers for OpenRouter
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/jhd3197/prompture",  # Required by OpenRouter
            "Content-Type": "application/json",
        }

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
            raise RuntimeError("OpenRouter API key not found")

        model = options.get("model", self.model)

        # Lookup model-specific config (live models.dev data + hardcoded fallback)
        model_config = self._get_model_config("openrouter", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        # Validate capabilities against models.dev metadata
        self._validate_model_capabilities(
            "openrouter",
            model,
            using_json_schema=bool(options.get("json_schema")),
        )

        # Defaults
        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        # Base request data
        data = {
            "model": model,
            "messages": messages,
        }

        # Add token limit with correct parameter name
        data[tokens_param] = opts.get("max_tokens", 512)

        # Only include temperature if model supports it
        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
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
            body = ""
            if e.response is not None:
                with contextlib.suppress(Exception):
                    body = e.response.text
            error_msg = f"OpenRouter API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter API request failed: {e!s}") from e

        # Extract usage info
        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Calculate cost via shared mixin
        total_cost = self._calculate_cost("openrouter", model, prompt_tokens, completion_tokens)

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

        # Reasoning models may return content in reasoning_content when content is empty
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
            raise RuntimeError("OpenRouter API key not found")

        model = options.get("model", self.model)
        model_config = self._get_model_config("openrouter", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("openrouter", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

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
            error_msg = f"OpenRouter API request failed: {e!s}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        total_cost = self._calculate_cost("openrouter", model, prompt_tokens, completion_tokens)

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

        result: dict[str, Any] = {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }
        if choice["message"].get("reasoning_content") is not None:
            result["reasoning_content"] = choice["message"]["reasoning_content"]
        return result

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks via OpenRouter streaming API."""
        if not self.api_key:
            raise RuntimeError("OpenRouter API key not found")

        model = options.get("model", self.model)
        model_config = self._get_model_config("openrouter", model)
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
        full_reasoning = ""
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

            # Usage comes in the final chunk
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                reasoning_chunk = delta.get("reasoning_content") or ""
                if reasoning_chunk:
                    full_reasoning += reasoning_chunk
                if not content and reasoning_chunk:
                    content = reasoning_chunk
                if content:
                    full_text += content
                    yield {"type": "delta", "text": content}

        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("openrouter", model, prompt_tokens, completion_tokens)

        done_chunk: dict[str, Any] = {
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
        if full_reasoning:
            done_chunk["reasoning_content"] = full_reasoning
        yield done_chunk
