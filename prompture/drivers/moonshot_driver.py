"""Moonshot AI (Kimi) driver implementation.
Requires the `requests` package. Uses MOONSHOT_API_KEY env var.

The Moonshot API is fully OpenAI-compatible (/v1/chat/completions).
All pricing comes from models.dev (provider: "moonshotai") — no hardcoded pricing.

Moonshot-specific constraints:
- Temperature clamped to [0, 1] (OpenAI allows [0, 2])
- tool_choice: "required" not supported — only "auto" or "none"
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import requests

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .base import Driver, _parse_tool_arguments

logger = logging.getLogger("prompture.drivers.moonshot")


class MoonshotDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = True

    # Fallback pricing (per 1K tokens) when models.dev data is missing/zero.
    # Source: https://openrouter.ai/moonshotai/ (Jan 2026)
    MODEL_PRICING: dict[str, dict[str, Any]] = {
        "kimi-k2.5": {"prompt": 0.0005, "completion": 0.0028},
        "kimi-k2-0905-preview": {"prompt": 0.0006, "completion": 0.0025},
        "kimi-k2-0711-preview": {"prompt": 0.0006, "completion": 0.0025},
        "kimi-k2-thinking": {"prompt": 0.0006, "completion": 0.0025},
        "kimi-k2-thinking-turbo": {"prompt": 0.00115, "completion": 0.008},
        "kimi-k2-turbo-preview": {"prompt": 0.0024, "completion": 0.01},
    }

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

    @classmethod
    def list_models(
        cls,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: int = 10,
        **kw: object,
    ) -> list[str] | None:
        """List models available via the Moonshot API (OpenAI-compatible)."""
        from .base import _fetch_openai_compatible_models

        key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not key:
            return None
        base = (endpoint or os.getenv("MOONSHOT_ENDPOINT") or "https://api.moonshot.ai/v1").rstrip("/")
        return _fetch_openai_compatible_models(base, api_key=key, timeout=timeout)

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

    @staticmethod
    def _sanitize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sanitize tool definitions for Moonshot API compatibility.

        Fixes common issues that cause 400 errors:
        - Ensures every function has a non-empty description.
        - Removes empty ``required`` arrays from parameter schemas.
        - Ensures parameter descriptions are meaningful (not just 'Parameter: x').
        """
        sanitized = []
        for tool in tools:
            tool = json.loads(json.dumps(tool))  # deep copy
            func = tool.get("function", {})

            # Ensure description is non-empty
            if not func.get("description"):
                func["description"] = f"Call the {func.get('name', 'unknown')} function"

            params = func.get("parameters", {})

            # Remove empty required arrays — Moonshot rejects these
            if "required" in params and not params["required"]:
                del params["required"]

            # Ensure properties exist even if empty
            if "properties" not in params:
                params["properties"] = {}

            sanitized.append(tool)
        return sanitized

    @staticmethod
    def _format_error_body(data: dict[str, Any], error: Exception) -> str:
        """Build a redacted request body excerpt for error diagnostics.

        Omits message content (can be large) and the Authorization header.
        Includes model, tools, response_format, and tool_choice if present.
        """
        excerpt: dict[str, Any] = {}
        for key in ("model", "response_format", "tool_choice"):
            if key in data:
                excerpt[key] = data[key]
        if "tools" in data:
            # Show tool names/parameter keys only, not full schemas
            excerpt["tools"] = [
                {
                    "name": t.get("function", {}).get("name"),
                    "params": list(t.get("function", {}).get("parameters", {}).get("properties", {}).keys()),
                }
                for t in data.get("tools", [])
            ]
        # Include response body from the error if available
        response_text = ""
        resp = getattr(error, "response", None)
        if resp is not None:
            import contextlib

            with contextlib.suppress(Exception):
                response_text = resp.text[:500] if hasattr(resp, "text") else ""
        parts = [f"\nRequest excerpt: {json.dumps(excerpt, default=str)}"]
        if response_text:
            parts.append(f"Response body: {response_text}")
        return "\n".join(parts)

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

        opts = {"temperature": 1.0, "max_tokens": 512, "timeout": 300, **options}
        opts = self._clamp_temperature(opts)

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        # Native JSON mode support — disable thinking for reasoning models
        # so the response goes to content (not reasoning_content) and
        # response_format works.  Users can override via options["thinking"].
        if options.get("json_mode"):
            from ..infra.model_rates import get_model_capabilities

            caps = get_model_capabilities("moonshot", model)
            is_reasoning = caps is not None and caps.is_reasoning is True

            if is_reasoning:
                data["thinking"] = options.get("thinking", {"type": "disabled"})

            thinking_active = is_reasoning and data.get("thinking", {}).get("type") != "disabled"
            model_supports_structured = (
                caps is None or caps.supports_structured_output is not False
            ) and not thinking_active

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
                timeout=opts.get("timeout", 300),  # nosec B113
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            error_body = self._format_error_body(data, e)
            error_msg = f"Moonshot API request failed: {e!s}{error_body}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Moonshot API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        message = resp["choices"][0]["message"]
        text = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")

        # Reasoning models may return content in reasoning_content when content is empty
        if not text and reasoning_content:
            text = reasoning_content

        # Structured output fallback: if we used json_schema mode and got an
        # empty response, retry with json_object mode and schema in the prompt.
        used_strict = data.get("response_format", {}).get("type") == "json_schema"
        if used_strict and not text.strip() and completion_tokens == 0:
            logger.info(
                "Moonshot returned empty response with json_schema mode for %s; retrying with json_object fallback",
                model,
            )
            fallback_data = {k: v for k, v in data.items() if k != "response_format"}
            fallback_data["response_format"] = {"type": "json_object"}

            # Inject schema instructions into messages
            json_schema = options.get("json_schema")
            schema_instruction = (
                "Return a JSON object that validates against this schema:\n"
                f"{json.dumps(json_schema, indent=2)}\n"
                "If a value is unknown use null."
            )
            fallback_messages = list(fallback_data["messages"])
            fallback_messages.append({"role": "system", "content": schema_instruction})
            fallback_data["messages"] = fallback_messages

            try:
                fb_response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=fallback_data,
                    timeout=opts.get("timeout", 300),  # nosec B113
                )
                fb_response.raise_for_status()
                fb_resp = fb_response.json()
            except requests.exceptions.RequestException:
                # Fallback failed — return original empty result
                pass
            else:
                fb_usage = fb_resp.get("usage", {})
                prompt_tokens += fb_usage.get("prompt_tokens", 0)
                completion_tokens = fb_usage.get("completion_tokens", 0)
                total_tokens = prompt_tokens + completion_tokens
                resp = fb_resp
                fb_message = fb_resp["choices"][0]["message"]
                text = fb_message.get("content") or ""
                reasoning_content = fb_message.get("reasoning_content")
                if not text and reasoning_content:
                    text = reasoning_content

        total_cost = self._calculate_cost("moonshot", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

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
            raise RuntimeError("Moonshot API key not found")

        model = options.get("model", self.model)
        model_config = self._get_model_config("moonshot", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("moonshot", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 4096, "timeout": 300, **options}
        opts = self._clamp_temperature(opts)

        sanitized_tools = self._sanitize_tools(tools)

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": sanitized_tools,
        }
        data[tokens_param] = opts.get("max_tokens", 4096)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        if "tool_choice" in options:
            data["tool_choice"] = options["tool_choice"]

        data = self._sanitize_tool_choice(data)

        # Disable thinking for reasoning models during tool use so
        # structured output lands in content, not reasoning_content.
        from ..infra.model_rates import get_model_capabilities

        caps = get_model_capabilities("moonshot", model)
        is_reasoning = caps is not None and caps.is_reasoning is True
        if is_reasoning:
            data["thinking"] = options.get("thinking", {"type": "disabled"})

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=opts.get("timeout", 300),  # nosec B113
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            error_body = self._format_error_body(data, e)
            error_msg = f"Moonshot API request failed: {e!s}{error_body}"
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
        message = choice["message"]
        text = message.get("content") or ""
        stop_reason = choice.get("finish_reason")

        tool_calls_out: list[dict[str, Any]] = []
        for tc in message.get("tool_calls", []):
            args = _parse_tool_arguments(
                tc["function"]["arguments"], tc["function"]["name"], stop_reason
            )
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

        # Preserve reasoning_content for reasoning models so the
        # conversation loop can include it when sending the assistant
        # message back (Moonshot requires it on subsequent requests).
        if message.get("reasoning_content") is not None:
            result["reasoning_content"] = message["reasoning_content"]

        return result

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

        opts = {"temperature": 1.0, "max_tokens": 512, "timeout": 300, **options}
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

        # Disable thinking for reasoning models during streaming with
        # json_mode so the response goes to content, not reasoning_content.
        if options.get("json_mode"):
            from ..infra.model_rates import get_model_capabilities

            caps = get_model_capabilities("moonshot", model)
            is_reasoning = caps is not None and caps.is_reasoning is True
            if is_reasoning:
                data["thinking"] = options.get("thinking", {"type": "disabled"})

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            stream=True,
            timeout=opts.get("timeout", 300),  # nosec B113
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

            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                reasoning_chunk = delta.get("reasoning_content") or ""
                if reasoning_chunk:
                    full_reasoning += reasoning_chunk
                if not content and reasoning_chunk:
                    content = reasoning_chunk
                if content:
                    full_text += content
                    yield {"type": "delta", "text": content}

        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("moonshot", model, prompt_tokens, completion_tokens)

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
