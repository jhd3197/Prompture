"""Async Moonshot AI (Kimi) driver using httpx.

All pricing comes from models.dev (provider: "moonshotai") — no hardcoded pricing.

Moonshot-specific constraints:
- Temperature clamped to [0, 1] (OpenAI allows [0, 2])
- tool_choice: "required" not supported — only "auto" or "none"
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin, prepare_strict_schema
from .moonshot_driver import MoonshotDriver

logger = logging.getLogger("prompture.drivers.moonshot")


class AsyncMoonshotDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = True

    MODEL_PRICING = MoonshotDriver.MODEL_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "kimi-k2-0905-preview",
        endpoint: str = "https://api.moonshot.ai/v1",
    ):
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

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(self._prepare_messages(messages), options)

    async def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
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
        MoonshotDriver._clamp_temperature(opts)

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
                error_body = MoonshotDriver._format_error_body(data, e)
                error_msg = f"Moonshot API request failed: {e!s}{error_body}"
                raise RuntimeError(error_msg) from e
            except Exception as e:
                raise RuntimeError(f"Moonshot API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        message = resp["choices"][0]["message"]
        text = message.get("content") or ""

        # Reasoning models may return content in reasoning_content when content is empty
        if not text and message.get("reasoning_content"):
            text = message["reasoning_content"]

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

            json_schema = options.get("json_schema")
            schema_instruction = (
                "Return a JSON object that validates against this schema:\n"
                f"{json.dumps(json_schema, indent=2)}\n"
                "If a value is unknown use null."
            )
            fallback_messages = list(fallback_data["messages"])
            fallback_messages.append({"role": "system", "content": schema_instruction})
            fallback_data["messages"] = fallback_messages

            async with httpx.AsyncClient() as fb_client:
                try:
                    fb_response = await fb_client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=fallback_data,
                        timeout=120,
                    )
                    fb_response.raise_for_status()
                    fb_resp = fb_response.json()
                except Exception:
                    pass  # Fallback failed — return original empty result
                else:
                    fb_usage = fb_resp.get("usage", {})
                    prompt_tokens += fb_usage.get("prompt_tokens", 0)
                    completion_tokens = fb_usage.get("completion_tokens", 0)
                    total_tokens = prompt_tokens + completion_tokens
                    resp = fb_resp
                    fb_message = fb_resp["choices"][0]["message"]
                    text = fb_message.get("content") or ""
                    if not text and fb_message.get("reasoning_content"):
                        text = fb_message["reasoning_content"]

        total_cost = self._calculate_cost("moonshot", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

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
        model_config = self._get_model_config("moonshot", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("moonshot", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}
        MoonshotDriver._clamp_temperature(opts)

        sanitized_tools = MoonshotDriver._sanitize_tools(tools)

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": sanitized_tools,
        }
        data[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        if "tool_choice" in options:
            data["tool_choice"] = options["tool_choice"]

        MoonshotDriver._sanitize_tool_choice(data)

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
                error_body = MoonshotDriver._format_error_body(data, e)
                error_msg = f"Moonshot API request failed: {e!s}{error_body}"
                raise RuntimeError(error_msg) from e
            except Exception as e:
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

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield response chunks via Moonshot streaming API."""
        model = options.get("model", self.model)
        model_config = self._get_model_config("moonshot", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}
        MoonshotDriver._clamp_temperature(opts)

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
                    content = delta.get("content") or ""
                    # Reasoning models stream thinking via reasoning_content
                    if not content:
                        content = delta.get("reasoning_content") or ""
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
