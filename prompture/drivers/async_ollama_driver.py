"""Async Ollama driver using httpx."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import httpx

from ..async_driver import AsyncDriver

logger = logging.getLogger(__name__)


class AsyncOllamaDriver(AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_vision = True

    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, endpoint: str | None = None, model: str = "llama3"):
        self.endpoint = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
        self.model = model
        self.options: dict[str, Any] = {}

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_ollama_vision_messages

        return _prepare_ollama_vision_messages(messages)

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        payload = {
            "prompt": prompt,
            "model": merged_options.get("model", self.model),
            "stream": False,
        }

        # Native JSON mode / structured output support
        if merged_options.get("json_mode"):
            json_schema = merged_options.get("json_schema")
            payload["format"] = json_schema if json_schema else "json"

        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(self.endpoint, json=payload, timeout=120)
                r.raise_for_status()
                response_data = r.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama request failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Ollama request failed: {e}") from e

        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        text = response_data.get("response", "")
        reasoning_content = response_data.get("thinking") or None

        if not text and reasoning_content:
            text = reasoning_content

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls via Ollama's /api/chat endpoint."""
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")

        payload: dict[str, Any] = {
            "model": merged_options.get("model", self.model),
            "messages": messages,
            "tools": tools,
            "stream": False,
        }

        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(chat_endpoint, json=payload, timeout=120)
                r.raise_for_status()
                response_data = r.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama tool use request failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Ollama tool use request failed: {e}") from e

        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        message = response_data.get("message", {})
        text = message.get("content") or ""
        reasoning_content = message.get("thinking") or None
        stop_reason = response_data.get("done_reason", "stop")

        if not text and reasoning_content:
            text = reasoning_content

        tool_calls_out: list[dict[str, Any]] = []
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            # Ollama returns arguments as a dict already (no JSON string parsing needed)
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            tool_calls_out.append({
                # Ollama does not return tool_call IDs — generate one locally
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "name": func.get("name", ""),
                "arguments": args,
            })

        result: dict[str, Any] = {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        """Use Ollama's /api/chat endpoint for multi-turn conversations."""
        messages = self._prepare_messages(messages)
        merged_options = self.options.copy()
        if options:
            merged_options.update(options)

        # Derive the chat endpoint from the generate endpoint
        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")

        payload: dict[str, Any] = {
            "model": merged_options.get("model", self.model),
            "messages": messages,
            "stream": False,
        }

        if merged_options.get("json_mode"):
            json_schema = merged_options.get("json_schema")
            payload["format"] = json_schema if json_schema else "json"

        if "temperature" in merged_options:
            payload["temperature"] = merged_options["temperature"]
        if "top_p" in merged_options:
            payload["top_p"] = merged_options["top_p"]
        if "top_k" in merged_options:
            payload["top_k"] = merged_options["top_k"]

        async with httpx.AsyncClient() as client:
            try:
                r = await client.post(chat_endpoint, json=payload, timeout=120)
                r.raise_for_status()
                response_data = r.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama chat request failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Ollama chat request failed: {e}") from e

        prompt_tokens = response_data.get("prompt_eval_count", 0)
        completion_tokens = response_data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": 0.0,
            "raw_response": response_data,
            "model_name": merged_options.get("model", self.model),
        }

        message = response_data.get("message", {})
        text = message.get("content", "")
        reasoning_content = message.get("thinking") or None

        if not text and reasoning_content:
            text = reasoning_content

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result
