"""Async Ollama driver using httpx."""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..agents.live_events import (
    LiveEvent,
    MessageStop,
    TextDelta,
    ToolInputDelta,
    ToolUseStart,
    ToolUseStop,
)
from ._prompted_tool_stream import PromptedToolStreamMixin
from .async_base import AsyncDriver

logger = logging.getLogger(__name__)


class AsyncOllamaDriver(PromptedToolStreamMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_streaming = True
    supports_streaming_tool_use = True
    supports_tool_use = True
    supports_vision = True
    prompted_tool_grammar = "xml_tags"

    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, endpoint: str | None = None, model: str = "llama3"):
        raw_endpoint: str = endpoint or os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")  # type: ignore[assignment]
        # Auto-append /api/generate when given a bare base URL (e.g. http://localhost:11434)
        if "/api/" not in raw_endpoint:
            self.endpoint = raw_endpoint.rstrip("/") + "/api/generate"
        else:
            self.endpoint = raw_endpoint
        self.model = model
        self.options: dict[str, Any] = {}

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_ollama_vision_messages

        return _prepare_ollama_vision_messages(messages)

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        # Delegate to /api/chat — the /api/generate endpoint produces
        # poor results with instruction-tuned models and does not support
        # JSON schema objects in the ``format`` field.
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_messages(messages, options or {})

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
                    if stop_reason == "length":
                        logger.warning(
                            "Tool arguments for %s were truncated due to max_tokens limit. "
                            "Increase max_tokens in options to allow longer tool outputs. "
                            "Truncated arguments: %r",
                            func.get("name", ""),
                            args[:200] if args else args,
                        )
                    else:
                        logger.warning(
                            "Failed to parse tool arguments for %s: %r",
                            func.get("name", ""),
                            args,
                        )
                    args = {}
            tool_calls_out.append(
                {
                    # Ollama does not return tool_call IDs — generate one locally
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "name": func.get("name", ""),
                    "arguments": args,
                }
            )

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

        # Native JSON mode / structured output (Ollama >=0.5 accepts a JSON
        # schema dict directly in `format`). A schema implies JSON mode.
        json_schema = merged_options.get("json_schema")
        if json_schema is not None:
            payload["format"] = json_schema
        elif merged_options.get("json_mode"):
            payload["format"] = "json"

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

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield response chunks via Ollama streaming API."""
        merged = self.options.copy()
        if options:
            merged.update(options)

        messages = self._prepare_messages(messages)
        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")

        payload: dict[str, Any] = {
            "model": merged.get("model", self.model),
            "messages": messages,
            "stream": True,
        }
        json_schema = merged.get("json_schema")
        if json_schema is not None:
            payload["format"] = json_schema
        elif merged.get("json_mode"):
            payload["format"] = "json"
        for key in ("temperature", "top_p", "top_k"):
            if key in merged:
                payload[key] = merged[key]

        full_text = ""
        full_reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0

        async with (
            httpx.AsyncClient(timeout=merged.get("timeout", 300.0)) as client,
            client.stream("POST", chat_endpoint, json=payload) as r,
        ):
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)
                else:
                    msg = chunk.get("message", {}) or {}
                    thinking = msg.get("thinking", "")
                    if thinking:
                        full_reasoning += thinking
                    content = msg.get("content", "")
                    if content:
                        full_text += content
                        yield {"type": "delta", "text": content}

        done_chunk: dict[str, Any] = {
            "type": "done",
            "text": full_text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": 0.0,
                "raw_response": {},
                "model_name": merged.get("model", self.model),
            },
        }
        if full_reasoning:
            done_chunk["reasoning_content"] = full_reasoning
        yield done_chunk

    # ------------------------------------------------------------------
    # Live streaming with interleaved tool calls
    # ------------------------------------------------------------------

    async def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[LiveEvent]:
        """Yield :class:`~prompture.agents.live_events.LiveEvent` for one
        assistant turn that may interleave narration and tool calls.

        ``options["prompted_tools"] = True`` switches to Tier 2 emulation
        (grammar injected into system prompt; tool calls parsed out of
        the token stream). Default is Tier 1 native: Ollama
        ``/api/chat`` with ``stream=True`` and ``tools=[...]``, text
        streamed live, tool calls emitted from the final ``done`` chunk.
        """
        if options.get("prompted_tools"):
            async for ev in self._astream_via_prompted_emulation(messages, tools, options):
                yield ev
            return

        async for ev in self._astream_native_with_tools(messages, tools, options):
            yield ev

    async def _astream_native_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[LiveEvent]:
        """Tier 1: native async Ollama streaming with tools."""
        merged = self.options.copy()
        if options:
            merged.update(options)

        messages = self._prepare_messages(messages)
        chat_endpoint = self.endpoint.replace("/api/generate", "/api/chat")

        payload: dict[str, Any] = {
            "model": merged.get("model", self.model),
            "messages": messages,
            "tools": tools,
            "stream": True,
        }
        for key in ("temperature", "top_p", "top_k"):
            if key in merged:
                payload[key] = merged[key]

        prompt_tokens = 0
        completion_tokens = 0
        stop_reason = "stop"
        tool_calls_raw: list[dict[str, Any]] = []

        async with (
            httpx.AsyncClient(timeout=merged.get("timeout", 300.0)) as client,
            client.stream("POST", chat_endpoint, json=payload) as r,
        ):
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)
                    stop_reason = chunk.get("done_reason", "stop")
                    msg = chunk.get("message", {}) or {}
                    tool_calls_raw = msg.get("tool_calls", []) or []
                    tail = msg.get("content", "")
                    if tail:
                        yield TextDelta(text=tail)
                else:
                    msg = chunk.get("message", {}) or {}
                    content = msg.get("content", "")
                    if content:
                        yield TextDelta(text=content)

        for tc in tool_calls_raw:
            func = tc.get("function", {}) or {}
            name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Ollama tool args for %s: %r", name, args[:200])
                    args = {}
            if not isinstance(args, dict):
                args = {}
            tool_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
            yield ToolUseStart(id=tool_id, name=name)
            try:
                fragment = json.dumps(args)
            except (TypeError, ValueError):
                fragment = "{}"
            if fragment and fragment != "{}":
                yield ToolInputDelta(id=tool_id, fragment=fragment)
            yield ToolUseStop(id=tool_id, name=name, input=args)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": 0.0,
            "model_name": merged.get("model", self.model),
        }
        yield MessageStop(stop_reason=stop_reason, usage=usage)
