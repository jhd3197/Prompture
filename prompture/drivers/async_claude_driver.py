"""Async Anthropic Claude driver. Requires the ``anthropic`` package."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

try:
    import anthropic
except Exception:
    anthropic = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from .async_base import AsyncDriver
from .claude_driver import (
    ClaudeDriver,
    _build_anthropic_json_mode_tool_def,
    _build_anthropic_meta,
    _build_anthropic_stream_done,
    _convert_tools_to_anthropic,
    _extract_anthropic_system_and_messages,
    _extract_anthropic_text_and_tool_calls,
)


class AsyncClaudeDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = True

    MODEL_PRICING = ClaudeDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL_NAME", "claude-haiku-4-5-20251001")

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_claude_vision_messages

        return _prepare_claude_vision_messages(messages)

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(self._prepare_messages(messages), options)

    async def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)

        # Validate capabilities against models.dev metadata
        self._validate_model_capabilities(
            "claude",
            model,
            using_json_schema=bool(options.get("json_schema")),
        )

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)

        common_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": opts["temperature"],
            "max_tokens": opts["max_tokens"],
        }
        if system_content:
            common_kwargs["system"] = system_content

        # Native JSON mode: use tool-use for schema enforcement
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                resp = await client.messages.create(  # type: ignore[call-overload]
                    **common_kwargs,
                    tools=[_build_anthropic_json_mode_tool_def(json_schema)],
                    tool_choice={"type": "tool", "name": "extract_json"},
                )
                text = ""
                for block in resp.content:
                    if block.type == "tool_use":
                        text = json.dumps(block.input)
                        break
            else:
                resp = await client.messages.create(**common_kwargs)
                text = resp.content[0].text
        else:
            resp = await client.messages.create(**common_kwargs)
            text = resp.content[0].text

        reasoning_content = ClaudeDriver._extract_thinking(resp.content)
        if not text and reasoning_content:
            text = reasoning_content

        total_cost = self._calculate_cost(
            "claude", model, resp.usage.input_tokens, resp.usage.output_tokens
        )
        meta = _build_anthropic_meta(resp, model, total_cost)

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_system_and_messages(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        return _extract_anthropic_system_and_messages(messages)

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls (Anthropic)."""
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)

        self._validate_model_capabilities("claude", model, using_tool_use=True)

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)
        anthropic_tools = _convert_tools_to_anthropic(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": opts["temperature"],
            "max_tokens": opts["max_tokens"],
            "tools": anthropic_tools,
        }
        if system_content:
            kwargs["system"] = system_content

        resp = await client.messages.create(**kwargs)

        total_cost = self._calculate_cost(
            "claude", model, resp.usage.input_tokens, resp.usage.output_tokens
        )
        meta = _build_anthropic_meta(resp, model, total_cost)

        text, tool_calls_out = _extract_anthropic_text_and_tool_calls(resp.content)
        reasoning_content = ClaudeDriver._extract_thinking(resp.content)

        result: dict[str, Any] = {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": resp.stop_reason,
        }
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
        """Yield response chunks via Anthropic streaming API."""
        if anthropic is None:
            raise RuntimeError("anthropic package not installed")

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)
        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": opts["temperature"],
            "max_tokens": opts["max_tokens"],
        }
        if system_content:
            kwargs["system"] = system_content

        full_text = ""
        full_reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta" and hasattr(event, "delta"):
                        delta_type = getattr(event.delta, "type", "")
                        if delta_type == "thinking_delta":
                            thinking_text = getattr(event.delta, "thinking", "")
                            if thinking_text:
                                full_reasoning += thinking_text
                                yield {"type": "thinking_delta", "text": thinking_text}
                        else:
                            delta_text = getattr(event.delta, "text", "")
                            if delta_text:
                                full_text += delta_text
                                yield {"type": "delta", "text": delta_text}
                    elif event.type == "message_delta" and hasattr(event, "usage"):
                        completion_tokens = getattr(event.usage, "output_tokens", 0)
                    elif event.type == "message_start" and hasattr(event, "message"):
                        usage = getattr(event.message, "usage", None)
                        if usage:
                            prompt_tokens = getattr(usage, "input_tokens", 0)

        total_cost = self._calculate_cost("claude", model, prompt_tokens, completion_tokens)
        yield _build_anthropic_stream_done(
            model, full_text, full_reasoning, prompt_tokens, completion_tokens, total_cost
        )
