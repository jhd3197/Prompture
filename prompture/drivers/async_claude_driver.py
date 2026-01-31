"""Async Anthropic Claude driver. Requires the ``anthropic`` package."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

try:
    import anthropic
except Exception:
    anthropic = None

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin
from .claude_driver import ClaudeDriver


class AsyncClaudeDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = True

    MODEL_PRICING = ClaudeDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL_NAME", "claude-3-5-haiku-20241022")

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

        # Anthropic requires system messages as a top-level parameter
        system_content, api_messages = self._extract_system_and_messages(messages)

        # Build common kwargs
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
                tool_def = {
                    "name": "extract_json",
                    "description": "Extract structured data matching the schema",
                    "input_schema": json_schema,
                }
                resp = await client.messages.create(
                    **common_kwargs,
                    tools=[tool_def],
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

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens

        total_cost = self._calculate_cost("claude", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": dict(resp),
            "model_name": model,
        }

        return {"text": text, "meta": meta}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_system_and_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Separate system message from conversation messages for Anthropic API."""
        system_content = None
        api_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                api_messages.append(msg)
        return system_content, api_messages

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

        system_content, api_messages = self._extract_system_and_messages(messages)

        # Convert tools from OpenAI format to Anthropic format if needed
        anthropic_tools = []
        for t in tools:
            if "type" in t and t["type"] == "function":
                # OpenAI format -> Anthropic format
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
            elif "input_schema" in t:
                # Already Anthropic format
                anthropic_tools.append(t)
            else:
                anthropic_tools.append(t)

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

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("claude", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": dict(resp),
            "model_name": model,
        }

        text = ""
        tool_calls_out: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls_out.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": resp.stop_reason,
        }

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

        system_content, api_messages = self._extract_system_and_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "temperature": opts["temperature"],
            "max_tokens": opts["max_tokens"],
        }
        if system_content:
            kwargs["system"] = system_content

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta" and hasattr(event, "delta"):
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

        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost("claude", model, prompt_tokens, completion_tokens)

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
