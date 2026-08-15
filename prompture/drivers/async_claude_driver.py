"""Async Anthropic Claude driver. Requires the ``anthropic`` package."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from ._prompt_cache import (
    apply_cache_control_to_messages as _apply_cache_control_to_messages,
)
from ._prompt_cache import (
    apply_cache_control_to_system as _apply_cache_control_to_system,
)
from ._prompt_cache import (
    apply_cache_control_to_tools as _apply_cache_control_to_tools,
)
from ._prompt_cache import (
    breakpoint_budget as _breakpoint_budget,
)
from ._prompt_cache import (
    cache_write_multiplier as _cache_write_multiplier,
)
from .async_base import AsyncDriver
from .base import _normalize_stop_reason, _translate_tool_choice
from .claude_driver import (
    ClaudeDriver,
    _build_anthropic_json_mode_tool_def,
    _build_anthropic_meta,
    _build_anthropic_stream_done,
    _cache_opts,
    _convert_tools_to_anthropic,
    _extract_anthropic_cache_tokens,
    _extract_anthropic_system_and_messages,
    _extract_anthropic_text_and_tool_calls,
)

logger = logging.getLogger(__name__)


class AsyncClaudeDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_streaming_tool_use = True
    supports_vision = True

    MODEL_PRICING = ClaudeDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL_NAME", "claude-haiku-4-5-20251001")
        if anthropic is None:
            self.client = None
            return
        if not self.api_key:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "CLAUDE_API_KEY is not set. Provide api_key=... or set the "
                "CLAUDE_API_KEY environment variable. "
                "See https://github.com/jhd3197/prompture#configuration"
            )
        # Cache one AsyncAnthropic client for the lifetime of this driver so
        # we reuse its HTTP/2 connection pool instead of building a fresh TLS
        # session per request.
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

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
        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'anthropic package not installed. Install it with: pip install "prompture[anthropic]"'
            )

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)

        # Validate capabilities against models.dev metadata
        self._validate_model_capabilities(
            "claude",
            model,
            using_json_schema=bool(options.get("json_schema")),
        )

        supports_temperature = self._get_model_config("claude", model)["supports_temperature"]

        client = self.client

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)

        cache_kwargs = _cache_opts(opts, model)
        wrapped_system = _apply_cache_control_to_system(system_content, **cache_kwargs)

        # Native JSON mode: use tool-use for schema enforcement
        json_mode_tools: list[dict[str, Any]] | None = None
        if options.get("json_mode") and options.get("json_schema"):
            json_mode_tools = _apply_cache_control_to_tools(
                [_build_anthropic_json_mode_tool_def(options["json_schema"])],
                **cache_kwargs,
            )

        # Messages last: the breakpoint budget is whatever system and
        # tools didn't already spend (Anthropic hard-errors on a 5th).
        api_messages = _apply_cache_control_to_messages(
            api_messages,
            max_breakpoints=_breakpoint_budget(wrapped_system, json_mode_tools),
            **cache_kwargs,
        )

        common_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": self._effective_max_tokens(model, opts),
        }
        if supports_temperature:
            common_kwargs["temperature"] = opts["temperature"]
        if wrapped_system is not None:
            common_kwargs["system"] = wrapped_system
        if opts.get("timeout") is not None:
            common_kwargs["timeout"] = opts["timeout"]

        if options.get("json_mode"):
            if json_mode_tools is not None:
                resp = await client.messages.create(  # type: ignore[call-overload]
                    **common_kwargs,
                    tools=json_mode_tools,
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

        cache_read, cache_create = _extract_anthropic_cache_tokens(resp.usage)
        total_cost = self._calculate_cost(
            "claude",
            model,
            resp.usage.input_tokens + cache_read + cache_create,
            resp.usage.output_tokens,
            cached_tokens=cache_read,
            cache_creation_tokens=cache_create,
            cache_write_multiplier=_cache_write_multiplier(opts.get("cache_ttl", "5m")),
        )
        meta = _build_anthropic_meta(resp, model, total_cost)

        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_max_tokens(self, model: str, opts: dict[str, Any]) -> int:
        """Clamp the requested max_tokens to the model's known output cap.

        Callers commonly pass a generous default (e.g. 65536). Anthropic
        rejects values above the model ceiling (Claude 4 family: 64000),
        and the SDK refuses long non-streaming requests outright - so
        clamp when the capabilities KB knows the real limit.
        """
        requested = int(opts.get("max_tokens") or 512)
        try:
            cap = self._get_model_config("claude", model).get("max_output_tokens")
        except Exception:
            cap = None
        if cap and requested > int(cap):
            return int(cap)
        return requested

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
        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'anthropic package not installed. Install it with: pip install "prompture[anthropic]"'
            )

        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}
        model = options.get("model", self.model)

        self._validate_model_capabilities("claude", model, using_tool_use=True)

        supports_temperature = self._get_model_config("claude", model)["supports_temperature"]

        client = self.client

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)
        cache_kwargs = _cache_opts(opts, model)
        anthropic_tools = _apply_cache_control_to_tools(_convert_tools_to_anthropic(tools), **cache_kwargs)
        wrapped_system = _apply_cache_control_to_system(system_content, **cache_kwargs)
        api_messages = _apply_cache_control_to_messages(
            api_messages,
            max_breakpoints=_breakpoint_budget(wrapped_system, anthropic_tools),
            **cache_kwargs,
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": self._effective_max_tokens(model, opts),
            "tools": anthropic_tools,
        }
        if supports_temperature:
            kwargs["temperature"] = opts["temperature"]
        if wrapped_system is not None:
            kwargs["system"] = wrapped_system
        if opts.get("timeout") is not None:
            kwargs["timeout"] = opts["timeout"]
        tool_choice = _translate_tool_choice(options.get("tool_choice"), "anthropic")
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        resp = await client.messages.create(**kwargs)

        cache_read, cache_create = _extract_anthropic_cache_tokens(resp.usage)
        total_cost = self._calculate_cost(
            "claude",
            model,
            resp.usage.input_tokens + cache_read + cache_create,
            resp.usage.output_tokens,
            cached_tokens=cache_read,
            cache_creation_tokens=cache_create,
            cache_write_multiplier=_cache_write_multiplier(opts.get("cache_ttl", "5m")),
        )
        meta = _build_anthropic_meta(resp, model, total_cost)
        meta["raw_stop_reason"] = resp.stop_reason

        text, tool_calls_out = _extract_anthropic_text_and_tool_calls(resp.content)
        reasoning_content = ClaudeDriver._extract_thinking(resp.content)

        result: dict[str, Any] = {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": _normalize_stop_reason(resp.stop_reason, tool_calls_present=bool(tool_calls_out)),
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
        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'anthropic package not installed. Install it with: pip install "prompture[anthropic]"'
            )

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)
        supports_temperature = self._get_model_config("claude", model)["supports_temperature"]
        client = self.client

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)
        cache_kwargs = _cache_opts(opts, model)
        wrapped_system = _apply_cache_control_to_system(system_content, **cache_kwargs)
        api_messages = _apply_cache_control_to_messages(
            api_messages,
            max_breakpoints=_breakpoint_budget(wrapped_system),
            **cache_kwargs,
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": self._effective_max_tokens(model, opts),
        }
        if supports_temperature:
            kwargs["temperature"] = opts["temperature"]
        if wrapped_system is not None:
            kwargs["system"] = wrapped_system
        if opts.get("timeout") is not None:
            kwargs["timeout"] = opts["timeout"]

        full_text = ""
        full_reasoning = ""
        base_input = 0
        completion_tokens = 0
        cache_read = 0
        cache_create = 0

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
                            base_input = getattr(usage, "input_tokens", 0)
                            cache_read, cache_create = _extract_anthropic_cache_tokens(usage)

        prompt_tokens = base_input + cache_read + cache_create
        total_cost = self._calculate_cost(
            "claude",
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens=cache_read,
            cache_creation_tokens=cache_create,
            cache_write_multiplier=_cache_write_multiplier(opts.get("cache_ttl", "5m")),
        )
        yield _build_anthropic_stream_done(
            model,
            full_text,
            full_reasoning,
            prompt_tokens,
            completion_tokens,
            total_cost,
            cached_prompt_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )

    # ------------------------------------------------------------------
    # Live streaming with interleaved tool calls
    # ------------------------------------------------------------------

    async def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Async sibling of :meth:`ClaudeDriver.generate_messages_with_tools_stream`.

        Maps Anthropic SSE events to :class:`LiveEvent` for one turn,
        preserving interleaved text + thinking + tool_use ordering.
        """
        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'anthropic package not installed. Install it with: pip install "prompture[anthropic]"'
            )

        from ..agents.live_events import (
            MessageStop,
            TextDelta,
            ThinkingDelta,
            ToolInputDelta,
            ToolUseStart,
            ToolUseStop,
        )

        opts = {**{"temperature": 0.0, "max_tokens": 4096}, **options}
        model = options.get("model", self.model)
        supports_temperature = self._get_model_config("claude", model)["supports_temperature"]
        client = self.client

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)
        cache_kwargs = _cache_opts(opts, model)
        anthropic_tools = _apply_cache_control_to_tools(_convert_tools_to_anthropic(tools), **cache_kwargs)
        wrapped_system = _apply_cache_control_to_system(system_content, **cache_kwargs)
        api_messages = _apply_cache_control_to_messages(
            api_messages,
            max_breakpoints=_breakpoint_budget(wrapped_system, anthropic_tools),
            **cache_kwargs,
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": self._effective_max_tokens(model, opts),
            "tools": anthropic_tools,
        }
        if supports_temperature:
            kwargs["temperature"] = opts["temperature"]
        if wrapped_system is not None:
            kwargs["system"] = wrapped_system
        if opts.get("timeout") is not None:
            kwargs["timeout"] = opts["timeout"]
        tool_choice = _translate_tool_choice(options.get("tool_choice"), "anthropic")
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        block_kinds: dict[int, str] = {}
        tool_block_info: dict[int, dict[str, Any]] = {}
        tool_input_buffers: dict[int, list[str]] = {}
        # Tool blocks whose input JSON failed to parse are withheld until
        # the message-level stop_reason arrives (message_delta comes AFTER
        # content_block_stop) so truncation can be flagged accurately.
        pending_failed_stops: list[dict[str, Any]] = []

        base_input = 0
        cache_read = 0
        cache_create = 0
        completion_tokens = 0
        stop_reason = "end_turn"

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                ev_type = getattr(event, "type", "")
                if ev_type == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    if usage is not None:
                        base_input = getattr(usage, "input_tokens", 0) or 0
                        cache_read, cache_create = _extract_anthropic_cache_tokens(usage)
                elif ev_type == "content_block_start":
                    idx = getattr(event, "index", 0)
                    block = getattr(event, "content_block", None)
                    kind = getattr(block, "type", "") if block is not None else ""
                    block_kinds[idx] = kind
                    if kind == "tool_use":
                        tool_id = getattr(block, "id", "") or ""
                        tool_name = getattr(block, "name", "") or ""
                        tool_block_info[idx] = {"id": tool_id, "name": tool_name}
                        tool_input_buffers[idx] = []
                        yield ToolUseStart(id=tool_id, name=tool_name)
                elif ev_type == "content_block_delta":
                    idx = getattr(event, "index", 0)
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", "") if delta is not None else ""
                    if delta_type == "text_delta":
                        text_piece = getattr(delta, "text", "") or ""
                        if text_piece:
                            yield TextDelta(text=text_piece)
                    elif delta_type == "thinking_delta":
                        thinking_piece = getattr(delta, "thinking", "") or ""
                        if thinking_piece:
                            yield ThinkingDelta(text=thinking_piece)
                    elif delta_type == "input_json_delta":
                        fragment = getattr(delta, "partial_json", "") or ""
                        info = tool_block_info.get(idx)
                        if fragment and info is not None:
                            tool_input_buffers.setdefault(idx, []).append(fragment)
                            yield ToolInputDelta(id=info["id"], fragment=fragment)
                elif ev_type == "content_block_stop":
                    idx = getattr(event, "index", 0)
                    if block_kinds.get(idx) == "tool_use":
                        info = tool_block_info.get(idx, {})
                        buf = "".join(tool_input_buffers.get(idx, []))
                        parse_failed = False
                        try:
                            parsed = json.loads(buf) if buf else {}
                            if not isinstance(parsed, dict):
                                parse_failed = True
                                parsed = {}
                        except json.JSONDecodeError:
                            parse_failed = True
                            parsed = {}
                        if parse_failed:
                            logger.warning(
                                "Failed to parse streamed tool input for %s: %r",
                                info.get("name", "?"),
                                buf[:200],
                            )
                            pending_failed_stops.append({"info": info})
                        else:
                            yield ToolUseStop(
                                id=info.get("id", ""),
                                name=info.get("name", ""),
                                input=parsed,
                            )
                elif ev_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        completion_tokens = getattr(usage, "output_tokens", 0) or completion_tokens
                    sr = getattr(getattr(event, "delta", None), "stop_reason", None)
                    if sr:
                        stop_reason = sr

        # Flush tool stops whose input failed to parse, now that the final
        # stop_reason is known (contract C3: truncated iff max_tokens).
        for pending in pending_failed_stops:
            info = pending["info"]
            yield ToolUseStop(
                id=info.get("id", ""),
                name=info.get("name", ""),
                input={},
                truncated=stop_reason == "max_tokens",
                raw_stop_reason=stop_reason,
            )

        prompt_tokens = base_input + cache_read + cache_create
        total_cost = self._calculate_cost(
            "claude",
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens=cache_read,
            cache_creation_tokens=cache_create,
            cache_write_multiplier=_cache_write_multiplier(opts.get("cache_ttl", "5m")),
        )
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_prompt_tokens": cache_read,
            "cache_creation_tokens": cache_create,
            "cost": round(total_cost, 6),
            "model_name": model,
            "raw_stop_reason": stop_reason,
        }
        yield MessageStop(stop_reason=_normalize_stop_reason(stop_reason), usage=meta)
