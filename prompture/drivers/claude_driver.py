"""Driver for Anthropic's Claude models. Requires the `anthropic` library.
Use with API key in CLAUDE_API_KEY env var or provide directly.
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

import requests

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from ._prompt_cache import (
    CACHE_PROMPT_MIN_CHARS,
    MAX_CACHE_BREAKPOINTS,
)
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
from .base import Driver

logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility: callers and tests have long
# imported the cache threshold from this module. The implementation now
# lives in ``_prompt_cache`` so the Bedrock and async drivers share it.
__all__ = ["CACHE_PROMPT_MIN_CHARS", "MAX_CACHE_BREAKPOINTS", "ClaudeDriver"]


# ----------------------------------------------------------------------
# Shared helpers (used by both sync ClaudeDriver and AsyncClaudeDriver)
# ----------------------------------------------------------------------


def _extract_anthropic_system_and_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Separate system message and translate OpenAI-shaped tool messages to Anthropic format.

    See ClaudeDriver._extract_system_and_messages for the shape contract.
    """
    system_content: str | None = None
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_content = msg.get("content", "")
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content = msg.get("content") or ""
            blocks: list[dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in msg["tool_calls"]:
                fn = tc.get("function", tc)
                raw_args = fn.get("arguments", tc.get("arguments", {}))
                if isinstance(raw_args, str):
                    try:
                        tool_input = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        tool_input = {"_raw": raw_args}
                else:
                    tool_input = raw_args or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", tc.get("name", "")),
                        "input": tool_input,
                    }
                )
            api_messages.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # Merge consecutive tool_results into one user turn (Anthropic prefers
            # one user turn per assistant turn even with multiple tool calls).
            if (
                api_messages
                and api_messages[-1].get("role") == "user"
                and isinstance(api_messages[-1].get("content"), list)
                and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in api_messages[-1]["content"])
            ):
                api_messages[-1]["content"].append(result_block)
            else:
                api_messages.append({"role": "user", "content": [result_block]})
            continue

        api_messages.append(msg)
    return system_content, api_messages


def _convert_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anthropic_tools: list[dict[str, Any]] = []
    for t in tools:
        if "type" in t and t["type"] == "function":
            fn = t["function"]
            anthropic_tools.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        elif "input_schema" in t:
            anthropic_tools.append(t)
        else:
            anthropic_tools.append(t)
    return anthropic_tools


def _extract_anthropic_cache_tokens(usage: Any) -> tuple[int, int]:
    """Return ``(cache_read_input_tokens, cache_creation_input_tokens)``.

    Anthropic reports prompt-cache reads and writes as **separate** fields
    on ``usage`` — they are NOT included in ``input_tokens``. Both default
    to zero when caching is not used or when the SDK version doesn't
    populate the fields.
    """
    if usage is None:
        return 0, 0

    def _safe(name: str) -> int:
        val = getattr(usage, name, 0)
        if val is None or not isinstance(val, (int, float)):
            return 0
        return int(val)

    return _safe("cache_read_input_tokens"), _safe("cache_creation_input_tokens")


def _build_anthropic_meta(resp: Any, model: str, total_cost: float) -> dict[str, Any]:
    base_input = resp.usage.input_tokens
    completion_tokens = resp.usage.output_tokens
    cache_read, cache_create = _extract_anthropic_cache_tokens(resp.usage)
    # Anthropic reports cache reads/writes separately from input_tokens.
    # Surface a synthesized prompt_tokens that represents the *total* input
    # tokens consumed (matching how OpenAI reports prompt_tokens), so
    # downstream tracking and dashboards see comparable numbers.
    prompt_tokens = base_input + cache_read + cache_create
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_prompt_tokens": cache_read,
        "cache_creation_tokens": cache_create,
        "cost": round(total_cost, 6),
        "raw_response": dict(resp),
        "model_name": model,
    }


def _extract_anthropic_text_and_tool_calls(content_blocks: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    text = ""
    tool_calls_out: list[dict[str, Any]] = []
    for block in content_blocks:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tool_calls_out.append(
                {
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                }
            )
    return text, tool_calls_out


def _build_anthropic_json_mode_tool_def(json_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "extract_json",
        "description": "Extract structured data matching the schema",
        "input_schema": json_schema,
    }


# ----------------------------------------------------------------------
# Prompt caching (Anthropic "ephemeral" cache_control blocks)
#
# Anthropic prompt caching delivers ~90% input-cost savings on the
# repeated prefix: cache_creation runs at 1.25x the normal input rate
# (2x for the 1-hour TTL), but cache_read runs at 0.1x. Setting
# ``cache_control`` on a block caches everything up to and including
# that block.
#
# Three sections are markable and all three are marked here:
#   * ``system``   — stable, marked once
#   * ``tools``    — stable, marked on the last tool
#   * ``messages`` — the one that actually grows. Without a breakpoint
#     here, every round of an agentic tool loop re-sends the whole
#     accumulated conversation at full input price, which is what keeps
#     real-world cache hit rates in the single digits.
#
# Per-model minimums, TTL handling, and breakpoint placement all live in
# ``_prompt_cache`` so the Bedrock driver (same Anthropic body shape)
# and the async driver share one implementation.
# ----------------------------------------------------------------------


def _cache_opts(opts: dict[str, Any], model: str) -> dict[str, Any]:
    """Common cache kwargs derived from the caller's options.

    ``cache_prompt`` defaults on; ``cache_ttl`` selects the 5-minute
    (default) or 1-hour cache window.
    """
    return {
        "cache_prompt": opts.get("cache_prompt", True),
        "model": model,
        "ttl": opts.get("cache_ttl", "5m"),
    }


def _build_anthropic_stream_done(
    model: str,
    full_text: str,
    full_reasoning: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_cost: float,
    cached_prompt_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> dict[str, Any]:
    done_chunk: dict[str, Any] = {
        "type": "done",
        "text": full_text,
        "meta": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cost": round(total_cost, 6),
            "raw_response": {},
            "model_name": model,
        },
    }
    if full_reasoning:
        done_chunk["reasoning_content"] = full_reasoning
    return done_chunk


class ClaudeDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_streaming_tool_use = True
    supports_vision = True

    # All pricing and model config now resolved from JSON rate files (KB) and
    # models.dev live data.  See prompture/infra/rates/anthropic.json.
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(self, api_key: str | None = None, model: str = "claude-haiku-4-5-20251001"):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.model = model or os.getenv("CLAUDE_MODEL_NAME", "claude-haiku-4-5-20251001")
        # Only validate when the SDK is installed — when anthropic is missing we
        # surface that at call time so plain import-and-discover still works.
        if anthropic is not None and not self.api_key:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "CLAUDE_API_KEY is not set. Provide api_key=... or set the "
                "CLAUDE_API_KEY environment variable. "
                "See https://github.com/jhd3197/prompture#configuration"
            )

    @classmethod
    def list_models(cls, *, api_key: str | None = None, timeout: int = 10, **kw: object) -> list[str] | None:
        """List models available via the Anthropic API.

        Anthropic uses ``x-api-key`` header (not Bearer token) and requires
        an ``anthropic-version`` header.
        """
        key = api_key or os.getenv("CLAUDE_API_KEY")
        if not key:
            return None
        try:
            resp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                timeout=timeout,
            )
            if resp.status_code != 200:
                logger.debug("ClaudeDriver.list_models returned %s", resp.status_code)
                return None
            data = resp.json()
            return [m["id"] for m in data.get("data", []) if m.get("id")]
        except Exception:
            logger.debug("ClaudeDriver.list_models failed", exc_info=True)
            return None

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_claude_vision_messages

        return _prepare_claude_vision_messages(messages)

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(self._prepare_messages(messages), options)

    def _do_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        if anthropic is None:
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

        client = anthropic.Anthropic(api_key=self.api_key)

        # _do_generate uses the simple system/non-system split (no tool-message
        # translation) — pre-existing behaviour, intentionally distinct from the
        # tool-aware _extract_system_and_messages used elsewhere.
        system_content = None
        api_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                api_messages.append(msg)

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
            "max_tokens": opts["max_tokens"],
        }
        if supports_temperature:
            common_kwargs["temperature"] = opts["temperature"]
        if wrapped_system is not None:
            common_kwargs["system"] = wrapped_system

        if options.get("json_mode"):
            if json_mode_tools is not None:
                resp = client.messages.create(  # type: ignore[call-overload]
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
                resp = client.messages.create(**common_kwargs)
                text = resp.content[0].text
        else:
            resp = client.messages.create(**common_kwargs)
            text = resp.content[0].text

        reasoning_content = self._extract_thinking(resp.content)
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

    @staticmethod
    def _extract_thinking(content_blocks: list[Any]) -> str | None:
        """Extract thinking/reasoning text from Claude content blocks."""
        parts: list[str] = []
        for block in content_blocks:
            if getattr(block, "type", None) == "thinking":
                thinking_text = getattr(block, "thinking", "")
                if thinking_text:
                    parts.append(thinking_text)
        return "\n".join(parts) if parts else None

    def _extract_system_and_messages(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        return _extract_anthropic_system_and_messages(messages)

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a response that may include tool calls (Anthropic)."""
        if anthropic is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'anthropic package not installed. Install it with: pip install "prompture[anthropic]"'
            )

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)

        self._validate_model_capabilities("claude", model, using_tool_use=True)

        supports_temperature = self._get_model_config("claude", model)["supports_temperature"]

        client = anthropic.Anthropic(api_key=self.api_key)

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
            "max_tokens": opts["max_tokens"],
            "tools": anthropic_tools,
        }
        if supports_temperature:
            kwargs["temperature"] = opts["temperature"]
        if wrapped_system is not None:
            kwargs["system"] = wrapped_system

        resp = client.messages.create(**kwargs)

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

        text, tool_calls_out = _extract_anthropic_text_and_tool_calls(resp.content)
        reasoning_content = self._extract_thinking(resp.content)

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

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks via Anthropic streaming API."""
        if anthropic is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'anthropic package not installed. Install it with: pip install "prompture[anthropic]"'
            )

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)
        supports_temperature = self._get_model_config("claude", model)["supports_temperature"]
        client = anthropic.Anthropic(api_key=self.api_key)

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
            "max_tokens": opts["max_tokens"],
        }
        if supports_temperature:
            kwargs["temperature"] = opts["temperature"]
        if wrapped_system is not None:
            kwargs["system"] = wrapped_system

        full_text = ""
        full_reasoning = ""
        base_input = 0
        completion_tokens = 0
        cache_read = 0
        cache_create = 0

        with client.messages.stream(**kwargs) as stream:
            for event in stream:
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

    def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[Any]:
        """Stream one Anthropic turn as :class:`LiveEvent`, with native
        interleaved text and tool_use blocks.

        Maps Anthropic SSE events:
        - ``message_start`` → captures input/cache token counts
        - ``content_block_start`` (text|tool_use|thinking) → tracks block index
          and emits ``ToolUseStart`` for tool_use blocks
        - ``content_block_delta`` (text_delta|input_json_delta|thinking_delta)
          → emits ``TextDelta`` / ``ToolInputDelta`` / ``ThinkingDelta``
        - ``content_block_stop`` → emits ``ToolUseStop`` (with parsed input)
          for tool_use blocks
        - ``message_delta`` → captures output tokens and stop_reason
        - finally yields ``MessageStop`` with cost + usage
        """
        if anthropic is None:
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
        client = anthropic.Anthropic(api_key=self.api_key)

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
            "max_tokens": opts["max_tokens"],
            "tools": anthropic_tools,
        }
        if supports_temperature:
            kwargs["temperature"] = opts["temperature"]
        if wrapped_system is not None:
            kwargs["system"] = wrapped_system

        block_kinds: dict[int, str] = {}
        tool_block_info: dict[int, dict[str, Any]] = {}
        tool_input_buffers: dict[int, list[str]] = {}

        base_input = 0
        cache_read = 0
        cache_create = 0
        completion_tokens = 0
        stop_reason = "end_turn"

        with client.messages.stream(**kwargs) as stream:
            for event in stream:
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
                        try:
                            parsed = json.loads(buf) if buf else {}
                            if not isinstance(parsed, dict):
                                parsed = {}
                        except json.JSONDecodeError:
                            logger.warning(
                                "Failed to parse streamed tool input for %s: %r",
                                info.get("name", "?"),
                                buf[:200],
                            )
                            parsed = {}
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
        }
        yield MessageStop(stop_reason=stop_reason, usage=meta)
