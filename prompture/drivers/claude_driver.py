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
except Exception:
    anthropic = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from .base import Driver

logger = logging.getLogger(__name__)


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
            raise RuntimeError("anthropic package not installed")

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

        common_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": opts["max_tokens"],
        }
        if supports_temperature:
            common_kwargs["temperature"] = opts["temperature"]
        if system_content:
            common_kwargs["system"] = system_content

        # Native JSON mode: use tool-use for schema enforcement
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                resp = client.messages.create(  # type: ignore[call-overload]
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
            raise RuntimeError("anthropic package not installed")

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)

        self._validate_model_capabilities("claude", model, using_tool_use=True)

        supports_temperature = self._get_model_config("claude", model)["supports_temperature"]

        client = anthropic.Anthropic(api_key=self.api_key)

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)
        anthropic_tools = _convert_tools_to_anthropic(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": opts["max_tokens"],
            "tools": anthropic_tools,
        }
        if supports_temperature:
            kwargs["temperature"] = opts["temperature"]
        if system_content:
            kwargs["system"] = system_content

        resp = client.messages.create(**kwargs)

        cache_read, cache_create = _extract_anthropic_cache_tokens(resp.usage)
        total_cost = self._calculate_cost(
            "claude",
            model,
            resp.usage.input_tokens + cache_read + cache_create,
            resp.usage.output_tokens,
            cached_tokens=cache_read,
            cache_creation_tokens=cache_create,
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
            raise RuntimeError("anthropic package not installed")

        opts = {**{"temperature": 0.0, "max_tokens": 512}, **options}
        model = options.get("model", self.model)
        supports_temperature = self._get_model_config("claude", model)["supports_temperature"]
        client = anthropic.Anthropic(api_key=self.api_key)

        system_content, api_messages = _extract_anthropic_system_and_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_tokens": opts["max_tokens"],
        }
        if supports_temperature:
            kwargs["temperature"] = opts["temperature"]
        if system_content:
            kwargs["system"] = system_content

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
