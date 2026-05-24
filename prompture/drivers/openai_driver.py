"""Minimal OpenAI driver (migrated to openai>=1.0.0).
Requires the `openai` package. Uses OPENAI_API_KEY env var.
"""

import logging
import os
from collections.abc import Iterator
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .base import Driver, _parse_tool_arguments

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Shared helpers (used by both sync OpenAIDriver and AsyncOpenAIDriver)
# ----------------------------------------------------------------------


def _build_openai_base_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    opts: dict[str, Any],
    tokens_param: str,
    supports_temperature: bool,
    default_max_tokens: int,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if extra:
        kwargs.update(extra)
    kwargs[tokens_param] = opts.get("max_tokens", default_max_tokens)
    if supports_temperature and "temperature" in opts:
        kwargs["temperature"] = opts["temperature"]
    return kwargs


def _build_openai_json_mode_response_format(json_schema: dict[str, Any]) -> dict[str, Any]:
    schema_copy = prepare_strict_schema(json_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "strict": True,
            "schema": schema_copy,
        },
    }


def _extract_openai_cached_tokens(usage: Any) -> int:
    """Return the count of input tokens served from OpenAI's prompt cache.

    Returns 0 when the response carries no ``prompt_tokens_details`` block,
    which is the case for older models or short prompts that don't trigger
    automatic caching.
    """
    if usage is None:
        return 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    return int(getattr(details, "cached_tokens", 0) or 0)


def _extract_openai_meta(resp: Any, model: str, total_cost: float) -> dict[str, Any]:
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    completion_tokens = getattr(usage, "completion_tokens", 0)
    total_tokens = getattr(usage, "total_tokens", 0)
    cached_prompt_tokens = _extract_openai_cached_tokens(usage)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "cost": round(total_cost, 6),
        "raw_response": resp.model_dump(),
        "model_name": model,
    }


def _extract_openai_tool_calls(message: Any, stop_reason: str | None) -> list[dict[str, Any]]:
    tool_calls_out: list[dict[str, Any]] = []
    if message.tool_calls:
        for tc in message.tool_calls:
            args = _parse_tool_arguments(tc.function.arguments, tc.function.name, stop_reason)
            tool_calls_out.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                }
            )
    return tool_calls_out


def _build_openai_stream_done(
    model: str,
    full_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_cost: float,
    cached_prompt_tokens: int = 0,
) -> dict[str, Any]:
    return {
        "type": "done",
        "text": full_text,
        "meta": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "cost": round(total_cost, 6),
            "raw_response": {},
            "model_name": model,
        },
    }


class OpenAIDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = True

    # All pricing and model config now resolved from JSON rate files (KB) and
    # models.dev live data.  See prompture/infra/rates/openai.json.
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if OpenAI is None:
            self.client = None
            return
        if not self.api_key:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Provide api_key=... or set the "
                "OPENAI_API_KEY environment variable. "
                "See https://github.com/jhd3197/prompture#configuration"
            )
        self.client = OpenAI(api_key=self.api_key)

    @classmethod
    def list_models(cls, *, api_key: str | None = None, timeout: int = 10, **kw: object) -> list[str] | None:
        """List models available via the OpenAI API."""
        from .base import _fetch_openai_compatible_models

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return _fetch_openai_compatible_models("https://api.openai.com/v1", api_key=key, timeout=timeout)

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(self._prepare_messages(messages), options)

    def _do_generate(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "openai package (>=1.0.0) is not installed. "
                'Install it with: pip install "prompture[openai]"'
            )

        model = options.get("model", self.model)

        # Lookup model-specific config (live models.dev data + hardcoded fallback)
        model_config = self._get_model_config("openai", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        # Validate capabilities against models.dev metadata
        self._validate_model_capabilities(
            "openai",
            model,
            using_json_schema=bool(options.get("json_schema")),
        )

        opts = {"temperature": 1.0, "max_tokens": 512, **options}
        kwargs = _build_openai_base_kwargs(model, messages, opts, tokens_param, supports_temperature, 512)

        # Native JSON mode support — with graceful fallback
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema and self._should_use_json_schema("openai", model):
                kwargs["response_format"] = _build_openai_json_mode_response_format(json_schema)
            else:
                kwargs["response_format"] = {"type": "json_object"}
                if json_schema:
                    messages = self._inject_schema_into_messages(messages, json_schema)
                    kwargs["messages"] = messages

        resp = self.client.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        cached_prompt_tokens = _extract_openai_cached_tokens(usage)
        total_cost = self._calculate_cost(
            "openai", model, prompt_tokens, completion_tokens, cached_tokens=cached_prompt_tokens
        )
        meta = _extract_openai_meta(resp, model, total_cost)

        text = resp.choices[0].message.content
        return {"text": text, "meta": meta}

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
        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "openai package (>=1.0.0) is not installed. "
                'Install it with: pip install "prompture[openai]"'
            )

        model = options.get("model", self.model)
        model_config = self._get_model_config("openai", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("openai", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 4096, **options}
        kwargs = _build_openai_base_kwargs(
            model, messages, opts, tokens_param, supports_temperature, 4096, extra={"tools": tools}
        )

        resp = self.client.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        cached_prompt_tokens = _extract_openai_cached_tokens(usage)
        total_cost = self._calculate_cost(
            "openai", model, prompt_tokens, completion_tokens, cached_tokens=cached_prompt_tokens
        )
        meta = _extract_openai_meta(resp, model, total_cost)

        choice = resp.choices[0]
        text = choice.message.content or ""
        stop_reason = choice.finish_reason
        tool_calls_out = _extract_openai_tool_calls(choice.message, stop_reason)

        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks via OpenAI streaming API."""
        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "openai package (>=1.0.0) is not installed. "
                'Install it with: pip install "prompture[openai]"'
            )

        model = options.get("model", self.model)
        model_config = self._get_model_config("openai", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}
        kwargs = _build_openai_base_kwargs(
            model,
            messages,
            opts,
            tokens_param,
            supports_temperature,
            512,
            extra={"stream": True, "stream_options": {"include_usage": True}},
        )

        stream = self.client.chat.completions.create(**kwargs)

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        cached_prompt_tokens = 0

        for chunk in stream:
            # Usage comes in the final chunk
            if getattr(chunk, "usage", None):
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
                cached_prompt_tokens = _extract_openai_cached_tokens(chunk.usage)

            if chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or ""
                if content:
                    full_text += content
                    yield {"type": "delta", "text": content}

        total_cost = self._calculate_cost(
            "openai", model, prompt_tokens, completion_tokens, cached_tokens=cached_prompt_tokens
        )
        yield _build_openai_stream_done(
            model, full_text, prompt_tokens, completion_tokens, total_cost, cached_prompt_tokens
        )
