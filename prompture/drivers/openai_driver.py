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
from ..infra.rate_limits import add_rate_limits, attach_rate_limit_hook, capture_rate_limits, limits_from_response
from ._prompt_cache import derive_prompt_cache_key
from ._usage_reporting import usage_meta
from .base import Driver, _apply_openai_tool_options, _normalize_stop_reason, _tool_call_dict

logger = logging.getLogger(__name__)


def _apply_openai_reporting_options(kwargs: dict[str, Any], options: dict[str, Any]) -> None:
    for key in ("service_tier", "prompt_cache_retention", "reasoning_effort"):
        if key in options:
            kwargs[key] = options[key]


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
    prompt_cache_key: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if extra:
        kwargs.update(extra)
    kwargs[tokens_param] = opts.get("max_tokens", default_max_tokens)
    if supports_temperature and "temperature" in opts:
        kwargs["temperature"] = opts["temperature"]
    # Opt-in only: OpenAI-compatible third-party endpoints (Grok, DeepSeek,
    # Moonshot, Groq) share this builder and reject unknown fields, so
    # callers pass the key explicitly rather than it being derived here.
    if prompt_cache_key:
        kwargs["prompt_cache_key"] = prompt_cache_key
    return kwargs


def _openai_prompt_cache_key(
    messages: list[dict[str, Any]],
    opts: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> str | None:
    """Derive OpenAI's ``prompt_cache_key`` from the stable request prefix.

    OpenAI caches prefixes automatically — there is no ``cache_control``
    equivalent to send — but it routes requests across machines, and two
    requests only share a cache if they land on the same one. Hashing the
    system prompt and tool definitions (the stable prefix, never the
    volatile message tail) sends every request built on that prefix to
    the same node, which is what lifts the hit rate at any real volume.

    Honours an explicit ``prompt_cache_key`` option, and is disabled
    alongside the rest of caching via ``cache_prompt=False``.
    """
    if not opts.get("cache_prompt", True):
        return None
    system_text = next(
        (str(m.get("content", "")) for m in messages if m.get("role") == "system"),
        None,
    )
    return derive_prompt_cache_key(
        system=system_text,
        tools=tools,
        explicit=opts.get("prompt_cache_key"),
    )


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
            tool_calls_out.append(
                _tool_call_dict(getattr(tc, "id", None), tc.function.name, tc.function.arguments, stop_reason)
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
    supports_streaming_tool_use = True
    supports_vision = True

    # All pricing and model config now resolved from JSON rate files (KB) and
    # models.dev live data.  See prompture/infra/rates/openai.json.
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        *,
        api: str = "chat_completions",
    ):
        if api not in ("chat_completions", "responses"):
            raise ValueError("api must be chat_completions or responses")
        self.api = api
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Optional OpenAI-compatible endpoint override (gateways/proxies such as
        # prompture-hub). Default (None) keeps the official OpenAI endpoint.
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
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
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)
        attach_rate_limit_hook(self.client)

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
        if options.get("api", getattr(self, "api", "chat_completions")) == "responses":
            if self.client is None or not hasattr(self.client, "responses"):
                from ..exceptions import ConfigurationError

                raise ConfigurationError("Responses API requires a recent openai SDK and a configured client")
            from ._openai_responses import generate

            return generate(self, self._prepare_messages(messages), options)

        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'openai package (>=1.0.0) is not installed. Install it with: pip install "prompture[openai]"'
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
        kwargs = _build_openai_base_kwargs(
            model,
            messages,
            opts,
            tokens_param,
            supports_temperature,
            512,
            prompt_cache_key=_openai_prompt_cache_key(messages, opts),
        )

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

        _apply_openai_reporting_options(kwargs, options)
        with capture_rate_limits() as limits:
            resp = self.client.chat.completions.create(**kwargs)

        meta = usage_meta(self, "openai", model, getattr(resp, "usage", None), response=resp, options=options)
        meta["raw_response"] = resp.model_dump()
        add_rate_limits(meta, limits.snapshot)

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
        if options.get("api", getattr(self, "api", "chat_completions")) == "responses":
            if self.client is None or not hasattr(self.client, "responses"):
                from ..exceptions import ConfigurationError

                raise ConfigurationError("Responses API requires a recent openai SDK and a configured client")
            from ._openai_responses import generate

            return generate(self, self._prepare_messages(messages), options, tools)

        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'openai package (>=1.0.0) is not installed. Install it with: pip install "prompture[openai]"'
            )

        model = options.get("model", self.model)
        model_config = self._get_model_config("openai", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("openai", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 4096, **options}
        kwargs = _build_openai_base_kwargs(
            model,
            messages,
            opts,
            tokens_param,
            supports_temperature,
            4096,
            extra={"tools": tools},
            prompt_cache_key=_openai_prompt_cache_key(messages, opts, tools),
        )
        _apply_openai_tool_options(kwargs, options)

        _apply_openai_reporting_options(kwargs, options)
        with capture_rate_limits() as limits:
            resp = self.client.chat.completions.create(**kwargs)

        meta = usage_meta(self, "openai", model, getattr(resp, "usage", None), response=resp, options=options)
        meta["raw_response"] = resp.model_dump()
        add_rate_limits(meta, limits.snapshot)

        choice = resp.choices[0]
        text = choice.message.content or ""
        raw_stop_reason = choice.finish_reason
        tool_calls_out = _extract_openai_tool_calls(choice.message, raw_stop_reason)
        stop_reason = _normalize_stop_reason(raw_stop_reason, tool_calls_present=bool(tool_calls_out))
        meta["raw_stop_reason"] = raw_stop_reason

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
        if options.get("api", getattr(self, "api", "chat_completions")) == "responses":
            if self.client is None or not hasattr(self.client, "responses"):
                from ..exceptions import ConfigurationError

                raise ConfigurationError("Responses API requires a recent openai SDK and a configured client")
            from ._openai_responses import stream

            yield from stream(self, self._prepare_messages(messages), options)
            return

        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'openai package (>=1.0.0) is not installed. Install it with: pip install "prompture[openai]"'
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
            prompt_cache_key=_openai_prompt_cache_key(messages, opts),
        )

        _apply_openai_reporting_options(kwargs, options)
        stream = self.client.chat.completions.create(**kwargs)

        full_text = ""
        final_usage = None
        response_info = {}

        for chunk in stream:
            for key in ("id", "model", "service_tier"):
                if isinstance(getattr(chunk, key, None), str):
                    response_info[key] = getattr(chunk, key)
            # Usage comes in the final chunk
            if getattr(chunk, "usage", None):
                final_usage = chunk.usage

            if chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or ""
                if content:
                    full_text += content
                    yield {"type": "delta", "text": content}

        meta = usage_meta(self, "openai", model, final_usage, response=response_info, options=options)
        meta["raw_response"] = {}
        add_rate_limits(meta, limits_from_response(getattr(stream, "response", None)))
        yield {"type": "done", "text": full_text, "meta": meta}

    # ------------------------------------------------------------------
    # Live streaming with interleaved tool calls
    # ------------------------------------------------------------------

    def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[Any]:
        """Stream one OpenAI turn as :class:`LiveEvent` via the shared
        OpenAI-compat helper. See
        :mod:`prompture.drivers._openai_compat_stream` for the protocol."""
        if options.get("api", getattr(self, "api", "chat_completions")) == "responses":
            if self.client is None or not hasattr(self.client, "responses"):
                from ..exceptions import ConfigurationError

                raise ConfigurationError("Responses API requires a recent openai SDK and a configured client")
            from ._openai_responses import stream

            yield from stream(self, self._prepare_messages(messages), options, tools)
            return

        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'openai package (>=1.0.0) is not installed. Install it with: pip install "prompture[openai]"'
            )

        from ._openai_compat_stream import stream_openai_compat_tool_call

        yield from stream_openai_compat_tool_call(self, messages, tools, options, provider="openai")
