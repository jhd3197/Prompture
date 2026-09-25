"""Async OpenAI driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[misc, assignment]

from ..infra.cost_mixin import CostMixin
from ..infra.rate_limits import add_rate_limits, attach_rate_limit_hook, capture_rate_limits, limits_from_response
from ._usage_reporting import usage_meta
from .async_base import AsyncDriver
from .base import _apply_openai_tool_options, _normalize_stop_reason
from .openai_driver import (
    OpenAIDriver,
    _apply_openai_reporting_options,
    _build_openai_base_kwargs,
    _build_openai_json_mode_response_format,
    _extract_openai_tool_calls,
    _openai_prompt_cache_key,
)

logger = logging.getLogger(__name__)


class AsyncOpenAIDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = True
    supports_streaming_tool_use = True
    supports_vision = True

    MODEL_PRICING = OpenAIDriver.MODEL_PRICING

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
        self.model = model
        if AsyncOpenAI is None:
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
        if base_url or os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = AsyncOpenAI(**client_kwargs)
        attach_rate_limit_hook(self.client)

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
        if options.get("api", getattr(self, "api", "chat_completions")) == "responses":
            if self.client is None or not hasattr(self.client, "responses"):
                from ..exceptions import ConfigurationError

                raise ConfigurationError("Responses API requires a recent openai SDK and a configured client")
            from ._openai_responses import agenerate

            return await agenerate(self, self._prepare_messages(messages), options)

        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'openai package (>=1.0.0) is not installed. Install it with: pip install "prompture[openai]"'
            )

        model = options.get("model", self.model)

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
            resp = await self.client.chat.completions.create(**kwargs)

        meta = usage_meta(self, "openai", model, getattr(resp, "usage", None), response=resp, options=options)
        meta["raw_response"] = resp.model_dump()
        add_rate_limits(meta, limits.snapshot)

        text = resp.choices[0].message.content
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
        if options.get("api", getattr(self, "api", "chat_completions")) == "responses":
            if self.client is None or not hasattr(self.client, "responses"):
                from ..exceptions import ConfigurationError

                raise ConfigurationError("Responses API requires a recent openai SDK and a configured client")
            from ._openai_responses import agenerate

            return await agenerate(self, self._prepare_messages(messages), options, tools)

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
            resp = await self.client.chat.completions.create(**kwargs)

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

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield response chunks via OpenAI streaming API."""
        if options.get("api", getattr(self, "api", "chat_completions")) == "responses":
            if self.client is None or not hasattr(self.client, "responses"):
                from ..exceptions import ConfigurationError

                raise ConfigurationError("Responses API requires a recent openai SDK and a configured client")
            from ._openai_responses import astream

            async for response_event in astream(self, self._prepare_messages(messages), options):
                yield response_event
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
        stream = await self.client.chat.completions.create(**kwargs)

        full_text = ""
        final_usage = None
        response_info = {}

        async for chunk in stream:
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

    async def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Async streaming-tool via the shared OpenAI-compat helper."""
        if options.get("api", getattr(self, "api", "chat_completions")) == "responses":
            if self.client is None or not hasattr(self.client, "responses"):
                from ..exceptions import ConfigurationError

                raise ConfigurationError("Responses API requires a recent openai SDK and a configured client")
            from ._openai_responses import astream

            async for response_event in astream(self, self._prepare_messages(messages), options, tools):
                yield response_event
            return

        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'openai package (>=1.0.0) is not installed. Install it with: pip install "prompture[openai]"'
            )

        from ._openai_compat_stream import astream_openai_compat_tool_call

        async for ev in astream_openai_compat_tool_call(self, messages, tools, options, provider="openai"):
            yield ev
