"""Async generic OpenAI-compatible driver using httpx."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..infra.cost_mixin import CostMixin
from ..infra.rate_limits import add_rate_limits, limits_from_response
from .async_base import AsyncDriver
from .openai_compatible_driver import (
    OPENAI_COMPATIBLE_PROFILES,
    OpenAICompatibleDriver,
    _parse_profile_and_model,
)

logger = logging.getLogger(__name__)


class AsyncOpenAICompatibleDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming_tool_use = True
    supports_streaming = True
    supports_messages = True

    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "",
        endpoint: str | None = None,
        profile: str | None = None,
    ):
        if profile is None and endpoint is None:
            detected_profile, parsed_model = _parse_profile_and_model(model)
            if detected_profile is not None:
                profile = detected_profile
                model = parsed_model

        if profile is not None and profile not in OPENAI_COMPATIBLE_PROFILES:
            raise ValueError(
                f"Unknown openai_compatible profile {profile!r}. Known: {sorted(OPENAI_COMPATIBLE_PROFILES)}"
            )

        resolved_endpoint: str | None = endpoint
        env_var: str | None = None
        if resolved_endpoint is None and profile is not None:
            resolved_endpoint = OPENAI_COMPATIBLE_PROFILES[profile]["endpoint"]
            env_var = OPENAI_COMPATIBLE_PROFILES[profile]["env_var"]

        if not resolved_endpoint:
            raise ValueError(
                "AsyncOpenAICompatibleDriver requires either a `profile` (one of "
                f"{sorted(OPENAI_COMPATIBLE_PROFILES)}) or an explicit "
                "`endpoint`."
            )

        resolved_key = api_key
        if not resolved_key and env_var:
            resolved_key = os.getenv(env_var)
        if not resolved_key:
            resolved_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")

        self.api_key = resolved_key
        self.endpoint = resolved_endpoint.rstrip("/")
        self.profile = profile
        self.model = model

    def _usage_model_name(self, model: str) -> str | None:
        """Record usage as ``openai_compatible/<profile>/<model>``, the same
        string the model is requested with, so ledgers and gateways agree."""
        if model.startswith("openai_compatible/"):
            return model
        return f"openai_compatible/{self.profile}/{model}" if self.profile else f"openai_compatible/{model}"

    def _headers(self, api_key: str | None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(messages, options)

    # Keep request shaping, tool parsing and metadata identical across transports.
    _build_request = OpenAICompatibleDriver._build_request
    _parse_response = OpenAICompatibleDriver._parse_response
    _stream_usage = OpenAICompatibleDriver._stream_usage

    async def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._do_generate(messages, options, tools=tools)

    async def _do_generate(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        endpoint, api_key, model, data = self._build_request(messages, options, tools)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{endpoint}/chat/completions",
                    headers=self._headers(api_key),
                    json=data,
                    timeout=120,
                )
                response.raise_for_status()
                resp = response.json()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"OpenAI-compatible API request failed: {e!s}") from e
            except Exception as e:
                raise RuntimeError(f"OpenAI-compatible API request failed: {e!s}") from e

        result = self._parse_response(resp, model, endpoint, tools=tools)
        add_rate_limits(result["meta"], limits_from_response(response))
        return result

    async def _stream_events(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Any]:
        from ._openai_compat_stream import astream_raw_http_compat_tool_call

        endpoint, api_key, _model, payload = self._build_request(messages, options, tools, stream=True)
        try:
            async for event in astream_raw_http_compat_tool_call(
                self,
                messages,
                tools or [],
                options,
                provider=self.profile or "openai_compatible",
                url=f"{endpoint}/chat/completions",
                headers=self._headers(api_key),
                payload=payload,
            ):
                if event.event_type == "message_stop":
                    event.usage.update(self._stream_usage(event.usage, endpoint))
                yield event
        except httpx.HTTPError as e:
            raise RuntimeError(f"OpenAI-compatible API request failed: {e!s}") from e

    async def generate_messages_with_tools_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[Any]:
        """Stream text and native tool calls as LiveEvents."""
        async for event in self._stream_events(messages, options, tools):
            yield event

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream text deltas followed by complete text and usage metadata."""
        full_text = ""
        async for event in self._stream_events(messages, options):
            if event.event_type == "text_delta":
                full_text += event.text
                yield {"type": "delta", "text": event.text}
            elif event.event_type == "thinking_delta":
                yield {"type": "thinking_delta", "text": event.text}
            elif event.event_type == "message_stop":
                yield {"type": "done", "text": full_text, "meta": event.usage}
