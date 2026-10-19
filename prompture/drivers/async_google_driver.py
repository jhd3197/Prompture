"""Async Google Generative AI (Gemini) driver."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from ._google_genai import add_tools, build_generation_args, extract_usage, response_parts, tool_response
from .async_base import AsyncDriver
from .base import DriverHTTPError
from .google_driver import GoogleDriver

logger = logging.getLogger(__name__)


class AsyncGoogleDriver(CostMixin, AsyncDriver):
    """Async driver for Google's Generative AI API (Gemini)."""

    supports_json_mode = True
    supports_json_schema = True
    supports_vision = True
    supports_tool_use = True
    supports_streaming = True

    MODEL_PRICING = GoogleDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-pro"):
        if genai is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                'google-genai package is not installed. Install it with: pip install "prompture[google]"'
            )
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "GOOGLE_API_KEY is not set. Provide api_key=... or set the "
                "GOOGLE_API_KEY environment variable. "
                "See https://github.com/jhd3197/prompture#configuration"
            )
        self.model = model
        self._client = genai.Client(api_key=self.api_key)
        self.options: dict[str, Any] = {}

    def _calculate_cost_chars(self, prompt_chars: int, completion_chars: int) -> float:
        """Calculate cost from character counts (same logic as sync GoogleDriver).

        Estimates tokens from characters (~4 chars/token) and uses
        models.dev live rates.  Returns 0.0 if no rates available.
        """
        from ..infra.model_rates import get_model_rates

        live_rates = get_model_rates("google", self.model)
        if live_rates:
            est_prompt_tokens = prompt_chars / 4
            est_completion_tokens = completion_chars / 4
            prompt_cost = (est_prompt_tokens / 1_000_000) * live_rates["input"]
            completion_cost = (est_completion_tokens / 1_000_000) * live_rates["output"]
            return round(prompt_cost + completion_cost, 6)

        return 0.0

    def _extract_usage_metadata(self, response: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return extract_usage(
            response, messages, lambda p, c, **kw: self._calculate_cost("google", self.model, p, c, **kw)
        )

    supports_messages = True

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_google_vision_messages

        return _prepare_google_vision_messages(messages)

    def _build_generation_args(
        self, messages: list[dict[str, Any]], options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        return build_generation_args(self, messages, options, types)

    async def generate(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(self._prepare_messages(messages), options)

    async def _do_generate(
        self, messages: list[dict[str, str]], options: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        gen_input, config_dict = self._build_generation_args(messages, options)

        # Validate capabilities against models.dev metadata
        self._validate_model_capabilities(
            "google",
            self.model,
            using_json_schema=bool((options or {}).get("json_schema")),
        )

        try:
            config = types.GenerateContentConfig(**config_dict)
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=gen_input,
                config=config,
            )

            if not response.text:
                raise ValueError("Empty response from model")

            usage_meta = self._extract_usage_metadata(response, messages)

            meta = {
                **usage_meta,
                "raw_response": response.prompt_feedback if hasattr(response, "prompt_feedback") else None,
                "model_name": self.model,
            }

            return {"text": response.text, "meta": meta, "_google_parts": response_parts(response)}

        except (genai_errors.APIError, ValueError) as e:
            logger.error(f"Google API request failed: {e}")
            raise DriverHTTPError(
                f"Google API request failed: {e}",
                status_code=getattr(e, "code", None),
                provider="google",
            ) from e

    # ------------------------------------------------------------------
    # Tool use
    # ------------------------------------------------------------------

    async def generate_messages_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], options: dict[str, Any]
    ) -> dict[str, Any]:
        self._validate_model_capabilities("google", self.model, using_tool_use=True)
        gen_input, config_dict = self._build_generation_args(self._prepare_messages(messages), options)
        add_tools(config_dict, tools, options, types)
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=gen_input,
                config=types.GenerateContentConfig(**config_dict),
            )
            meta = {
                **self._extract_usage_metadata(response, messages),
                "raw_response": getattr(response, "prompt_feedback", None),
                "model_name": self.model,
            }
            return tool_response(response, meta)
        except (genai_errors.APIError, ValueError) as e:
            raise DriverHTTPError(
                f"Google API tool call request failed: {e}",
                status_code=getattr(e, "code", None),
                provider="google",
            ) from e

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield response chunks via Gemini async streaming API."""
        gen_input, config_dict = self._build_generation_args(self._prepare_messages(messages), options)

        try:
            config = types.GenerateContentConfig(**config_dict)
            response = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=gen_input,
                config=config,
            )

            full_text = ""
            last_usage = None
            stream_parts = []
            async for chunk in response:
                if getattr(chunk, "usage_metadata", None) is not None:
                    last_usage = chunk.usage_metadata
                stream_parts.extend(response_parts(chunk))
                chunk_text = getattr(chunk, "text", None) or ""
                if chunk_text:
                    full_text += chunk_text
                    yield {"type": "delta", "text": chunk_text}

            # Gemini reports cumulative usage on response chunks.
            usage_meta = self._extract_usage_metadata(
                SimpleNamespace(usage_metadata=last_usage, text=full_text), messages
            )

            yield {
                "type": "done",
                "_google_parts": stream_parts,
                "text": full_text,
                "meta": {
                    **usage_meta,
                    "raw_response": {},
                    "model_name": self.model,
                },
            }

        except (genai_errors.APIError, ValueError) as e:
            logger.error(f"Google API streaming request failed: {e}")
            raise DriverHTTPError(
                f"Google API streaming request failed: {e}",
                status_code=getattr(e, "code", None),
                provider="google",
            ) from e
