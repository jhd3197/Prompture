"""Async Groq driver. Requires the ``groq`` package."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import groq
except ImportError:
    groq = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from ..infra.rate_limits import add_rate_limits, attach_rate_limit_hook, capture_rate_limits
from ._effort import apply_effort
from ._openai_chat import AsyncOpenAISDKChatTools, AsyncOpenAISDKChatToolStream, get_chat_profile
from .async_base import AsyncDriver
from .groq_driver import GroqDriver

logger = logging.getLogger(__name__)


class AsyncGroqDriver(AsyncOpenAISDKChatTools, AsyncOpenAISDKChatToolStream, CostMixin, AsyncDriver):
    chat_profile = get_chat_profile("groq")

    supports_json_mode = True
    supports_tool_use = True
    supports_streaming_tool_use = True
    supports_vision = True

    MODEL_PRICING = GroqDriver.MODEL_PRICING

    def __init__(self, api_key: str | None = None, model: str = "llama2-70b-4096"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        if groq is None:
            self.client: Any = None
            return
        if not self.api_key:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "GROQ_API_KEY is not set. Provide api_key=... or set the "
                "GROQ_API_KEY environment variable. "
                "See https://github.com/jhd3197/prompture#configuration"
            )
        self.client: Any = groq.AsyncClient(api_key=self.api_key)
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
        if self.client is None:
            from ..exceptions import ConfigurationError

            raise ConfigurationError('groq package is not installed. Install it with: pip install "prompture[groq]"')

        model = options.get("model", self.model)

        model_config = self._get_model_config("groq", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 0.7, "max_tokens": 512, **options}

        kwargs = {
            "model": model,
            "messages": messages,
        }
        kwargs[tokens_param] = opts.get("max_tokens", 512)
        apply_effort(kwargs, opts, "openai")

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            kwargs["response_format"] = {"type": "json_object"}

        with capture_rate_limits() as limits:
            resp = await self.client.chat.completions.create(**kwargs)

        from .openai_driver import _extract_openai_cached_tokens

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
        cached_prompt_tokens = _extract_openai_cached_tokens(usage)

        total_cost = self._calculate_cost(
            "groq",
            model,
            prompt_tokens,
            completion_tokens,
            cached_tokens=cached_prompt_tokens,
        )

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp.model_dump(),
            "model_name": model,
        }

        text = resp.choices[0].message.content or ""
        reasoning_content = getattr(resp.choices[0].message, "reasoning_content", None)

        if not text and reasoning_content:
            text = reasoning_content

        add_rate_limits(meta, limits.snapshot)
        result: dict[str, Any] = {"text": text, "meta": meta}
        if reasoning_content is not None:
            result["reasoning_content"] = reasoning_content
        return result
