"""Async generic OpenAI-compatible driver using httpx."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .async_base import AsyncDriver
from .openai_compatible_driver import (
    OPENAI_COMPATIBLE_PROFILES,
    _parse_profile_and_model,
)

logger = logging.getLogger(__name__)


class AsyncOpenAICompatibleDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = False
    supports_streaming = False
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

    def _headers(self, api_key: str | None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return await self._do_generate(messages, options)

    async def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return await self._do_generate(messages, options)

    async def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        endpoint = (options.get("endpoint") or self.endpoint).rstrip("/")
        api_key = options.get("api_key") or self.api_key
        model = options.get("model", self.model)

        cap_provider = self.profile or "openai_compatible"
        model_config = self._get_model_config(cap_provider, model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 0.7, "max_tokens": 512, **options}
        data: dict[str, Any] = {"model": model, "messages": messages}
        data[tokens_param] = opts.get("max_tokens", 512)
        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        from ._openai_compat import apply_guided_decoding, merge_extra_body

        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                schema_copy = prepare_strict_schema(json_schema)
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": schema_copy,
                    },
                }
            else:
                data["response_format"] = {"type": "json_object"}

        # vLLM-style FSM-constrained decoding pass-through (8A-lite).
        apply_guided_decoding(
            data,
            json_schema=options.get("json_schema"),
            guided_decoding=options.get("guided_decoding"),
        )
        merge_extra_body(data, options)

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

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        total_cost = self._calculate_cost(cap_provider, model, prompt_tokens, completion_tokens)
        pricing_unknown = total_cost == 0.0

        meta: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
            "endpoint": endpoint,
            "profile": self.profile,
        }
        if pricing_unknown:
            meta["pricing_unknown"] = True

        message = resp["choices"][0]["message"]
        text = message.get("content") or ""
        return {"text": text, "meta": meta}
