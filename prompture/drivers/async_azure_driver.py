"""Async Azure OpenAI driver. Requires the ``openai`` package (>=1.0.0)."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from openai import AsyncAzureOpenAI
except Exception:
    AsyncAzureOpenAI = None

from ..async_driver import AsyncDriver
from ..cost_mixin import CostMixin, prepare_strict_schema
from .azure_driver import AzureDriver


class AsyncAzureDriver(CostMixin, AsyncDriver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_vision = True

    MODEL_PRICING = AzureDriver.MODEL_PRICING

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        deployment_id: str | None = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv("AZURE_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_API_ENDPOINT")
        self.deployment_id = deployment_id or os.getenv("AZURE_DEPLOYMENT_ID")
        self.api_version = os.getenv("AZURE_API_VERSION", "2023-07-01-preview")
        self.model = model

        if not self.api_key:
            raise ValueError("Missing Azure API key (AZURE_API_KEY).")
        if not self.endpoint:
            raise ValueError("Missing Azure API endpoint (AZURE_API_ENDPOINT).")
        if not self.deployment_id:
            raise ValueError("Missing Azure deployment ID (AZURE_DEPLOYMENT_ID).")

        if AsyncAzureOpenAI:
            self.client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )
        else:
            self.client = None

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
            raise RuntimeError("openai package (>=1.0.0) with AsyncAzureOpenAI not installed")

        model = options.get("model", self.model)
        model_config = self._get_model_config("azure", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        kwargs = {
            "model": self.deployment_id,
            "messages": messages,
        }
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        # Native JSON mode support
        if options.get("json_mode"):
            json_schema = options.get("json_schema")
            if json_schema:
                schema_copy = prepare_strict_schema(json_schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": schema_copy,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}

        resp = await self.client.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)

        total_cost = self._calculate_cost("azure", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp.model_dump(),
            "model_name": model,
            "deployment_id": self.deployment_id,
        }

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
        if self.client is None:
            raise RuntimeError("openai package (>=1.0.0) with AsyncAzureOpenAI not installed")

        model = options.get("model", self.model)
        model_config = self._get_model_config("azure", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("azure", model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        kwargs: dict[str, Any] = {
            "model": self.deployment_id,
            "messages": messages,
            "tools": tools,
        }
        kwargs[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            kwargs["temperature"] = opts["temperature"]

        resp = await self.client.chat.completions.create(**kwargs)

        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", 0)
        total_cost = self._calculate_cost("azure", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp.model_dump(),
            "model_name": model,
            "deployment_id": self.deployment_id,
        }

        choice = resp.choices[0]
        text = choice.message.content or ""
        stop_reason = choice.finish_reason

        tool_calls_out: list[dict[str, Any]] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls_out.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }
