"""Mistral AI driver implementation.

Uses Mistral's OpenAI-compatible Chat Completions API at
``https://api.mistral.ai/v1``. Requires the ``MISTRAL_API_KEY`` env var
(or pass ``api_key=`` explicitly).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .base import Driver, _parse_tool_arguments

logger = logging.getLogger(__name__)


class MistralDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_json_schema = True
    supports_tool_use = True
    supports_streaming = False
    supports_vision = True
    supports_messages = True

    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def __init__(self, api_key: str | None = None, model: str = "mistral-large-latest"):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            from ..exceptions import ConfigurationError

            raise ConfigurationError(
                "MISTRAL_API_KEY is not set. Provide api_key=... or set the "
                "MISTRAL_API_KEY environment variable."
            )
        self.model = model
        self.base_url = "https://api.mistral.ai/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def list_models(cls, *, api_key: str | None = None, timeout: int = 10, **kw: object) -> list[str] | None:
        from .base import _fetch_openai_compatible_models

        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            return None
        return _fetch_openai_compatible_models(
            "https://api.mistral.ai/v1",
            api_key=key,
            timeout=timeout,
        )

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from .vision_helpers import _prepare_openai_vision_messages

        return _prepare_openai_vision_messages(messages)

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(self._prepare_messages(messages), options)

    def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            from ..exceptions import ConfigurationError

            raise ConfigurationError("MISTRAL_API_KEY is not set.")

        model = options.get("model", self.model)
        model_config = self._get_model_config("mistral", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("mistral", model, using_json_schema=bool(options.get("json_schema")))

        opts = {"temperature": 0.7, "max_tokens": 512, **options}

        data: dict[str, Any] = {"model": model, "messages": messages}
        data[tokens_param] = opts.get("max_tokens", 512)
        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

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

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            body = ""
            if e.response is not None:
                with contextlib.suppress(Exception):
                    body = e.response.text
            error_msg = f"Mistral API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            from ..exceptions import DriverError

            err = DriverError(error_msg)
            err.status_code = e.response.status_code if e.response is not None else None
            raise err from e
        except requests.exceptions.RequestException as e:
            from ..exceptions import DriverError

            raise DriverError(f"Mistral API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        total_cost = self._calculate_cost("mistral", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        message = resp["choices"][0]["message"]
        text = message.get("content") or ""
        return {"text": text, "meta": meta}

    def generate_messages_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.api_key:
            from ..exceptions import ConfigurationError

            raise ConfigurationError("MISTRAL_API_KEY is not set.")

        model = options.get("model", self.model)
        model_config = self._get_model_config("mistral", model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities("mistral", model, using_tool_use=True)

        opts = {"temperature": 0.7, "max_tokens": 4096, **options}

        data: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        data[tokens_param] = opts.get("max_tokens", 4096)
        if supports_temperature and "temperature" in opts:
            data["temperature"] = opts["temperature"]

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=data,
                timeout=120,
            )
            response.raise_for_status()
            resp = response.json()
        except requests.exceptions.HTTPError as e:
            body = ""
            if e.response is not None:
                with contextlib.suppress(Exception):
                    body = e.response.text
            error_msg = f"Mistral API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            from ..exceptions import DriverError

            err = DriverError(error_msg)
            err.status_code = e.response.status_code if e.response is not None else None
            raise err from e
        except requests.exceptions.RequestException as e:
            from ..exceptions import DriverError

            raise DriverError(f"Mistral API request failed: {e!s}") from e

        usage = resp.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        total_cost = self._calculate_cost("mistral", model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": resp,
            "model_name": model,
        }

        choice = resp["choices"][0]
        text = choice["message"].get("content") or ""
        stop_reason = choice.get("finish_reason")

        tool_calls_out: list[dict[str, Any]] = []
        for tc in choice["message"].get("tool_calls", []) or []:
            args = _parse_tool_arguments(tc["function"]["arguments"], tc["function"]["name"], stop_reason)
            tool_calls_out.append(
                {
                    "id": tc.get("id", ""),
                    "name": tc["function"]["name"],
                    "arguments": args,
                }
            )

        return {
            "text": text,
            "meta": meta,
            "tool_calls": tool_calls_out,
            "stop_reason": stop_reason,
        }
