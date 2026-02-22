"""CachiBot.ai proxy driver.

Routes requests through the CachiBot hosted API (OpenAI-compatible).
Uses CACHIBOT_API_KEY env var for authentication.

The proxy's ``/v1/models`` endpoint returns IDs with an upstream provider
prefix (e.g. ``openai/gpt-4o``).  ``list_models()`` strips that prefix so
discovery presents models as ``cachibot/gpt-4o``.  Internally the driver
maintains a mapping to reconstruct the full API model ID for requests and
to resolve the upstream provider for capability / pricing lookups.
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

try:
    import requests as _requests
except Exception:
    _requests = None  # type: ignore[assignment]

from ..infra.cost_mixin import CostMixin
from .base import Driver

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "https://cachibot.ai/api/v1"


class CachiBotDriver(CostMixin, Driver):
    supports_json_mode = True
    supports_tool_use = True
    supports_streaming = True
    supports_vision = True

    # No hardcoded MODEL_PRICING — models are dynamically discovered from the
    # proxy's /v1/models endpoint which already includes pricing.
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    # Populated by list_models(): maps short name → full proxy model ID.
    # e.g. {"gpt-4o": "openai/gpt-4o", "claude-3-5-haiku": "anthropic/claude-3-5-haiku"}
    _MODEL_MAP: dict[str, str] = {}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-4o-mini",
        endpoint: str = _DEFAULT_ENDPOINT,
    ):
        self.api_key = api_key or os.getenv("CACHIBOT_API_KEY")
        self.model = model
        self.base_url = endpoint.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @classmethod
    def list_models(
        cls,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout: int = 10,
        **kw: object,
    ) -> list[str] | None:
        """List models available via the CachiBot proxy API.

        The proxy returns full IDs like ``openai/gpt-4o``.  This method strips
        the upstream provider prefix so discovery presents them as
        ``cachibot/gpt-4o``.  The mapping from short name → full proxy ID is
        stored in ``cls._MODEL_MAP`` for later resolution.
        """
        from .base import _fetch_openai_compatible_models

        key = api_key or os.getenv("CACHIBOT_API_KEY")
        if not key:
            return None
        base = (endpoint or os.getenv("CACHIBOT_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip("/")
        raw_models = _fetch_openai_compatible_models(base, api_key=key, timeout=timeout)
        if raw_models is None:
            return None

        model_map: dict[str, str] = {}
        short_names: list[str] = []
        for full_id in raw_models:
            if "/" in full_id:
                # "openai/gpt-4o" → short="gpt-4o", keep full for API calls
                short = full_id.split("/", 1)[1]
            else:
                short = full_id
            if short in model_map:
                logger.debug(
                    "CachiBot model name collision: %r already mapped to %r, overwriting with %r",
                    short,
                    model_map[short],
                    full_id,
                )
            model_map[short] = full_id
            short_names.append(short)

        cls._MODEL_MAP = model_map
        return short_names

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
        if _requests is None:
            raise RuntimeError("requests package is not installed")
        if not self.api_key:
            raise RuntimeError("CACHIBOT_API_KEY is required")

        model = options.get("model", self.model)
        api_model, upstream_provider, upstream_model = _resolve_model(model)

        model_config = self._get_model_config(upstream_provider, upstream_model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        payload: dict[str, Any] = {
            "model": api_model,
            "messages": messages,
        }
        payload[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        if options.get("json_mode"):
            payload["response_format"] = {"type": "json_object"}

        resp = _requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        total_cost = self._calculate_cost(upstream_provider, upstream_model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": data,
            "model_name": model,
        }

        text = data["choices"][0]["message"].get("content") or ""
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
        if _requests is None:
            raise RuntimeError("requests package is not installed")
        if not self.api_key:
            raise RuntimeError("CACHIBOT_API_KEY is required")

        model = options.get("model", self.model)
        api_model, upstream_provider, upstream_model = _resolve_model(model)

        model_config = self._get_model_config(upstream_provider, upstream_model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        self._validate_model_capabilities(upstream_provider, upstream_model, using_tool_use=True)

        opts = {"temperature": 1.0, "max_tokens": 4096, **options}

        payload: dict[str, Any] = {
            "model": api_model,
            "messages": messages,
            "tools": tools,
        }
        payload[tokens_param] = opts.get("max_tokens", 4096)

        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        resp = _requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        total_cost = self._calculate_cost(upstream_provider, upstream_model, prompt_tokens, completion_tokens)

        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cost": round(total_cost, 6),
            "raw_response": data,
            "model_name": model,
        }

        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        stop_reason = choice.get("finish_reason")

        tool_calls_out: list[dict[str, Any]] = []
        for tc in choice["message"].get("tool_calls", []):
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                raw = tc["function"].get("arguments")
                logger.warning("Failed to parse tool arguments for %s: %r", tc["function"]["name"], raw)
                args = {}
            tool_calls_out.append(
                {
                    "id": tc["id"],
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

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_messages_stream(
        self,
        messages: list[dict[str, Any]],
        options: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        """Yield response chunks via CachiBot streaming API (SSE)."""
        if _requests is None:
            raise RuntimeError("requests package is not installed")
        if not self.api_key:
            raise RuntimeError("CACHIBOT_API_KEY is required")

        model = options.get("model", self.model)
        api_model, upstream_provider, upstream_model = _resolve_model(model)

        model_config = self._get_model_config(upstream_provider, upstream_model)
        tokens_param = model_config["tokens_param"]
        supports_temperature = model_config["supports_temperature"]

        opts = {"temperature": 1.0, "max_tokens": 512, **options}

        payload: dict[str, Any] = {
            "model": api_model,
            "messages": messages,
            "stream": True,
        }
        payload[tokens_param] = opts.get("max_tokens", 512)

        if supports_temperature and "temperature" in opts:
            payload["temperature"] = opts["temperature"]

        resp = _requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()

        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload_str = line[len("data: "):]
            if payload_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    full_text += content
                    yield {"type": "delta", "text": content}

        total_tokens = prompt_tokens + completion_tokens
        total_cost = self._calculate_cost(upstream_provider, upstream_model, prompt_tokens, completion_tokens)

        yield {
            "type": "done",
            "text": full_text,
            "meta": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": round(total_cost, 6),
                "raw_response": {},
                "model_name": model,
            },
        }


def _resolve_model(model: str) -> tuple[str, str, str]:
    """Resolve a model name to (api_model_id, upstream_provider, upstream_model).

    Resolution order:

    1. Check ``CachiBotDriver._MODEL_MAP`` — populated by ``list_models()``.
       e.g. ``"gpt-4o"`` → ``("openai/gpt-4o", "openai", "gpt-4o")``.
    2. If the model already contains ``/`` (legacy full-ID format),
       split directly.
    3. Fallback: treat the model as-is with provider ``"cachibot"``.

    Returns:
        A 3-tuple ``(api_model_id, upstream_provider, upstream_model)``
        where *api_model_id* is sent in the API payload and the other
        two are used for capability / pricing lookups.
    """
    # 1. Model map (short name → full proxy ID)
    full_id = CachiBotDriver._MODEL_MAP.get(model)
    if full_id and "/" in full_id:
        provider, name = full_id.split("/", 1)
        return full_id, provider, name

    # 2. Legacy format: model already has upstream prefix
    if "/" in model:
        provider, name = model.split("/", 1)
        return model, provider, name

    # 3. Unknown — pass through
    return model, "cachibot", model


# Keep backward-compatible alias for external imports
_split_proxy_model = _resolve_model
