"""Generic OpenAI-compatible driver.

Many vendors expose an OpenAI-compatible Chat Completions API at their own
base URL. This driver makes it trivial to talk to any of them without
writing a dedicated driver.

Usage:

    # Via a curated profile (preferred — pulls endpoint + env var from
    # OPENAI_COMPATIBLE_PROFILES below):
    drv = OpenAICompatibleDriver(profile="fireworks",
                                 model="accounts/fireworks/models/llama-v3p1-70b-instruct")

    # Via an explicit endpoint (for anything not in the profile table):
    drv = OpenAICompatibleDriver(endpoint="https://my-host.example.com/v1",
                                 api_key="sk-...", model="my-model")

Model strings: registry-routed calls look like
``"openai_compatible/<profile>/<model>"`` or
``"openai_compatible/<model>"`` (the latter requires an explicit
``endpoint=`` override). The registry strips the leading
``"openai_compatible/"`` and passes the rest as ``model``; this driver
parses ``"<profile>/<model>"`` automatically by checking the first
segment against ``OPENAI_COMPATIBLE_PROFILES``.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any

import requests

from ..infra.cost_mixin import CostMixin, prepare_strict_schema
from .base import Driver

logger = logging.getLogger(__name__)


# Curated table of well-known OpenAI-compatible endpoints.
# Profile name → {endpoint, env_var (for default API key)}.
OPENAI_COMPATIBLE_PROFILES: dict[str, dict[str, str]] = {
    "fireworks": {
        "endpoint": "https://api.fireworks.ai/inference/v1",
        "env_var": "FIREWORKS_API_KEY",
    },
    "together": {
        "endpoint": "https://api.together.xyz/v1",
        "env_var": "TOGETHER_API_KEY",
    },
    "cerebras": {
        "endpoint": "https://api.cerebras.ai/v1",
        "env_var": "CEREBRAS_API_KEY",
    },
    "sambanova": {
        "endpoint": "https://api.sambanova.ai/v1",
        "env_var": "SAMBANOVA_API_KEY",
    },
    "perplexity": {
        "endpoint": "https://api.perplexity.ai",
        "env_var": "PERPLEXITY_API_KEY",
    },
    "nvidia": {
        "endpoint": "https://integrate.api.nvidia.com/v1",
        "env_var": "NVIDIA_API_KEY",
    },
    "deepinfra": {
        "endpoint": "https://api.deepinfra.com/v1/openai",
        "env_var": "DEEPINFRA_API_KEY",
    },
    "siliconflow": {
        "endpoint": "https://api.siliconflow.cn/v1",
        "env_var": "SILICONFLOW_API_KEY",
    },
    "github_models": {
        "endpoint": "https://models.github.ai/inference",
        "env_var": "GITHUB_TOKEN",  # nosec B105 — env var name, not a credential
    },
}


def _parse_profile_and_model(model: str) -> tuple[str | None, str]:
    """Split a model string of the form ``"<profile>/<model_id>"``.

    Returns ``(profile, model_id)``. If the first ``/``-separated segment
    isn't a known profile, returns ``(None, model)``.
    """
    if not model or "/" not in model:
        return None, model
    head, rest = model.split("/", 1)
    if head in OPENAI_COMPATIBLE_PROFILES:
        return head, rest
    return None, model


class OpenAICompatibleDriver(CostMixin, Driver):
    """Generic driver for any OpenAI-compatible Chat Completions endpoint."""

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
        # If model is "profile/model_id" and no explicit profile was given,
        # auto-detect the profile from the first segment.
        if profile is None and endpoint is None:
            detected_profile, parsed_model = _parse_profile_and_model(model)
            if detected_profile is not None:
                profile = detected_profile
                model = parsed_model

        if profile is not None and profile not in OPENAI_COMPATIBLE_PROFILES:
            raise ValueError(
                f"Unknown openai_compatible profile {profile!r}. Known: {sorted(OPENAI_COMPATIBLE_PROFILES)}"
            )

        # Resolve endpoint: explicit endpoint > profile lookup > error.
        resolved_endpoint: str | None = endpoint
        env_var: str | None = None
        if resolved_endpoint is None and profile is not None:
            resolved_endpoint = OPENAI_COMPATIBLE_PROFILES[profile]["endpoint"]
            env_var = OPENAI_COMPATIBLE_PROFILES[profile]["env_var"]

        if not resolved_endpoint:
            raise ValueError(
                "OpenAICompatibleDriver requires either a `profile` (one of "
                f"{sorted(OPENAI_COMPATIBLE_PROFILES)}) or an explicit "
                "`endpoint`."
            )

        # Resolve API key: explicit > profile env var > generic fallback.
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

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "user", "content": prompt}]
        return self._do_generate(messages, options)

    def generate_messages(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        return self._do_generate(messages, options)

    def _do_generate(self, messages: list[dict[str, str]], options: dict[str, Any]) -> dict[str, Any]:
        # Per-call overrides take precedence over instance values.
        endpoint = (options.get("endpoint") or self.endpoint).rstrip("/")
        api_key = options.get("api_key") or self.api_key
        model = options.get("model", self.model)

        # Capabilities lookup uses the profile name (so e.g. "fireworks"
        # finds entries in rates/fireworks.json). For explicit-endpoint
        # mode we use "openai_compatible".
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
        # Safe on every OpenAI-compatible server — unrecognised keys are ignored.
        apply_guided_decoding(
            data,
            json_schema=options.get("json_schema"),
            guided_decoding=options.get("guided_decoding"),
        )
        # Generic vendor escape hatch (mirrors openai SDK's extra_body).
        merge_extra_body(data, options)

        try:
            response = requests.post(
                f"{endpoint}/chat/completions",
                headers=self._headers(api_key),
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
            error_msg = f"OpenAI-compatible API request failed: {e!s}"
            if body:
                error_msg += f"\nResponse: {body}"
            raise RuntimeError(error_msg) from e
        except requests.exceptions.RequestException as e:
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
