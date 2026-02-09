"""Shared cost-calculation mixin for LLM drivers."""

from __future__ import annotations

import copy
from typing import Any


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate USD cost for a model call (public API).

    This is the recommended way for external consumers to calculate
    costs.  Delegates to :meth:`UsageTracker.calculate_cost`.

    Args:
        model_name: Full model identifier (e.g. ``"openai/gpt-4"``).
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of completion/output tokens.

    Returns:
        Estimated cost in USD (rounded to 6 decimal places).
    """
    from .tracker import UsageTracker

    return UsageTracker.calculate_cost(model_name, input_tokens, output_tokens)


def prepare_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prepare a JSON schema for OpenAI strict structured-output mode.

    OpenAI's ``strict: true`` requires every object to have
    ``"additionalProperties": false`` and a ``"required"`` array listing
    all property keys.  This function recursively patches a schema copy
    so callers don't need to worry about these constraints.
    """
    schema = copy.deepcopy(schema)
    _patch_strict(schema)
    return schema


def _patch_strict(node: dict[str, Any]) -> None:
    """Recursively add strict-mode constraints to an object schema node."""
    if node.get("type") == "object" and "properties" in node:
        node.setdefault("additionalProperties", False)
        node.setdefault("required", list(node["properties"].keys()))
        for prop in node["properties"].values():
            _patch_strict(prop)
    elif node.get("type") == "array" and isinstance(node.get("items"), dict):
        _patch_strict(node["items"])


class CostMixin:
    """Mixin that provides ``_calculate_cost`` to sync and async drivers.

    Drivers that charge per-token should inherit from this mixin alongside
    their base class (``Driver`` or ``AsyncDriver``).  Free/local drivers
    (Ollama, LM Studio, LocalHTTP, HuggingFace, AirLLM) can skip it.
    """

    # Subclasses must define MODEL_PRICING as a class attribute.
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    # Divisor for hardcoded MODEL_PRICING rates.
    # Most drivers use per-1K-token pricing (1_000).
    # Grok uses per-1M-token pricing (1_000_000).
    # Google uses per-1M-character pricing (1_000_000).
    _PRICING_UNIT: int = 1_000

    def _calculate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | float,
        completion_tokens: int | float,
    ) -> float:
        """Calculate USD cost for a generation call.

        Resolution order:
        1. Live rates from ``model_rates.get_model_rates()`` (per 1M tokens).
        2. Hardcoded ``MODEL_PRICING`` on the driver class (unit set by ``_PRICING_UNIT``).
        3. Zero if neither source has data.
        """
        from .model_rates import get_model_rates

        live_rates = get_model_rates(provider, model)
        if live_rates and (live_rates.get("input") or live_rates.get("output")):
            prompt_cost = (prompt_tokens / 1_000_000) * live_rates["input"]
            completion_cost = (completion_tokens / 1_000_000) * live_rates["output"]
        else:
            unit = self._PRICING_UNIT
            model_pricing = self.MODEL_PRICING.get(model, {"prompt": 0, "completion": 0})
            prompt_cost = (prompt_tokens / unit) * model_pricing["prompt"]
            completion_cost = (completion_tokens / unit) * model_pricing["completion"]

        return round(prompt_cost + completion_cost, 6)

    def _get_model_config(self, provider: str, model: str) -> dict[str, Any]:
        """Merge live models.dev capabilities with hardcoded ``MODEL_PRICING``.

        Returns a dict with:
        - ``tokens_param`` — always from hardcoded ``MODEL_PRICING`` (API-specific)
        - ``supports_temperature`` — prefers live data, falls back to hardcoded, default ``True``
        - ``context_window`` — from live data only (``None`` if unavailable)
        - ``max_output_tokens`` — from live data only (``None`` if unavailable)
        """
        from .model_rates import get_model_capabilities

        hardcoded = self.MODEL_PRICING.get(model, {})

        # tokens_param is always from hardcoded config (API-specific, not in models.dev)
        tokens_param = hardcoded.get("tokens_param", "max_tokens")

        # Start with hardcoded supports_temperature, default True
        supports_temperature = hardcoded.get("supports_temperature", True)

        context_window: int | None = None
        max_output_tokens: int | None = None

        # Override with live data when available
        caps = get_model_capabilities(provider, model)
        if caps is not None:
            if caps.supports_temperature is not None:
                supports_temperature = caps.supports_temperature
            context_window = caps.context_window
            max_output_tokens = caps.max_output_tokens

        return {
            "tokens_param": tokens_param,
            "supports_temperature": supports_temperature,
            "context_window": context_window,
            "max_output_tokens": max_output_tokens,
        }
