"""Shared cost-calculation mixin for LLM drivers."""

from __future__ import annotations

from typing import Any


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
        if live_rates:
            prompt_cost = (prompt_tokens / 1_000_000) * live_rates["input"]
            completion_cost = (completion_tokens / 1_000_000) * live_rates["output"]
        else:
            unit = self._PRICING_UNIT
            model_pricing = self.MODEL_PRICING.get(model, {"prompt": 0, "completion": 0})
            prompt_cost = (prompt_tokens / unit) * model_pricing["prompt"]
            completion_cost = (completion_tokens / unit) * model_pricing["completion"]

        return round(prompt_cost + completion_cost, 6)
