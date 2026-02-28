"""Shared cost-calculation mixin for LLM drivers."""

from __future__ import annotations

import copy
from typing import Any


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

    # Kept as an empty dict for backward compatibility (external code may
    # reference ``driver.MODEL_PRICING``).  No longer used for cost
    # calculation or config lookup — all data comes from the capabilities
    # knowledge base (JSON rate files) and models.dev live data.
    MODEL_PRICING: dict[str, dict[str, Any]] = {}

    def _calculate_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | float,
        completion_tokens: int | float,
    ) -> float:
        """Calculate USD cost for a generation call.

        Uses live rates from ``model_rates.get_model_rates()`` (per 1M tokens).
        Returns 0.0 if no rates are available.
        """
        from .model_rates import get_model_rates

        live_rates = get_model_rates(provider, model)
        if live_rates and (live_rates.get("input") or live_rates.get("output")):
            prompt_cost = (prompt_tokens / 1_000_000) * live_rates["input"]
            completion_cost = (completion_tokens / 1_000_000) * live_rates["output"]
            return round(prompt_cost + completion_cost, 6)

        return 0.0

    def _get_model_config(self, provider: str, model: str) -> dict[str, Any]:
        """Return per-model configuration from capabilities knowledge base.

        Returns a dict with:
        - ``tokens_param`` — from KB / models.dev, default ``"max_tokens"``
        - ``supports_temperature`` — from KB / models.dev, default ``True``
        - ``context_window`` — from KB / models.dev (``None`` if unavailable)
        - ``max_output_tokens`` — from KB / models.dev (``None`` if unavailable)
        """
        from .model_rates import get_model_capabilities

        caps = get_model_capabilities(provider, model)

        tokens_param = "max_tokens"
        supports_temperature = True
        context_window: int | None = None
        max_output_tokens: int | None = None

        if caps is not None:
            if caps.tokens_param is not None:
                tokens_param = caps.tokens_param
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


class AudioCostMixin:
    """Mixin that provides ``_calculate_audio_cost`` to STT and TTS drivers.

    Audio pricing differs from LLM pricing: STT is typically per-second of
    audio, while TTS is per-character of input text.
    """

    # Subclasses should define AUDIO_PRICING as a class attribute.
    # Format: {"model_id": {"per_second": float, "per_character": float}}
    AUDIO_PRICING: dict[str, dict[str, float]] = {}

    def _calculate_audio_cost(
        self,
        provider: str,
        model: str,
        *,
        duration_seconds: float = 0,
        characters: int = 0,
    ) -> float:
        """Calculate USD cost for an audio API call.

        Args:
            provider: Provider name (e.g. ``"openai"``, ``"elevenlabs"``).
            model: Model identifier (e.g. ``"whisper-1"``, ``"tts-1"``).
            duration_seconds: Audio duration in seconds (for STT).
            characters: Number of text characters (for TTS).

        Returns:
            Estimated cost in USD, rounded to 6 decimal places.
        """
        pricing = self.AUDIO_PRICING.get(model, {})

        cost = 0.0
        if duration_seconds > 0 and "per_second" in pricing:
            cost += duration_seconds * pricing["per_second"]
        if characters > 0 and "per_character" in pricing:
            cost += characters * pricing["per_character"]

        return round(cost, 6)


class EmbeddingCostMixin:
    """Mixin that provides ``_calculate_embedding_cost`` to embedding drivers.

    Embedding pricing is typically per-million input tokens (no output tokens).
    """

    # Subclasses should define EMBEDDING_PRICING as a class attribute.
    # Format: {"model_id": {"per_million_tokens": float}}
    EMBEDDING_PRICING: dict[str, dict[str, float]] = {}

    def _calculate_embedding_cost(
        self,
        provider: str,
        model: str,
        *,
        total_tokens: int = 0,
    ) -> float:
        """Calculate USD cost for an embedding API call.

        Resolution order:
        1. Live rates from ``model_rates.get_model_rates()`` (per 1M tokens, input only).
        2. Hardcoded ``EMBEDDING_PRICING`` on the driver class.
        3. Zero if neither source has data.

        Args:
            provider: Provider name (e.g. ``"openai"``).
            model: Model identifier (e.g. ``"text-embedding-3-small"``).
            total_tokens: Total number of input tokens processed.

        Returns:
            Estimated cost in USD, rounded to 6 decimal places.
        """
        from .model_rates import get_model_rates

        live_rates = get_model_rates(provider, model)
        if live_rates and live_rates.get("input"):
            cost = (total_tokens / 1_000_000) * live_rates["input"]
        else:
            pricing = self.EMBEDDING_PRICING.get(model, {})
            per_million = pricing.get("per_million_tokens", 0.0)
            cost = (total_tokens / 1_000_000) * per_million

        return round(cost, 6)


class ImageCostMixin:
    """Mixin that provides ``_calculate_image_cost`` to image generation drivers.

    Image generation pricing is typically per-image, varying by size and quality.
    """

    # Subclasses should define IMAGE_PRICING as a class attribute.
    # Format: {"model_id": {"size/quality": float_per_image, ...}}
    # e.g. {"dall-e-3": {"1024x1024/standard": 0.04, "1024x1024/hd": 0.08}}
    IMAGE_PRICING: dict[str, dict[str, float]] = {}

    def _calculate_image_cost(
        self,
        provider: str,
        model: str,
        *,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1,
    ) -> float:
        """Calculate USD cost for an image generation call.

        Lookup order: ``"{size}/{quality}"`` → ``"{size}"`` → ``"default"`` → 0.

        Args:
            provider: Provider name (e.g. ``"openai"``, ``"stability"``).
            model: Model identifier (e.g. ``"dall-e-3"``).
            size: Image dimensions (e.g. ``"1024x1024"``).
            quality: Quality tier (e.g. ``"standard"``, ``"hd"``).
            n: Number of images generated.

        Returns:
            Estimated cost in USD, rounded to 6 decimal places.
        """
        pricing = self.IMAGE_PRICING.get(model, {})

        per_image = pricing.get(f"{size}/{quality}") or pricing.get(size) or pricing.get("default", 0.0)

        return round(per_image * n, 6)
