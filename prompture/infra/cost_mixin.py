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
    """Recursively add strict-mode constraints to an object schema node.

    Walks ``properties``, ``items``, composite keywords (``anyOf`` /
    ``oneOf`` / ``allOf``), and Pydantic's ``$defs`` registry — the last
    matters because Pydantic v2 puts nested ``BaseModel`` schemas there
    and references them via ``$ref``; without descending into ``$defs``
    those nested objects keep their default open-properties semantics
    and OpenAI's strict mode rejects the whole request with
    ``'additionalProperties' is required to be supplied and to be false``.

    Also strips sibling keywords from ``$ref`` nodes — Pydantic emits
    things like ``{"$ref": "#/$defs/Foo", "description": "..."}`` for
    referenced sub-models, and OpenAI strict mode rejects with
    ``$ref cannot have keywords {'description'}``. Older JSON Schema
    drafts ignored siblings of ``$ref``, so dropping them is lossless
    for our purposes.
    """
    if not isinstance(node, dict):
        return
    if "$ref" in node and len(node) > 1:
        ref_value = node["$ref"]
        node.clear()
        node["$ref"] = ref_value
        return
    if node.get("type") == "object" and "properties" in node:
        node["additionalProperties"] = False
        # Strict mode requires `required` to list *every* property — not
        # just the ones Pydantic considers required (i.e. those without a
        # default). Overwrite rather than setdefault so a partial Pydantic-
        # generated list doesn't slip through and get rejected with
        # "'required' is required to be supplied and to be an array
        # including every key in properties".
        node["required"] = list(node["properties"].keys())
        for prop in node["properties"].values():
            _patch_strict(prop)
    if node.get("type") == "array" and isinstance(node.get("items"), dict):
        _patch_strict(node["items"])
    for keyword in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(keyword, []) or []:
            _patch_strict(sub)
    # Pydantic v2 nested model definitions live here. They're referenced
    # via $ref from properties but never reached by walking properties
    # alone — recurse explicitly.
    for sub in (node.get("$defs") or {}).values():
        _patch_strict(sub)
    for sub in (node.get("definitions") or {}).values():
        _patch_strict(sub)


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
        **kwargs: Any,
    ) -> float:
        """Return the legacy rounded numeric estimate; use details for reporting."""
        return round(
            self._calculate_cost_details(provider, model, prompt_tokens, completion_tokens, **kwargs)["cost"], 6
        )

    def _calculate_cost_details(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | float,
        completion_tokens: int | float,
        *,
        cached_tokens: int | float = 0,
        cache_creation_tokens: int | float = 0,
        cache_write_multiplier: float = 1.0,
        cache_creation_5m_tokens: int | float | None = None,
        cache_creation_1h_tokens: int | float | None = None,
        service_tier: str | None = None,
        inference_geo: str | None = None,
        usage_complete: bool = True,
        tool_usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Estimate USD line items without rounding away small requests.

        ``prompt_tokens`` must include cache reads and cache writes. Detailed
        write counts override the request TTL multiplier. Missing rates remain
        numeric zero for compatibility but are explicitly marked unknown.
        Unsupported billing dimensions produce a partial estimate, with reasons
        in ``pricing.unpriced``. Reasoning tokens are already included in output.
        """
        from .model_rates import PROVIDER_MAP, get_model_rates
        from .pricing import get_model_pricing_rules

        raw_rates = get_model_rates(provider, model)
        available = bool(
            raw_rates is not None and raw_rates.get("input") is not None and raw_rates.get("output") is not None
        )
        rates = dict(raw_rates or {})
        rules = get_model_pricing_rules(provider, model)
        provider_key = PROVIDER_MAP.get(provider, provider)
        unpriced: list[str] = []
        applied: list[str] = []
        if not usage_complete:
            unpriced.append("incomplete_usage")
        tier = service_tier or "default"
        multiplier = 1.0
        if tier not in {"default", "standard", "standard_only"}:
            tier_multiplier = rules.get("service_tiers", {}).get(tier)
            if tier_multiplier is None:
                unpriced.append("service_tier:" + tier)
            else:
                multiplier *= tier_multiplier
                applied.append("service_tier:" + tier)
        if inference_geo and inference_geo != "global":
            geo_multiplier = rules.get("geographies", {}).get(inference_geo)
            # For OpenAI this flag describes a regional processing endpoint.
            if provider_key == "openai" and inference_geo == "regional":
                geo_multiplier = rules.get("regional_multiplier")
            if geo_multiplier is None:
                unpriced.append("inference_geo:" + inference_geo)
            else:
                multiplier *= geo_multiplier
                applied.append("inference_geo:" + inference_geo)
        effective = {key: float(value) * multiplier for key, value in rates.items()}
        long_context = rules.get("long_context", {})
        if prompt_tokens > long_context.get("threshold", float("inf")):
            if tier in {"default", "standard", "batch", "flex"}:
                for key in ("input", "cache_read", "cache_write"):
                    if key in effective:
                        effective[key] *= long_context["input_multiplier"]
                if "output" in effective:
                    effective["output"] *= long_context["output_multiplier"]
                applied.append("long_context")
            else:
                unpriced.append("long_context_service_tier")
        input_rate = effective.get("input", 0.0)
        output_rate = effective.get("output", 0.0)
        read_rate = effective.get("cache_read", input_rate)
        write_rate = effective.get("cache_write", input_rate * (1.25 if provider_key == "anthropic" else 1.0))
        cached = max(0, cached_tokens)
        detailed_writes = cache_creation_5m_tokens is not None or cache_creation_1h_tokens is not None
        if detailed_writes:
            write_5m = max(0, cache_creation_5m_tokens or 0)
            write_1h = max(0, cache_creation_1h_tokens or 0)
            if cache_creation_tokens and write_5m + write_1h != cache_creation_tokens:
                unpriced.append("cache_creation_breakdown_mismatch")
        elif cache_write_multiplier == 1.6:
            write_5m, write_1h = 0, max(0, cache_creation_tokens)
        else:
            write_5m, write_1h = max(0, cache_creation_tokens), 0
            write_rate *= cache_write_multiplier
        created = write_5m + write_1h
        if cached + created > prompt_tokens:
            unpriced.append("inconsistent_prompt_tokens")
        if cached and "cache_read" not in effective:
            unpriced.append("cache_read_rate")
        if created and provider_key == "openai" and "cache_write" not in effective:
            unpriced.append("cache_write_rate")
        non_cached = max(0, prompt_tokens - cached - created)
        breakdown = {
            "uncached_input": non_cached * input_rate / 1_000_000,
            "cache_read": cached * read_rate / 1_000_000,
            "cache_write_5m": write_5m * write_rate / 1_000_000,
            "cache_write_1h": write_1h * (effective.get("cache_write", input_rate * 1.25) * 1.6) / 1_000_000,
            "output": max(0, completion_tokens) * output_rate / 1_000_000,
            "tools": 0.0,
        }
        tool_rates: dict[str, Any] = {}
        for tool, count in (tool_usage or {}).items():
            if not count:
                continue
            if provider_key == "anthropic" and tool == "web_search_requests":
                breakdown["tools"] += max(0, count) * 0.01
            elif provider_key == "anthropic" and tool == "web_fetch_requests":
                continue
            elif provider_key == "openai" and tool == "file_search_call":
                # Responses API tool fee; retrieval tokens are already in usage.
                # Storage is organization-level usage, not a per-call charge.
                # https://developers.openai.com/api/docs/pricing (2026-09-18)
                breakdown["tools"] += max(0, count) * 0.0025
                tool_rates[tool] = 0.0025
            elif provider_key == "openai" and tool == "web_search_requests":
                breakdown["tools"] += max(0, count) * 0.01
                tool_rates[tool] = 0.01
                # Do not add search input tokens a second time. The response
                # does not identify their billing allocation; mini models also
                # have a fixed 8k search-content block rather than token usage.
                unpriced.append("web_search_content_tokens")
            elif provider_key == "openai" and tool == "web_search_preview_calls":
                from .model_rates import get_model_capabilities

                capabilities = get_model_capabilities(provider, model)
                reasoning = getattr(capabilities, "is_reasoning", None)
                if reasoning is True or reasoning is False:
                    fee = 0.01 if reasoning else 0.025
                    breakdown["tools"] += max(0, count) * fee
                    tool_rates[tool] = fee
                    unpriced.append("web_search_content_tokens" if reasoning else "free_search_content_allocation")
                else:
                    unpriced.append("tool:" + tool)
            else:
                unpriced.append("tool:" + tool)
        breakdown["total"] = sum(breakdown.values())
        cache_spend = breakdown["cache_read"] + breakdown["cache_write_5m"] + breakdown["cache_write_1h"]
        return {
            "cost": breakdown["total"],
            "cost_status": "unknown" if not available else ("partial" if unpriced else "estimated"),
            "rates_available": available,
            "usage_complete": usage_complete,
            "pricing": {
                "source": getattr(raw_rates, "source", "unspecified" if available else None),
                "currency": "USD",
                "rates_per_million": rates,
                "effective_rates_per_million": effective,
                "rules": rules,
                "applied_rules": applied,
                "unpriced": unpriced,
                "tool_rates_per_call": tool_rates,
                "tool_rates_source": "https://developers.openai.com/api/docs/pricing" if tool_rates else None,
                "tool_rates_verified_at": "2026-09-18" if tool_rates else None,
            },
            "cost_breakdown": breakdown,
            "cache_savings": (cached + created) * input_rate / 1_000_000 - cache_spend if available else None,
        }

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

        tokens_param = _default_tokens_param(provider, model)
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


def _default_tokens_param(provider: str, model: str) -> str:
    """Pick the per-call output-tokens parameter when the capabilities KB
    has no entry for ``model``.

    OpenAI's GPT-5 family and the o-series reasoning models (o1, o3, o4)
    only accept ``max_completion_tokens`` — sending ``max_tokens`` 400s
    with ``Unsupported parameter``. We name-detect those so brand-new
    model IDs (e.g. ``gpt-5.4-mini``) work without waiting for a KB
    update. Everything else stays on the legacy ``max_tokens`` default.
    """
    if provider != "openai":
        return "max_tokens"
    name = (model or "").split("/")[-1].lower()
    if name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
        return "max_completion_tokens"
    return "max_tokens"


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
            Estimated cost in USD.
        """
        from .model_rates import get_model_rates

        live_rates = get_model_rates(provider, model)
        if live_rates and live_rates.get("input"):
            cost = (total_tokens / 1_000_000) * live_rates["input"]
        else:
            pricing = self.EMBEDDING_PRICING.get(model, {})
            per_million = pricing.get("per_million_tokens", 0.0)
            cost = (total_tokens / 1_000_000) * per_million

        # 12 places, not 6: a typical embedding call is a few dozen tokens,
        # which prices in the e-7 range — round(…, 6) floored every single
        # call to $0.00, so embedding spend never summed to anything.
        return round(cost, 12)


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


class VideoCostMixin:
    """Mixin that provides ``_calculate_video_cost`` to video generation drivers.

    Video generation pricing is typically per-second or per-video. Subclasses
    can define ``VIDEO_PRICING`` with any of these keys:

    ``{"model_id": {"per_second": 0.1, "per_video": 1.0, "default": 1.0}}``.
    Resolution-specific pricing can use
    ``{"per_second_by_resolution": {"720p": 0.07}}``.
    """

    VIDEO_PRICING: dict[str, dict[str, Any]] = {}

    def _calculate_video_cost(
        self,
        provider: str,
        model: str,
        *,
        duration_seconds: float = 0,
        n: int = 1,
        resolution: str | None = None,
    ) -> float:
        """Calculate USD cost for a video generation call.

        Lookup order: resolution-specific ``per_second`` × duration →
        generic ``per_second`` × duration → ``per_video`` → ``default`` → 0.
        """
        pricing = self.VIDEO_PRICING.get(model, {})

        per_second_by_resolution = pricing.get("per_second_by_resolution")
        if duration_seconds > 0 and resolution and isinstance(per_second_by_resolution, dict):
            rate = per_second_by_resolution.get(resolution)
            if rate is not None:
                return round(float(rate) * duration_seconds * n, 6)

        if duration_seconds > 0 and "per_second" in pricing:
            return round(float(pricing["per_second"]) * duration_seconds * n, 6)

        per_video = pricing.get("per_video", pricing.get("default", 0.0))
        return round(float(per_video) * n, 6)
