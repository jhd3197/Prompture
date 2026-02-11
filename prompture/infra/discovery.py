"""Discovery module for auto-detecting available models."""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, overload

from ..drivers import (
    AirLLMDriver,
    AzureDriver,
    ClaudeDriver,
    ElevenLabsSTTDriver,
    ElevenLabsTTSDriver,
    GoogleDriver,
    GoogleImageGenDriver,
    GrokDriver,
    GrokImageGenDriver,
    GroqDriver,
    LMStudioDriver,
    LocalHTTPDriver,
    ModelScopeDriver,
    MoonshotDriver,
    OllamaDriver,
    OpenAIDriver,
    OpenAIImageGenDriver,
    OpenAISTTDriver,
    OpenAITTSDriver,
    OpenRouterDriver,
    StabilityImageGenDriver,
    ZaiDriver,
)
from .settings import settings

logger = logging.getLogger(__name__)


def _get_list_models_kwargs(provider: str) -> dict[str, Any]:
    """Return keyword arguments for ``driver_cls.list_models()`` based on provider config."""
    kw: dict[str, Any] = {}
    if provider == "openai":
        kw["api_key"] = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
    elif provider == "claude":
        kw["api_key"] = settings.claude_api_key or os.getenv("CLAUDE_API_KEY")
    elif provider == "google":
        kw["api_key"] = settings.google_api_key or os.getenv("GOOGLE_API_KEY")
    elif provider == "groq":
        kw["api_key"] = settings.groq_api_key or os.getenv("GROQ_API_KEY")
    elif provider == "grok":
        kw["api_key"] = settings.grok_api_key or os.getenv("GROK_API_KEY")
    elif provider == "openrouter":
        kw["api_key"] = settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    elif provider == "moonshot":
        kw["api_key"] = settings.moonshot_api_key or os.getenv("MOONSHOT_API_KEY")
        kw["endpoint"] = settings.moonshot_endpoint or os.getenv("MOONSHOT_ENDPOINT")
    elif provider == "ollama":
        kw["endpoint"] = settings.ollama_endpoint or os.getenv("OLLAMA_ENDPOINT")
    elif provider == "lmstudio":
        kw["endpoint"] = settings.lmstudio_endpoint or os.getenv("LMSTUDIO_ENDPOINT")
        kw["api_key"] = settings.lmstudio_api_key or os.getenv("LMSTUDIO_API_KEY")
    return kw


@overload
def get_available_models(*, include_capabilities: bool = False, verified_only: bool = False) -> list[str]: ...


@overload
def get_available_models(*, include_capabilities: bool = True, verified_only: bool = False) -> list[dict[str, Any]]: ...


def get_available_models(
    *,
    include_capabilities: bool = False,
    verified_only: bool = False,
) -> list[str] | list[dict[str, Any]]:
    """Auto-detect available models based on configured drivers and environment variables.

    Iterates through supported providers and checks if they are configured
    (e.g. API key present).  For static drivers, returns models from their
    ``MODEL_PRICING`` keys.  For dynamic drivers (like Ollama), attempts to
    fetch available models from the endpoint.

    Args:
        include_capabilities: When ``True``, return enriched dicts with
            ``model``, ``provider``, ``model_id``, and ``capabilities``
            fields instead of plain ``"provider/model_id"`` strings.
        verified_only: When ``True``, only return models that have been
            successfully used (as recorded by the usage ledger).

    Returns:
        A sorted list of unique model strings (default) or enriched dicts.
    """
    available_models: set[str] = set()
    configured_providers: set[str] = set()

    # Map of provider name to driver class
    provider_classes = {
        "openai": OpenAIDriver,
        "azure": AzureDriver,
        "claude": ClaudeDriver,
        "google": GoogleDriver,
        "groq": GroqDriver,
        "openrouter": OpenRouterDriver,
        "grok": GrokDriver,
        "ollama": OllamaDriver,
        "lmstudio": LMStudioDriver,
        "local_http": LocalHTTPDriver,
        "moonshot": MoonshotDriver,
        "zai": ZaiDriver,
        "modelscope": ModelScopeDriver,
        "airllm": AirLLMDriver,
    }

    for provider, driver_cls in provider_classes.items():
        try:
            is_configured = False

            if provider == "openai":
                if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
                    is_configured = True
            elif provider == "azure":
                from ..drivers.azure_config import has_azure_config_resolver, has_registered_configs

                if (
                    (
                        (settings.azure_api_key or os.getenv("AZURE_API_KEY"))
                        and (settings.azure_api_endpoint or os.getenv("AZURE_API_ENDPOINT"))
                    )
                    or (settings.azure_claude_api_key or os.getenv("AZURE_CLAUDE_API_KEY"))
                    or (settings.azure_mistral_api_key or os.getenv("AZURE_MISTRAL_API_KEY"))
                    or has_registered_configs()
                    or has_azure_config_resolver()
                ):
                    is_configured = True
            elif provider == "claude":
                if settings.claude_api_key or os.getenv("CLAUDE_API_KEY"):
                    is_configured = True
            elif provider == "google":
                if settings.google_api_key or os.getenv("GOOGLE_API_KEY"):
                    is_configured = True
            elif provider == "groq":
                if settings.groq_api_key or os.getenv("GROQ_API_KEY"):
                    is_configured = True
            elif provider == "openrouter":
                if settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"):
                    is_configured = True
            elif provider == "grok":
                if settings.grok_api_key or os.getenv("GROK_API_KEY"):
                    is_configured = True
            elif provider == "moonshot":
                if settings.moonshot_api_key or os.getenv("MOONSHOT_API_KEY"):
                    is_configured = True
            elif provider == "zai":
                if settings.zhipu_api_key or os.getenv("ZHIPU_API_KEY"):
                    is_configured = True
            elif provider == "modelscope":
                if settings.modelscope_api_key or os.getenv("MODELSCOPE_API_KEY"):
                    is_configured = True
            elif provider == "airllm":
                # AirLLM runs locally, always considered configured
                is_configured = True
            elif (
                provider == "ollama"
                or provider == "lmstudio"
                or (provider == "local_http" and os.getenv("LOCAL_HTTP_ENDPOINT"))
            ):
                is_configured = True

            if not is_configured:
                continue

            configured_providers.add(provider)

            # Static Detection: Get models from MODEL_PRICING
            if hasattr(driver_cls, "MODEL_PRICING"):
                pricing = driver_cls.MODEL_PRICING
                for model_id in pricing:
                    if model_id == "default":
                        continue
                    available_models.add(f"{provider}/{model_id}")

            # API-based Detection: call driver's list_models() classmethod
            api_kwargs = _get_list_models_kwargs(provider)
            try:
                api_models = driver_cls.list_models(**api_kwargs)
                if api_models is not None:
                    for model_id in api_models:
                        available_models.add(f"{provider}/{model_id}")
            except Exception as e:
                logger.debug("list_models() failed for %s: %s", provider, e)

        except Exception as e:
            logger.warning(f"Error detecting models for provider {provider}: {e}")
            continue

    # Enrich with live model list from models.dev cache
    from .model_rates import PROVIDER_MAP, get_all_provider_models

    for prompture_name, api_name in PROVIDER_MAP.items():
        if prompture_name in configured_providers:
            for model_id in get_all_provider_models(api_name):
                available_models.add(f"{prompture_name}/{model_id}")

    sorted_models = sorted(available_models)

    # --- verified_only filtering ---
    verified_set: set[str] | None = None
    if verified_only or include_capabilities:
        try:
            from .ledger import _get_ledger

            ledger = _get_ledger()
            verified_set = ledger.get_verified_models()
        except Exception:
            logger.debug("Could not load ledger for verified models", exc_info=True)
            verified_set = set()

    if verified_only and verified_set is not None:
        sorted_models = [m for m in sorted_models if m in verified_set]

    if not include_capabilities:
        return sorted_models

    # Build enriched dicts with capabilities from models.dev
    from .model_rates import get_model_capabilities

    # Fetch all ledger stats for annotation (keyed by model_name)
    ledger_stats: dict[str, dict[str, Any]] = {}
    try:
        from .ledger import _get_ledger

        for row in _get_ledger().get_all_stats():
            name = row["model_name"]
            if name not in ledger_stats:
                ledger_stats[name] = row
            else:
                # Aggregate across API key hashes
                existing = ledger_stats[name]
                existing["use_count"] += row["use_count"]
                existing["total_tokens"] += row["total_tokens"]
                existing["total_cost"] += row["total_cost"]
                if row["last_used"] > existing["last_used"]:
                    existing["last_used"] = row["last_used"]
    except Exception:
        logger.debug("Could not load ledger stats for enrichment", exc_info=True)

    enriched: list[dict[str, Any]] = []
    for model_str in sorted_models:
        parts = model_str.split("/", 1)
        provider = parts[0]
        model_id = parts[1] if len(parts) > 1 else parts[0]

        caps = get_model_capabilities(provider, model_id)
        caps_dict = dataclasses.asdict(caps) if caps is not None else None

        entry: dict[str, Any] = {
            "model": model_str,
            "provider": provider,
            "model_id": model_id,
            "capabilities": caps_dict,
            "verified": verified_set is not None and model_str in verified_set,
        }

        stats = ledger_stats.get(model_str)
        if stats:
            entry["last_used"] = stats["last_used"]
            entry["use_count"] = stats["use_count"]
        else:
            entry["last_used"] = None
            entry["use_count"] = 0

        enriched.append(entry)

    return enriched


def get_available_audio_models(
    *,
    modality: str | None = None,
) -> list[str]:
    """Auto-detect available audio models (STT and TTS) based on configured API keys.

    Checks which audio providers are configured and returns their supported models.

    Args:
        modality: Filter by ``"stt"`` or ``"tts"``. Returns both when ``None``.

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/whisper-1"``).
    """
    available: set[str] = set()

    # OpenAI audio models (requires same openai_api_key as LLM)
    if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
        if modality is None or modality == "stt":
            for model_id in OpenAISTTDriver.AUDIO_PRICING:
                available.add(f"openai/{model_id}")
        if modality is None or modality == "tts":
            for model_id in OpenAITTSDriver.AUDIO_PRICING:
                available.add(f"openai/{model_id}")

    # ElevenLabs audio models
    elevenlabs_key = getattr(settings, "elevenlabs_api_key", None) or os.getenv("ELEVENLABS_API_KEY")
    elevenlabs_endpoint = (
        getattr(settings, "elevenlabs_endpoint", None) or "https://api.elevenlabs.io/v1"
    )
    if elevenlabs_key:
        if modality is None or modality == "stt":
            # STT: static list (no API listing endpoint for STT models)
            stt_models = ElevenLabsSTTDriver.list_models(
                api_key=elevenlabs_key, endpoint=elevenlabs_endpoint
            )
            if stt_models:
                for model_id in stt_models:
                    available.add(f"elevenlabs/{model_id}")

        if modality is None or modality == "tts":
            # TTS: dynamic discovery via GET /v1/models, fallback to AUDIO_PRICING
            tts_models = ElevenLabsTTSDriver.list_models(
                api_key=elevenlabs_key, endpoint=elevenlabs_endpoint
            )
            if tts_models is not None:
                for model_id in tts_models:
                    available.add(f"elevenlabs/{model_id}")
            else:
                for model_id in ElevenLabsTTSDriver.AUDIO_PRICING:
                    available.add(f"elevenlabs/{model_id}")

    return sorted(available)


def get_available_image_gen_models() -> list[str]:
    """Auto-detect available image generation models based on configured API keys.

    Checks which image gen providers are configured and returns their supported models.

    Returns:
        A sorted list of unique model strings (e.g. ``"openai/dall-e-3"``).
    """
    available: set[str] = set()

    # OpenAI image gen models (requires openai_api_key)
    if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
        for model_id in OpenAIImageGenDriver.IMAGE_PRICING:
            available.add(f"openai/{model_id}")

    # Google image gen models (requires google_api_key)
    if settings.google_api_key or os.getenv("GOOGLE_API_KEY"):
        for model_id in GoogleImageGenDriver.IMAGE_PRICING:
            available.add(f"google/{model_id}")

    # Stability AI image gen models
    stability_key = getattr(settings, "stability_api_key", None) or os.getenv("STABILITY_API_KEY")
    if stability_key:
        for model_id in StabilityImageGenDriver.IMAGE_PRICING:
            available.add(f"stability/{model_id}")

    # Grok/xAI image gen models (requires grok_api_key)
    if settings.grok_api_key or os.getenv("GROK_API_KEY"):
        for model_id in GrokImageGenDriver.IMAGE_PRICING:
            available.add(f"grok/{model_id}")

    return sorted(available)
