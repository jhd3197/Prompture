"""Discovery module for auto-detecting available models."""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, overload

import requests

from .drivers import (
    AzureDriver,
    ClaudeDriver,
    GoogleDriver,
    GrokDriver,
    GroqDriver,
    LMStudioDriver,
    LocalHTTPDriver,
    OllamaDriver,
    OpenAIDriver,
    OpenRouterDriver,
)
from .settings import settings

logger = logging.getLogger(__name__)


@overload
def get_available_models(*, include_capabilities: bool = False) -> list[str]: ...


@overload
def get_available_models(*, include_capabilities: bool = True) -> list[dict[str, Any]]: ...


def get_available_models(*, include_capabilities: bool = False) -> list[str] | list[dict[str, Any]]:
    """Auto-detect available models based on configured drivers and environment variables.

    Iterates through supported providers and checks if they are configured
    (e.g. API key present).  For static drivers, returns models from their
    ``MODEL_PRICING`` keys.  For dynamic drivers (like Ollama), attempts to
    fetch available models from the endpoint.

    Args:
        include_capabilities: When ``True``, return enriched dicts with
            ``model``, ``provider``, ``model_id``, and ``capabilities``
            fields instead of plain ``"provider/model_id"`` strings.

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
    }

    for provider, driver_cls in provider_classes.items():
        try:
            is_configured = False

            if provider == "openai":
                if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
                    is_configured = True
            elif provider == "azure":
                if (
                    (settings.azure_api_key or os.getenv("AZURE_API_KEY"))
                    and (settings.azure_api_endpoint or os.getenv("AZURE_API_ENDPOINT"))
                    and (settings.azure_deployment_id or os.getenv("AZURE_DEPLOYMENT_ID"))
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
            elif provider == "ollama":
                is_configured = True
            elif provider == "lmstudio":
                is_configured = True
            elif provider == "local_http" and os.getenv("LOCAL_HTTP_ENDPOINT"):
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

            # Dynamic Detection: Specific logic for Ollama
            if provider == "ollama":
                try:
                    endpoint = settings.ollama_endpoint or os.getenv(
                        "OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"
                    )
                    base_url = endpoint.split("/api/")[0]
                    tags_url = f"{base_url}/api/tags"

                    resp = requests.get(tags_url, timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("models", [])
                        for model in models:
                            name = model.get("name")
                            if name:
                                available_models.add(f"ollama/{name}")
                except Exception as e:
                    logger.debug(f"Failed to fetch Ollama models: {e}")

            # Dynamic Detection: LM Studio loaded models
            if provider == "lmstudio":
                try:
                    endpoint = settings.lmstudio_endpoint or os.getenv(
                        "LMSTUDIO_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions"
                    )
                    base_url = endpoint.split("/v1/")[0]
                    models_url = f"{base_url}/v1/models"

                    headers: dict[str, str] = {}
                    api_key = settings.lmstudio_api_key or os.getenv("LMSTUDIO_API_KEY")
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"

                    resp = requests.get(models_url, headers=headers, timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = data.get("data", [])
                        for model in models:
                            model_id = model.get("id")
                            if model_id:
                                available_models.add(f"lmstudio/{model_id}")
                except Exception as e:
                    logger.debug(f"Failed to fetch LM Studio models: {e}")

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

    if not include_capabilities:
        return sorted_models

    # Build enriched dicts with capabilities from models.dev
    from .model_rates import get_model_capabilities

    enriched: list[dict[str, Any]] = []
    for model_str in sorted_models:
        parts = model_str.split("/", 1)
        provider = parts[0]
        model_id = parts[1] if len(parts) > 1 else parts[0]

        caps = get_model_capabilities(provider, model_id)
        caps_dict = dataclasses.asdict(caps) if caps is not None else None

        enriched.append(
            {
                "model": model_str,
                "provider": provider,
                "model_id": model_id,
                "capabilities": caps_dict,
            }
        )

    return enriched
