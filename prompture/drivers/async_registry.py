"""Async driver registry — mirrors the sync DRIVER_REGISTRY."""

from __future__ import annotations

from ..settings import settings
from .async_airllm_driver import AsyncAirLLMDriver
from .async_azure_driver import AsyncAzureDriver
from .async_claude_driver import AsyncClaudeDriver
from .async_google_driver import AsyncGoogleDriver
from .async_grok_driver import AsyncGrokDriver
from .async_groq_driver import AsyncGroqDriver
from .async_lmstudio_driver import AsyncLMStudioDriver
from .async_local_http_driver import AsyncLocalHTTPDriver
from .async_ollama_driver import AsyncOllamaDriver
from .async_openai_driver import AsyncOpenAIDriver
from .async_openrouter_driver import AsyncOpenRouterDriver

ASYNC_DRIVER_REGISTRY = {
    "openai": lambda model=None: AsyncOpenAIDriver(
        api_key=settings.openai_api_key, model=model or settings.openai_model
    ),
    "ollama": lambda model=None: AsyncOllamaDriver(
        endpoint=settings.ollama_endpoint, model=model or settings.ollama_model
    ),
    "claude": lambda model=None: AsyncClaudeDriver(
        api_key=settings.claude_api_key, model=model or settings.claude_model
    ),
    "lmstudio": lambda model=None: AsyncLMStudioDriver(
        endpoint=settings.lmstudio_endpoint, model=model or settings.lmstudio_model
    ),
    "azure": lambda model=None: AsyncAzureDriver(
        api_key=settings.azure_api_key, endpoint=settings.azure_api_endpoint, deployment_id=settings.azure_deployment_id
    ),
    "local_http": lambda model=None: AsyncLocalHTTPDriver(endpoint=settings.local_http_endpoint, model=model),
    "google": lambda model=None: AsyncGoogleDriver(
        api_key=settings.google_api_key, model=model or settings.google_model
    ),
    "groq": lambda model=None: AsyncGroqDriver(api_key=settings.groq_api_key, model=model or settings.groq_model),
    "openrouter": lambda model=None: AsyncOpenRouterDriver(
        api_key=settings.openrouter_api_key, model=model or settings.openrouter_model
    ),
    "grok": lambda model=None: AsyncGrokDriver(api_key=settings.grok_api_key, model=model or settings.grok_model),
    "airllm": lambda model=None: AsyncAirLLMDriver(
        model=model or settings.airllm_model,
        compression=settings.airllm_compression,
    ),
}


def get_async_driver(provider_name: str | None = None):
    """Factory to get an async driver instance based on the provider name.

    Uses default model from settings if not overridden.
    """
    provider = (provider_name or settings.ai_provider or "ollama").strip().lower()
    if provider not in ASYNC_DRIVER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_name}")
    return ASYNC_DRIVER_REGISTRY[provider]()


def get_async_driver_for_model(model_str: str):
    """Factory to get an async driver instance based on a full model string.

    Format: ``provider/model_id``
    Example: ``"openai/gpt-4-turbo-preview"``
    """
    if not isinstance(model_str, str):
        raise ValueError("Model string must be a string, got {type(model_str)}")

    if not model_str:
        raise ValueError("Model string cannot be empty")

    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None

    if provider not in ASYNC_DRIVER_REGISTRY:
        raise ValueError(f"Unsupported provider '{provider}'")

    return ASYNC_DRIVER_REGISTRY[provider](model_id)
