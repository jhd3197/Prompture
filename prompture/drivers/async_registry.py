"""Async driver registry — mirrors the sync DRIVER_REGISTRY.

This module provides async driver registration and factory functions.
Custom async drivers can be registered via the ``register_async_driver()``
function or discovered via entry points.

Entry Point Discovery:
    Add to your pyproject.toml:

    [project.entry-points."prompture.async_drivers"]
    my_provider = "my_package.drivers:my_async_driver_factory"
"""

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
from .async_modelscope_driver import AsyncModelScopeDriver
from .async_moonshot_driver import AsyncMoonshotDriver
from .async_ollama_driver import AsyncOllamaDriver
from .async_openai_driver import AsyncOpenAIDriver
from .async_openrouter_driver import AsyncOpenRouterDriver
from .async_zai_driver import AsyncZaiDriver
from .registry import (
    _get_async_registry,
    get_async_driver_factory,
    register_async_driver,
)

# Register built-in async drivers
register_async_driver(
    "openai",
    lambda model=None: AsyncOpenAIDriver(api_key=settings.openai_api_key, model=model or settings.openai_model),
    overwrite=True,
)
register_async_driver(
    "ollama",
    lambda model=None: AsyncOllamaDriver(endpoint=settings.ollama_endpoint, model=model or settings.ollama_model),
    overwrite=True,
)
register_async_driver(
    "claude",
    lambda model=None: AsyncClaudeDriver(api_key=settings.claude_api_key, model=model or settings.claude_model),
    overwrite=True,
)
register_async_driver(
    "lmstudio",
    lambda model=None: AsyncLMStudioDriver(
        endpoint=settings.lmstudio_endpoint,
        model=model or settings.lmstudio_model,
        api_key=settings.lmstudio_api_key,
    ),
    overwrite=True,
)
register_async_driver(
    "azure",
    lambda model=None: AsyncAzureDriver(
        api_key=settings.azure_api_key,
        endpoint=settings.azure_api_endpoint,
        deployment_id=settings.azure_deployment_id,
        model=model or "gpt-4o-mini",
    ),
    overwrite=True,
)
register_async_driver(
    "local_http",
    lambda model=None: AsyncLocalHTTPDriver(endpoint=settings.local_http_endpoint, model=model),
    overwrite=True,
)
register_async_driver(
    "google",
    lambda model=None: AsyncGoogleDriver(api_key=settings.google_api_key, model=model or settings.google_model),
    overwrite=True,
)
register_async_driver(
    "groq",
    lambda model=None: AsyncGroqDriver(api_key=settings.groq_api_key, model=model or settings.groq_model),
    overwrite=True,
)
register_async_driver(
    "openrouter",
    lambda model=None: AsyncOpenRouterDriver(
        api_key=settings.openrouter_api_key, model=model or settings.openrouter_model
    ),
    overwrite=True,
)
register_async_driver(
    "grok",
    lambda model=None: AsyncGrokDriver(api_key=settings.grok_api_key, model=model or settings.grok_model),
    overwrite=True,
)
register_async_driver(
    "moonshot",
    lambda model=None: AsyncMoonshotDriver(
        api_key=settings.moonshot_api_key,
        model=model or settings.moonshot_model,
        endpoint=settings.moonshot_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "modelscope",
    lambda model=None: AsyncModelScopeDriver(
        api_key=settings.modelscope_api_key,
        model=model or settings.modelscope_model,
        endpoint=settings.modelscope_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "zai",
    lambda model=None: AsyncZaiDriver(
        api_key=settings.zhipu_api_key,
        model=model or settings.zhipu_model,
        endpoint=settings.zhipu_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "airllm",
    lambda model=None: AsyncAirLLMDriver(
        model=model or settings.airllm_model,
        compression=settings.airllm_compression,
    ),
    overwrite=True,
)

# Backwards compatibility: expose registry dict
ASYNC_DRIVER_REGISTRY = _get_async_registry()


def get_async_driver(provider_name: str | None = None):
    """Factory to get an async driver instance based on the provider name.

    Uses default model from settings if not overridden.
    """
    provider = (provider_name or settings.ai_provider or "ollama").strip().lower()
    factory = get_async_driver_factory(provider)
    return factory()


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

    factory = get_async_driver_factory(provider)
    return factory(model_id)
