from typing import Optional

from ..settings import settings
from .airllm_driver import AirLLMDriver
from .async_airllm_driver import AsyncAirLLMDriver
from .async_azure_driver import AsyncAzureDriver
from .async_claude_driver import AsyncClaudeDriver
from .async_google_driver import AsyncGoogleDriver
from .async_grok_driver import AsyncGrokDriver
from .async_groq_driver import AsyncGroqDriver
from .async_hugging_driver import AsyncHuggingFaceDriver
from .async_lmstudio_driver import AsyncLMStudioDriver
from .async_local_http_driver import AsyncLocalHTTPDriver
from .async_ollama_driver import AsyncOllamaDriver
from .async_openai_driver import AsyncOpenAIDriver
from .async_openrouter_driver import AsyncOpenRouterDriver
from .async_registry import ASYNC_DRIVER_REGISTRY, get_async_driver, get_async_driver_for_model
from .azure_driver import AzureDriver
from .claude_driver import ClaudeDriver
from .google_driver import GoogleDriver
from .grok_driver import GrokDriver
from .groq_driver import GroqDriver
from .lmstudio_driver import LMStudioDriver
from .local_http_driver import LocalHTTPDriver
from .ollama_driver import OllamaDriver
from .openai_driver import OpenAIDriver
from .openrouter_driver import OpenRouterDriver

# Central registry: maps provider → factory function
DRIVER_REGISTRY = {
    "openai": lambda model=None: OpenAIDriver(api_key=settings.openai_api_key, model=model or settings.openai_model),
    "ollama": lambda model=None: OllamaDriver(endpoint=settings.ollama_endpoint, model=model or settings.ollama_model),
    "claude": lambda model=None: ClaudeDriver(api_key=settings.claude_api_key, model=model or settings.claude_model),
    "lmstudio": lambda model=None: LMStudioDriver(
        endpoint=settings.lmstudio_endpoint, model=model or settings.lmstudio_model
    ),
    "azure": lambda model=None: AzureDriver(
        api_key=settings.azure_api_key, endpoint=settings.azure_api_endpoint, deployment_id=settings.azure_deployment_id
    ),
    "local_http": lambda model=None: LocalHTTPDriver(endpoint=settings.local_http_endpoint, model=model),
    "google": lambda model=None: GoogleDriver(api_key=settings.google_api_key, model=model or settings.google_model),
    "groq": lambda model=None: GroqDriver(api_key=settings.groq_api_key, model=model or settings.groq_model),
    "openrouter": lambda model=None: OpenRouterDriver(
        api_key=settings.openrouter_api_key, model=model or settings.openrouter_model
    ),
    "grok": lambda model=None: GrokDriver(api_key=settings.grok_api_key, model=model or settings.grok_model),
    "airllm": lambda model=None: AirLLMDriver(
        model=model or settings.airllm_model,
        compression=settings.airllm_compression,
    ),
}


def get_driver(provider_name: Optional[str] = None):
    """
    Factory to get a driver instance based on the provider name (legacy style).
    Uses default model from settings if not overridden.
    """
    provider = (provider_name or settings.ai_provider or "ollama").strip().lower()
    if provider not in DRIVER_REGISTRY:
        raise ValueError(f"Unknown provider: {provider_name}")
    return DRIVER_REGISTRY[provider]()  # use default model from settings


def get_driver_for_model(model_str: str):
    """
    Factory to get a driver instance based on a full model string.
    Format: provider/model_id
    Example: "openai/gpt-4-turbo-preview"

    Args:
        model_str: Model identifier string. Can be either:
                   - Full format: "provider/model" (e.g. "openai/gpt-4")
                   - Provider only: "provider" (e.g. "openai")

    Returns:
        A configured driver instance for the specified provider/model.

    Raises:
        ValueError: If provider is invalid or format is incorrect.
    """
    if not isinstance(model_str, str):
        raise ValueError("Model string must be a string, got {type(model_str)}")

    if not model_str:
        raise ValueError("Model string cannot be empty")

    # Extract provider and model ID
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None

    # Validate provider
    if provider not in DRIVER_REGISTRY:
        raise ValueError(f"Unsupported provider '{provider}'")

    # Create driver with model ID if provided, otherwise use default
    return DRIVER_REGISTRY[provider](model_id)


__all__ = [
    # Async drivers
    "ASYNC_DRIVER_REGISTRY",
    # Sync drivers
    "AirLLMDriver",
    "AsyncAirLLMDriver",
    "AsyncAzureDriver",
    "AsyncClaudeDriver",
    "AsyncGoogleDriver",
    "AsyncGrokDriver",
    "AsyncGroqDriver",
    "AsyncHuggingFaceDriver",
    "AsyncLMStudioDriver",
    "AsyncLocalHTTPDriver",
    "AsyncOllamaDriver",
    "AsyncOpenAIDriver",
    "AsyncOpenRouterDriver",
    "AzureDriver",
    "ClaudeDriver",
    "GoogleDriver",
    "GrokDriver",
    "GroqDriver",
    "LMStudioDriver",
    "LocalHTTPDriver",
    "OllamaDriver",
    "OpenAIDriver",
    "OpenRouterDriver",
    "get_async_driver",
    "get_async_driver_for_model",
    "get_driver",
    "get_driver_for_model",
]
