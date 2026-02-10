"""Driver registry and factory functions.

This module provides:
- Built-in drivers for popular LLM providers
- A pluggable registry system for custom drivers
- Factory functions to instantiate drivers by provider/model name

Custom Driver Registration:
    from prompture import register_driver

    def my_driver_factory(model=None):
        return MyCustomDriver(model=model)

    register_driver("my_provider", my_driver_factory)

    # Now you can use it
    driver = get_driver_for_model("my_provider/my-model")

Entry Point Discovery:
    Third-party packages can register drivers via entry points.
    Add to your pyproject.toml:

    [project.entry-points."prompture.drivers"]
    my_provider = "my_package.drivers:my_driver_factory"
"""

from typing import Optional

from ..infra.settings import settings
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
from .async_modelscope_driver import AsyncModelScopeDriver
from .async_moonshot_driver import AsyncMoonshotDriver
from .async_ollama_driver import AsyncOllamaDriver
from .async_openai_driver import AsyncOpenAIDriver
from .async_openrouter_driver import AsyncOpenRouterDriver
from .async_registry import ASYNC_DRIVER_REGISTRY, get_async_driver, get_async_driver_for_model
from .async_zai_driver import AsyncZaiDriver
from .azure_config import (
    clear_azure_configs,
    register_azure_config,
    set_azure_config_resolver,
    unregister_azure_config,
)
from .azure_driver import AzureDriver
from .claude_driver import ClaudeDriver
from .google_driver import GoogleDriver
from .grok_driver import GrokDriver
from .groq_driver import GroqDriver
from .lmstudio_driver import LMStudioDriver
from .local_http_driver import LocalHTTPDriver
from .modelscope_driver import ModelScopeDriver
from .moonshot_driver import MoonshotDriver
from .ollama_driver import OllamaDriver
from .openai_driver import OpenAIDriver
from .openrouter_driver import OpenRouterDriver
from .registry import (
    _get_sync_registry,
    get_async_driver_factory,
    get_async_stt_driver_factory,
    get_async_tts_driver_factory,
    get_driver_factory,
    get_stt_driver_factory,
    get_tts_driver_factory,
    is_async_driver_registered,
    is_async_stt_driver_registered,
    is_async_tts_driver_registered,
    is_driver_registered,
    is_stt_driver_registered,
    is_tts_driver_registered,
    list_registered_async_drivers,
    list_registered_async_stt_drivers,
    list_registered_async_tts_drivers,
    list_registered_drivers,
    list_registered_stt_drivers,
    list_registered_tts_drivers,
    load_entry_point_drivers,
    register_async_driver,
    register_async_stt_driver,
    register_async_tts_driver,
    register_driver,
    register_stt_driver,
    register_tts_driver,
    unregister_async_driver,
    unregister_async_stt_driver,
    unregister_async_tts_driver,
    unregister_driver,
    unregister_stt_driver,
    unregister_tts_driver,
)
from .async_base import AsyncDriver
from .async_elevenlabs_stt_driver import AsyncElevenLabsSTTDriver
from .async_elevenlabs_tts_driver import AsyncElevenLabsTTSDriver
from .async_openai_stt_driver import AsyncOpenAISTTDriver
from .async_openai_tts_driver import AsyncOpenAITTSDriver
from .async_stt_base import AsyncSTTDriver
from .async_tts_base import AsyncTTSDriver
from .elevenlabs_stt_driver import ElevenLabsSTTDriver
from .elevenlabs_tts_driver import ElevenLabsTTSDriver
from .openai_stt_driver import OpenAISTTDriver
from .openai_tts_driver import OpenAITTSDriver
from .stt_base import STTDriver
from .tts_base import TTSDriver
from .zai_driver import ZaiDriver

# Register built-in sync drivers
register_driver(
    "openai",
    lambda model=None: OpenAIDriver(api_key=settings.openai_api_key, model=model or settings.openai_model),
    overwrite=True,
)
register_driver(
    "ollama",
    lambda model=None: OllamaDriver(endpoint=settings.ollama_endpoint, model=model or settings.ollama_model),
    overwrite=True,
)
register_driver(
    "claude",
    lambda model=None: ClaudeDriver(api_key=settings.claude_api_key, model=model or settings.claude_model),
    overwrite=True,
)
register_driver(
    "lmstudio",
    lambda model=None: LMStudioDriver(
        endpoint=settings.lmstudio_endpoint,
        model=model or settings.lmstudio_model,
        api_key=settings.lmstudio_api_key,
    ),
    overwrite=True,
)
register_driver(
    "azure",
    lambda model=None: AzureDriver(
        api_key=settings.azure_api_key,
        endpoint=settings.azure_api_endpoint,
        deployment_id=settings.azure_deployment_id,
        model=model or "gpt-4o-mini",
    ),
    overwrite=True,
)
register_driver(
    "local_http",
    lambda model=None: LocalHTTPDriver(endpoint=settings.local_http_endpoint, model=model),
    overwrite=True,
)
register_driver(
    "google",
    lambda model=None: GoogleDriver(api_key=settings.google_api_key, model=model or settings.google_model),
    overwrite=True,
)
register_driver(
    "groq",
    lambda model=None: GroqDriver(api_key=settings.groq_api_key, model=model or settings.groq_model),
    overwrite=True,
)
register_driver(
    "openrouter",
    lambda model=None: OpenRouterDriver(api_key=settings.openrouter_api_key, model=model or settings.openrouter_model),
    overwrite=True,
)
register_driver(
    "grok",
    lambda model=None: GrokDriver(api_key=settings.grok_api_key, model=model or settings.grok_model),
    overwrite=True,
)
register_driver(
    "moonshot",
    lambda model=None: MoonshotDriver(
        api_key=settings.moonshot_api_key,
        model=model or settings.moonshot_model,
        endpoint=settings.moonshot_endpoint,
    ),
    overwrite=True,
)
register_driver(
    "modelscope",
    lambda model=None: ModelScopeDriver(
        api_key=settings.modelscope_api_key,
        model=model or settings.modelscope_model,
        endpoint=settings.modelscope_endpoint,
    ),
    overwrite=True,
)
register_driver(
    "zai",
    lambda model=None: ZaiDriver(
        api_key=settings.zhipu_api_key,
        model=model or settings.zhipu_model,
        endpoint=settings.zhipu_endpoint,
    ),
    overwrite=True,
)
register_driver(
    "airllm",
    lambda model=None: AirLLMDriver(
        model=model or settings.airllm_model,
        compression=settings.airllm_compression,
    ),
    overwrite=True,
)

# Trigger audio driver registration
from .audio_registry import (  # noqa: E402
    get_async_stt_driver_for_model,
    get_async_tts_driver_for_model,
    get_stt_driver_for_model,
    get_tts_driver_for_model,
)

# Backwards compatibility: expose registry dict (read-only view recommended)
DRIVER_REGISTRY = _get_sync_registry()


def get_driver(provider_name: Optional[str] = None):
    """
    Factory to get a driver instance based on the provider name (legacy style).
    Uses default model from settings if not overridden.
    """
    provider = (provider_name or settings.ai_provider or "ollama").strip().lower()
    factory = get_driver_factory(provider)
    return factory()  # use default model from settings


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

    # Get factory (validates provider exists)
    factory = get_driver_factory(provider)

    # Create driver with model ID if provided, otherwise use default
    return factory(model_id)


__all__ = [
    "ASYNC_DRIVER_REGISTRY",
    # Async base classes
    "AsyncDriver",
    "AsyncSTTDriver",
    "AsyncTTSDriver",
    # Legacy registry dicts (for backwards compatibility)
    "DRIVER_REGISTRY",
    # Sync LLM drivers
    "AirLLMDriver",
    # Async LLM drivers
    "AsyncAirLLMDriver",
    "AsyncAzureDriver",
    "AsyncClaudeDriver",
    "AsyncGoogleDriver",
    "AsyncGrokDriver",
    "AsyncGroqDriver",
    "AsyncHuggingFaceDriver",
    "AsyncLMStudioDriver",
    "AsyncLocalHTTPDriver",
    "AsyncModelScopeDriver",
    "AsyncMoonshotDriver",
    "AsyncOllamaDriver",
    "AsyncOpenAIDriver",
    "AsyncOpenRouterDriver",
    "AsyncZaiDriver",
    # Async audio drivers
    "AsyncElevenLabsSTTDriver",
    "AsyncElevenLabsTTSDriver",
    "AsyncOpenAISTTDriver",
    "AsyncOpenAITTSDriver",
    # Sync LLM drivers
    "AzureDriver",
    "ClaudeDriver",
    "GoogleDriver",
    "GrokDriver",
    "GroqDriver",
    "LMStudioDriver",
    "LocalHTTPDriver",
    "ModelScopeDriver",
    "MoonshotDriver",
    "OllamaDriver",
    "OpenAIDriver",
    "OpenRouterDriver",
    "ZaiDriver",
    # Sync audio drivers
    "ElevenLabsSTTDriver",
    "ElevenLabsTTSDriver",
    "OpenAISTTDriver",
    "OpenAITTSDriver",
    # STT/TTS base classes
    "STTDriver",
    "TTSDriver",
    # Azure config API
    "clear_azure_configs",
    "get_async_driver",
    "get_async_driver_for_model",
    # Audio factory functions
    "get_async_stt_driver_for_model",
    "get_async_tts_driver_for_model",
    "get_stt_driver_for_model",
    "get_tts_driver_for_model",
    # LLM factory functions
    "get_driver",
    "get_driver_for_model",
    # Audio registry query functions
    "get_async_stt_driver_factory",
    "get_async_tts_driver_factory",
    "get_stt_driver_factory",
    "get_tts_driver_factory",
    "is_async_driver_registered",
    "is_async_stt_driver_registered",
    "is_async_tts_driver_registered",
    "is_driver_registered",
    "is_stt_driver_registered",
    "is_tts_driver_registered",
    "list_registered_async_drivers",
    "list_registered_async_stt_drivers",
    "list_registered_async_tts_drivers",
    "list_registered_drivers",
    "list_registered_stt_drivers",
    "list_registered_tts_drivers",
    "load_entry_point_drivers",
    "register_async_driver",
    "register_async_stt_driver",
    "register_async_tts_driver",
    "register_azure_config",
    # Registry functions (public API)
    "register_driver",
    "register_stt_driver",
    "register_tts_driver",
    "set_azure_config_resolver",
    "unregister_async_driver",
    "unregister_async_stt_driver",
    "unregister_async_tts_driver",
    "unregister_azure_config",
    "unregister_driver",
    "unregister_stt_driver",
    "unregister_tts_driver",
]
