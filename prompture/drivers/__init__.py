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

from __future__ import annotations

from typing import Optional

from ..infra.provider_env import ProviderEnvironment
from ..infra.settings import settings
from .airllm_driver import AirLLMDriver
from .async_airllm_driver import AsyncAirLLMDriver
from .async_azure_driver import AsyncAzureDriver
from .async_base import AsyncDriver
from .async_claude_driver import AsyncClaudeDriver
from .async_elevenlabs_stt_driver import AsyncElevenLabsSTTDriver
from .async_elevenlabs_tts_driver import AsyncElevenLabsTTSDriver
from .async_google_driver import AsyncGoogleDriver
from .async_google_img_gen_driver import AsyncGoogleImageGenDriver
from .async_grok_driver import AsyncGrokDriver
from .async_grok_img_gen_driver import AsyncGrokImageGenDriver
from .async_groq_driver import AsyncGroqDriver
from .async_hugging_driver import AsyncHuggingFaceDriver
from .async_img_gen_base import AsyncImageGenDriver
from .async_lmstudio_driver import AsyncLMStudioDriver
from .async_local_http_driver import AsyncLocalHTTPDriver
from .async_modelscope_driver import AsyncModelScopeDriver
from .async_moonshot_driver import AsyncMoonshotDriver
from .async_ollama_driver import AsyncOllamaDriver
from .async_openai_driver import AsyncOpenAIDriver
from .async_openai_img_gen_driver import AsyncOpenAIImageGenDriver
from .async_openai_stt_driver import AsyncOpenAISTTDriver
from .async_openai_tts_driver import AsyncOpenAITTSDriver
from .async_openrouter_driver import AsyncOpenRouterDriver
from .async_registry import ASYNC_DRIVER_REGISTRY, get_async_driver, get_async_driver_for_model
from .async_stability_img_gen_driver import AsyncStabilityImageGenDriver
from .async_stt_base import AsyncSTTDriver
from .async_tts_base import AsyncTTSDriver
from .async_zai_driver import AsyncZaiDriver
from .azure_config import (
    clear_azure_configs,
    register_azure_config,
    set_azure_config_resolver,
    unregister_azure_config,
)
from .azure_driver import AzureDriver
from .claude_driver import ClaudeDriver
from .elevenlabs_stt_driver import ElevenLabsSTTDriver
from .elevenlabs_tts_driver import ElevenLabsTTSDriver
from .google_driver import GoogleDriver
from .google_img_gen_driver import GoogleImageGenDriver
from .grok_driver import GrokDriver
from .grok_img_gen_driver import GrokImageGenDriver
from .groq_driver import GroqDriver
from .hugging_driver import HuggingFaceDriver
from .img_gen_base import ImageGenDriver
from .lmstudio_driver import LMStudioDriver
from .local_http_driver import LocalHTTPDriver
from .modelscope_driver import ModelScopeDriver
from .moonshot_driver import MoonshotDriver
from .ollama_driver import OllamaDriver
from .openai_driver import OpenAIDriver
from .openai_img_gen_driver import OpenAIImageGenDriver
from .openai_stt_driver import OpenAISTTDriver
from .openai_tts_driver import OpenAITTSDriver
from .openrouter_driver import OpenRouterDriver
from .registry import (
    _get_sync_registry,
    get_async_driver_factory,
    get_async_img_gen_driver_factory,
    get_async_stt_driver_factory,
    get_async_tts_driver_factory,
    get_driver_factory,
    get_img_gen_driver_factory,
    get_stt_driver_factory,
    get_tts_driver_factory,
    is_async_driver_registered,
    is_async_img_gen_driver_registered,
    is_async_stt_driver_registered,
    is_async_tts_driver_registered,
    is_driver_registered,
    is_img_gen_driver_registered,
    is_stt_driver_registered,
    is_tts_driver_registered,
    list_registered_async_drivers,
    list_registered_async_img_gen_drivers,
    list_registered_async_stt_drivers,
    list_registered_async_tts_drivers,
    list_registered_drivers,
    list_registered_img_gen_drivers,
    list_registered_stt_drivers,
    list_registered_tts_drivers,
    load_entry_point_drivers,
    register_async_driver,
    register_async_img_gen_driver,
    register_async_stt_driver,
    register_async_tts_driver,
    register_driver,
    register_img_gen_driver,
    register_stt_driver,
    register_tts_driver,
    unregister_async_driver,
    unregister_async_img_gen_driver,
    unregister_async_stt_driver,
    unregister_async_tts_driver,
    unregister_driver,
    unregister_img_gen_driver,
    unregister_stt_driver,
    unregister_tts_driver,
)
from .stability_img_gen_driver import StabilityImageGenDriver
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
# Alias: "anthropic" maps to the same Claude driver for compatibility
register_driver(
    "anthropic",
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
register_driver(
    "huggingface",
    lambda model=None: HuggingFaceDriver(
        endpoint=settings.hf_endpoint,
        token=settings.hf_token,
        model=model or "bert-base-uncased",
    ),
    overwrite=True,
)

# ── Aliases ────────────────────────────────────────────────────────────────
# Common alternative names so users can write e.g. "gemini/..." or "chatgpt/..."
register_driver(
    "gemini",
    lambda model=None: GoogleDriver(api_key=settings.google_api_key, model=model or settings.google_model),
    overwrite=True,
)
register_driver(
    "chatgpt",
    lambda model=None: OpenAIDriver(api_key=settings.openai_api_key, model=model or settings.openai_model),
    overwrite=True,
)
register_driver(
    "xai",
    lambda model=None: GrokDriver(api_key=settings.grok_api_key, model=model or settings.grok_model),
    overwrite=True,
)
register_driver(
    "lm_studio",
    lambda model=None: LMStudioDriver(
        endpoint=settings.lmstudio_endpoint,
        model=model or settings.lmstudio_model,
        api_key=settings.lmstudio_api_key,
    ),
    overwrite=True,
)
register_driver(
    "lm-studio",
    lambda model=None: LMStudioDriver(
        endpoint=settings.lmstudio_endpoint,
        model=model or settings.lmstudio_model,
        api_key=settings.lmstudio_api_key,
    ),
    overwrite=True,
)
register_driver(
    "zhipu",
    lambda model=None: ZaiDriver(
        api_key=settings.zhipu_api_key,
        model=model or settings.zhipu_model,
        endpoint=settings.zhipu_endpoint,
    ),
    overwrite=True,
)
register_driver(
    "hf",
    lambda model=None: HuggingFaceDriver(
        endpoint=settings.hf_endpoint,
        token=settings.hf_token,
        model=model or "bert-base-uncased",
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

# Trigger image gen driver registration
from .img_gen_registry import (  # noqa: E402
    get_async_img_gen_driver_for_model,
    get_img_gen_driver_for_model,
)

# Backwards compatibility: expose registry dict (read-only view recommended)
DRIVER_REGISTRY = _get_sync_registry()

# ── Per-environment driver construction ────────────────────────────────────
# Maps provider name → (DriverClass, {ctor_kwarg: env/settings_attr}, default_model_attr)
# default_model_attr is either a settings attribute name (e.g. "openai_model") or a
# literal string (e.g. "gpt-4o-mini").  getattr(settings, x, x) resolves both.

PROVIDER_DRIVER_MAP: dict[str, tuple] = {
    "openai": (OpenAIDriver, {"api_key": "openai_api_key"}, "openai_model"),
    "chatgpt": (OpenAIDriver, {"api_key": "openai_api_key"}, "openai_model"),
    "claude": (ClaudeDriver, {"api_key": "claude_api_key"}, "claude_model"),
    "anthropic": (ClaudeDriver, {"api_key": "claude_api_key"}, "claude_model"),
    "google": (GoogleDriver, {"api_key": "google_api_key"}, "google_model"),
    "gemini": (GoogleDriver, {"api_key": "google_api_key"}, "google_model"),
    "groq": (GroqDriver, {"api_key": "groq_api_key"}, "groq_model"),
    "grok": (GrokDriver, {"api_key": "grok_api_key"}, "grok_model"),
    "xai": (GrokDriver, {"api_key": "grok_api_key"}, "grok_model"),
    "openrouter": (OpenRouterDriver, {"api_key": "openrouter_api_key"}, "openrouter_model"),
    "moonshot": (
        MoonshotDriver,
        {"api_key": "moonshot_api_key", "endpoint": "moonshot_endpoint"},
        "moonshot_model",
    ),
    "modelscope": (
        ModelScopeDriver,
        {"api_key": "modelscope_api_key", "endpoint": "modelscope_endpoint"},
        "modelscope_model",
    ),
    "zai": (ZaiDriver, {"api_key": "zhipu_api_key", "endpoint": "zhipu_endpoint"}, "zhipu_model"),
    "zhipu": (ZaiDriver, {"api_key": "zhipu_api_key", "endpoint": "zhipu_endpoint"}, "zhipu_model"),
    "ollama": (OllamaDriver, {"endpoint": "ollama_endpoint"}, "ollama_model"),
    "lmstudio": (
        LMStudioDriver,
        {"endpoint": "lmstudio_endpoint", "api_key": "lmstudio_api_key"},
        "lmstudio_model",
    ),
    "lm_studio": (
        LMStudioDriver,
        {"endpoint": "lmstudio_endpoint", "api_key": "lmstudio_api_key"},
        "lmstudio_model",
    ),
    "lm-studio": (
        LMStudioDriver,
        {"endpoint": "lmstudio_endpoint", "api_key": "lmstudio_api_key"},
        "lmstudio_model",
    ),
    "azure": (
        AzureDriver,
        {"api_key": "azure_api_key", "endpoint": "azure_api_endpoint", "deployment_id": "azure_deployment_id"},
        "gpt-4o-mini",
    ),
    "huggingface": (HuggingFaceDriver, {"endpoint": "hf_endpoint", "token": "hf_token"}, "bert-base-uncased"),
    "hf": (HuggingFaceDriver, {"endpoint": "hf_endpoint", "token": "hf_token"}, "bert-base-uncased"),
}


def _build_driver_with_env(
    provider: str,
    model_id: str | None,
    env: ProviderEnvironment,
) -> object:
    """Construct a sync driver using *env* for credential resolution."""
    info = PROVIDER_DRIVER_MAP.get(provider)
    if info is None:
        # Provider not in the direct map — fall back to registry (global settings)
        factory = get_driver_factory(provider)
        return factory(model_id)

    driver_cls, kwarg_map, default_model = info
    kwargs: dict[str, object] = {}
    for ctor_kwarg, attr_name in kwarg_map.items():
        kwargs[ctor_kwarg] = env.resolve(attr_name)

    if model_id:
        kwargs["model"] = model_id
    elif default_model:
        # If default_model is a settings attr name, resolve it; otherwise use as literal
        kwargs["model"] = getattr(settings, default_model, default_model)

    return driver_cls(**kwargs)


def get_driver(provider_name: Optional[str] = None, *, env: ProviderEnvironment | None = None):
    """
    Factory to get a driver instance based on the provider name (legacy style).
    Uses default model from settings if not overridden.

    Args:
        provider_name: Provider name (e.g. "openai"). Defaults to ``settings.ai_provider``.
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).
    """
    provider = (provider_name or settings.ai_provider or "ollama").strip().lower()
    if env is not None:
        return _build_driver_with_env(provider, None, env)
    factory = get_driver_factory(provider)
    return factory()  # use default model from settings


def get_driver_for_model(model_str: str, *, env: ProviderEnvironment | None = None):
    """
    Factory to get a driver instance based on a full model string.
    Format: provider/model_id
    Example: "openai/gpt-4-turbo-preview"

    Args:
        model_str: Model identifier string. Can be either:
                   - Full format: "provider/model" (e.g. "openai/gpt-4")
                   - Provider only: "provider" (e.g. "openai")
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).

    Returns:
        A configured driver instance for the specified provider/model.

    Raises:
        ValueError: If provider is invalid or format is incorrect.
    """
    if not isinstance(model_str, str):
        raise ValueError(f"Model string must be a string, got {type(model_str)}")

    if not model_str:
        raise ValueError("Model string cannot be empty")

    # Extract provider and model ID
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None

    if env is not None:
        return _build_driver_with_env(provider, model_id, env)

    # Get factory (validates provider exists)
    factory = get_driver_factory(provider)

    # Create driver with model ID if provided, otherwise use default
    return factory(model_id)


__all__ = [
    "ASYNC_DRIVER_REGISTRY",
    # Legacy registry dicts (for backwards compatibility)
    "DRIVER_REGISTRY",
    # Sync LLM drivers
    "AirLLMDriver",
    # Async LLM drivers
    "AsyncAirLLMDriver",
    "AsyncAzureDriver",
    "AsyncClaudeDriver",
    # Async base classes
    "AsyncDriver",
    # Async audio drivers
    "AsyncElevenLabsSTTDriver",
    "AsyncElevenLabsTTSDriver",
    "AsyncGoogleDriver",
    # Async image gen drivers
    "AsyncGoogleImageGenDriver",
    "AsyncGrokDriver",
    "AsyncGrokImageGenDriver",
    "AsyncGroqDriver",
    "AsyncHuggingFaceDriver",
    "AsyncImageGenDriver",
    "AsyncLMStudioDriver",
    "AsyncLocalHTTPDriver",
    "AsyncModelScopeDriver",
    "AsyncMoonshotDriver",
    "AsyncOllamaDriver",
    "AsyncOpenAIDriver",
    "AsyncOpenAIImageGenDriver",
    "AsyncOpenAISTTDriver",
    "AsyncOpenAITTSDriver",
    "AsyncOpenRouterDriver",
    "AsyncSTTDriver",
    "AsyncStabilityImageGenDriver",
    "AsyncTTSDriver",
    "AsyncZaiDriver",
    # Sync LLM drivers
    "AzureDriver",
    "ClaudeDriver",
    # Sync audio drivers
    "ElevenLabsSTTDriver",
    "ElevenLabsTTSDriver",
    "GoogleDriver",
    # Sync image gen drivers
    "GoogleImageGenDriver",
    "GrokDriver",
    "GrokImageGenDriver",
    "GroqDriver",
    "HuggingFaceDriver",
    # Image gen base class
    "ImageGenDriver",
    "LMStudioDriver",
    "LocalHTTPDriver",
    "ModelScopeDriver",
    "MoonshotDriver",
    "OllamaDriver",
    "OpenAIDriver",
    "OpenAIImageGenDriver",
    "OpenAISTTDriver",
    "OpenAITTSDriver",
    "OpenRouterDriver",
    # STT/TTS base classes
    "STTDriver",
    "StabilityImageGenDriver",
    "TTSDriver",
    "ZaiDriver",
    # Azure config API
    "clear_azure_configs",
    "get_async_driver",
    "get_async_driver_for_model",
    # Image gen registry query functions
    "get_async_img_gen_driver_factory",
    # Image gen factory functions
    "get_async_img_gen_driver_for_model",
    # Audio registry query functions
    "get_async_stt_driver_factory",
    # Audio factory functions
    "get_async_stt_driver_for_model",
    "get_async_tts_driver_factory",
    "get_async_tts_driver_for_model",
    # LLM factory functions
    "get_driver",
    "get_driver_for_model",
    "get_img_gen_driver_factory",
    "get_img_gen_driver_for_model",
    "get_stt_driver_factory",
    "get_stt_driver_for_model",
    "get_tts_driver_factory",
    "get_tts_driver_for_model",
    # Other registry query functions
    "is_async_driver_registered",
    "is_async_img_gen_driver_registered",
    "is_async_stt_driver_registered",
    "is_async_tts_driver_registered",
    "is_driver_registered",
    "is_img_gen_driver_registered",
    "is_stt_driver_registered",
    "is_tts_driver_registered",
    "list_registered_async_drivers",
    "list_registered_async_img_gen_drivers",
    "list_registered_async_stt_drivers",
    "list_registered_async_tts_drivers",
    "list_registered_drivers",
    "list_registered_img_gen_drivers",
    "list_registered_stt_drivers",
    "list_registered_tts_drivers",
    "load_entry_point_drivers",
    "register_async_driver",
    "register_async_img_gen_driver",
    "register_async_stt_driver",
    "register_async_tts_driver",
    "register_azure_config",
    # Registry functions (public API)
    "register_driver",
    "register_img_gen_driver",
    "register_stt_driver",
    "register_tts_driver",
    "set_azure_config_resolver",
    "unregister_async_driver",
    "unregister_async_img_gen_driver",
    "unregister_async_stt_driver",
    "unregister_async_tts_driver",
    "unregister_azure_config",
    "unregister_driver",
    "unregister_img_gen_driver",
    "unregister_stt_driver",
    "unregister_tts_driver",
]
