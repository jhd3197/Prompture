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

from ..infra.provider_env import ProviderEnvironment
from ..infra.settings import settings
from .airllm_driver import AirLLMDriver
from .async_airllm_driver import AsyncAirLLMDriver
from .async_azure_driver import AsyncAzureDriver
from .async_base import AsyncDriver
from .async_cachibot_driver import AsyncCachiBotDriver
from .async_claude_driver import AsyncClaudeDriver
from .async_elevenlabs_stt_driver import AsyncElevenLabsSTTDriver
from .async_elevenlabs_tts_driver import AsyncElevenLabsTTSDriver
from .async_embedding_base import AsyncEmbeddingDriver
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
from .async_ollama_embedding_driver import AsyncOllamaEmbeddingDriver
from .async_openai_driver import AsyncOpenAIDriver
from .async_openai_embedding_driver import AsyncOpenAIEmbeddingDriver
from .async_openai_img_gen_driver import AsyncOpenAIImageGenDriver
from .async_openai_stt_driver import AsyncOpenAISTTDriver
from .async_openai_tts_driver import AsyncOpenAITTSDriver
from .async_openrouter_driver import AsyncOpenRouterDriver
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
from .cachibot_driver import CachiBotDriver
from .claude_driver import ClaudeDriver
from .elevenlabs_stt_driver import ElevenLabsSTTDriver
from .elevenlabs_tts_driver import ElevenLabsTTSDriver
from .embedding_base import EMBEDDING_MODEL_DIMENSIONS, EmbeddingDriver
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
from .ollama_embedding_driver import OllamaEmbeddingDriver
from .openai_driver import OpenAIDriver
from .openai_embedding_driver import OpenAIEmbeddingDriver
from .openai_img_gen_driver import OpenAIImageGenDriver
from .openai_stt_driver import OpenAISTTDriver
from .openai_tts_driver import OpenAITTSDriver
from .openrouter_driver import OpenRouterDriver
from .provider_descriptors import (
    PROVIDER_DESCRIPTOR_MAP,
    PROVIDER_DESCRIPTORS,
    build_provider_driver_map,
    register_all_builtin_drivers,
)
from .registry import (
    _get_sync_registry,
    get_async_driver_factory,
    get_async_embedding_driver_factory,
    get_async_img_gen_driver_factory,
    get_async_stt_driver_factory,
    get_async_tts_driver_factory,
    get_driver_factory,
    get_embedding_driver_factory,
    get_img_gen_driver_factory,
    get_stt_driver_factory,
    get_tts_driver_factory,
    is_async_driver_registered,
    is_async_embedding_driver_registered,
    is_async_img_gen_driver_registered,
    is_async_stt_driver_registered,
    is_async_tts_driver_registered,
    is_driver_registered,
    is_embedding_driver_registered,
    is_img_gen_driver_registered,
    is_stt_driver_registered,
    is_tts_driver_registered,
    list_registered_async_drivers,
    list_registered_async_embedding_drivers,
    list_registered_async_img_gen_drivers,
    list_registered_async_stt_drivers,
    list_registered_async_tts_drivers,
    list_registered_drivers,
    list_registered_embedding_drivers,
    list_registered_img_gen_drivers,
    list_registered_stt_drivers,
    list_registered_tts_drivers,
    load_entry_point_drivers,
    register_async_driver,
    register_async_embedding_driver,
    register_async_img_gen_driver,
    register_async_stt_driver,
    register_async_tts_driver,
    register_driver,
    register_embedding_driver,
    register_img_gen_driver,
    register_stt_driver,
    register_tts_driver,
    unregister_async_driver,
    unregister_async_embedding_driver,
    unregister_async_img_gen_driver,
    unregister_async_stt_driver,
    unregister_async_tts_driver,
    unregister_driver,
    unregister_embedding_driver,
    unregister_img_gen_driver,
    unregister_stt_driver,
    unregister_tts_driver,
)
from .stability_img_gen_driver import StabilityImageGenDriver
from .stt_base import STTDriver
from .tts_base import TTSDriver
from .zai_driver import ZaiDriver

# Register all built-in drivers (LLM sync/async, STT, TTS, img_gen, embedding)
# from the unified ProviderDescriptor list — replaces ~200 lines of individual
# register_*() calls that were previously scattered across this file,
# async_registry.py, audio_registry.py, embedding_registry.py, and img_gen_registry.py.
register_all_builtin_drivers()

# Import the async factory functions (they no longer register drivers themselves).
from .async_registry import (
    ASYNC_DRIVER_REGISTRY,
    ASYNC_PROVIDER_DRIVER_MAP,
    get_async_driver,
    get_async_driver_for_model,
)

# Import the modality factory functions (they no longer register drivers themselves).
from .audio_registry import (
    get_async_stt_driver_for_model,
    get_async_tts_driver_for_model,
    get_stt_driver_for_model,
    get_tts_driver_for_model,
)
from .embedding_registry import (
    get_async_embedding_driver_for_model,
    get_embedding_driver_for_model,
)
from .img_gen_registry import (
    get_async_img_gen_driver_for_model,
    get_img_gen_driver_for_model,
)

# Backwards compatibility: expose registry dict (read-only view recommended)
DRIVER_REGISTRY = _get_sync_registry()

# ── Per-environment driver construction ────────────────────────────────────
# Derived from ProviderDescriptor — replaces a ~60-line hardcoded dict.
PROVIDER_DRIVER_MAP: dict[str, tuple[type, dict[str, str], str]] = build_provider_driver_map(is_async=False)


def _find_credential_kwarg(kwarg_map: dict[str, str]) -> str | None:
    """Find the constructor kwarg used for authentication credentials.

    Checks for ``"api_key"`` first, then ``"token"`` (e.g. HuggingFace).
    Returns ``None`` for providers with no auth kwarg (e.g. Ollama).
    """
    if "api_key" in kwarg_map:
        return "api_key"
    if "token" in kwarg_map:
        return "token"
    return None


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


def _build_driver_with_overrides(
    provider: str,
    model_id: str | None,
    api_key: str | None = None,
    env: ProviderEnvironment | None = None,
    **overrides: object,
) -> object:
    """Construct a sync driver with explicit ``api_key`` and/or ``**overrides``.

    Precedence for each kwarg: ``overrides`` > ``api_key`` > ``env`` > global settings.

    Raises:
        ValueError: If *provider* is not in :data:`PROVIDER_DRIVER_MAP`.
    """
    info = PROVIDER_DRIVER_MAP.get(provider)
    if info is None:
        raise ValueError(
            f"Unknown provider '{provider}' for explicit-credential construction. "
            f"Known providers: {', '.join(sorted(PROVIDER_DRIVER_MAP))}"
        )

    driver_cls, kwarg_map, default_model = info
    kwargs: dict[str, object] = {}

    # Base layer: resolve each kwarg from env (with settings fallback) or settings directly
    for ctor_kwarg, attr_name in kwarg_map.items():
        if env is not None:
            kwargs[ctor_kwarg] = env.resolve(attr_name)
        else:
            kwargs[ctor_kwarg] = getattr(settings, attr_name, None)

    # Inject api_key into the credential kwarg (overrides env/settings for that kwarg)
    if api_key is not None:
        cred_kwarg = _find_credential_kwarg(kwarg_map)
        if cred_kwarg:
            kwargs[cred_kwarg] = api_key

    # Set model
    if model_id:
        kwargs["model"] = model_id
    elif default_model:
        kwargs["model"] = getattr(settings, default_model, default_model)

    # Explicit overrides take top precedence
    kwargs.update(overrides)

    return driver_cls(**kwargs)


def get_driver(
    provider_name: str | None = None,
    *,
    env: ProviderEnvironment | None = None,
    api_key: str | None = None,
    **overrides: object,
) -> object:
    """Factory to get a driver instance based on the provider name (legacy style).

    Uses default model from settings if not overridden.

    Args:
        provider_name: Provider name (e.g. "openai"). Defaults to ``settings.ai_provider``.
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).
        api_key: Explicit API key injected into the driver's credential kwarg.
            Takes precedence over *env* and global settings for that kwarg.
        **overrides: Extra kwargs forwarded to the driver constructor
            (e.g. ``endpoint=``, ``deployment_id=``). Take top precedence.
    """
    provider = (provider_name or settings.ai_provider or "ollama").strip().lower()
    if api_key is not None or overrides:
        return _build_driver_with_overrides(provider, None, api_key=api_key, env=env, **overrides)
    if env is not None:
        return _build_driver_with_env(provider, None, env)
    factory = get_driver_factory(provider)
    return factory(None)  # use default model from settings


def get_driver_for_model(
    model_str: str,
    *,
    env: ProviderEnvironment | None = None,
    api_key: str | None = None,
    **overrides: object,
) -> object:
    """Factory to get a driver instance based on a full model string.

    Format: ``provider/model_id``
    Example: ``"openai/gpt-4-turbo-preview"``

    Args:
        model_str: Model identifier string. Can be either:
                   - Full format: ``"provider/model"`` (e.g. ``"openai/gpt-4"``)
                   - Provider only: ``"provider"`` (e.g. ``"openai"``)
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).
        api_key: Explicit API key injected into the driver's credential kwarg.
            Takes precedence over *env* and global settings for that kwarg.
        **overrides: Extra kwargs forwarded to the driver constructor
            (e.g. ``endpoint=``, ``deployment_id=``). Take top precedence.

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

    if api_key is not None or overrides:
        return _build_driver_with_overrides(provider, model_id, api_key=api_key, env=env, **overrides)
    if env is not None:
        return _build_driver_with_env(provider, model_id, env)

    # Get factory (validates provider exists)
    factory = get_driver_factory(provider)

    # Create driver with model ID if provided, otherwise use default
    return factory(model_id)


__all__ = [
    "ASYNC_DRIVER_REGISTRY",
    # Provider driver maps (for explicit-credential construction)
    "ASYNC_PROVIDER_DRIVER_MAP",
    # Legacy registry dicts (for backwards compatibility)
    "DRIVER_REGISTRY",
    # Embedding model dimension metadata
    "EMBEDDING_MODEL_DIMENSIONS",
    "PROVIDER_DESCRIPTORS",
    # Provider descriptors
    "PROVIDER_DESCRIPTOR_MAP",
    # Provider driver maps (for explicit-credential construction)
    "PROVIDER_DRIVER_MAP",
    # Sync LLM drivers
    "AirLLMDriver",
    # Async LLM drivers
    "AsyncAirLLMDriver",
    "AsyncAzureDriver",
    "AsyncCachiBotDriver",
    "AsyncClaudeDriver",
    # Async base classes
    "AsyncDriver",
    # Async audio drivers
    "AsyncElevenLabsSTTDriver",
    "AsyncElevenLabsTTSDriver",
    # Async embedding drivers
    "AsyncEmbeddingDriver",
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
    "AsyncOllamaEmbeddingDriver",
    "AsyncOpenAIDriver",
    "AsyncOpenAIEmbeddingDriver",
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
    "CachiBotDriver",
    "ClaudeDriver",
    # Sync audio drivers
    "ElevenLabsSTTDriver",
    "ElevenLabsTTSDriver",
    # Embedding base class
    "EmbeddingDriver",
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
    "OllamaEmbeddingDriver",
    "OpenAIDriver",
    "OpenAIEmbeddingDriver",
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
    # Embedding registry query functions
    "get_async_embedding_driver_factory",
    # Embedding factory functions
    "get_async_embedding_driver_for_model",
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
    "get_embedding_driver_factory",
    "get_embedding_driver_for_model",
    "get_img_gen_driver_factory",
    "get_img_gen_driver_for_model",
    "get_stt_driver_factory",
    "get_stt_driver_for_model",
    "get_tts_driver_factory",
    "get_tts_driver_for_model",
    # Other registry query functions
    "is_async_driver_registered",
    "is_async_embedding_driver_registered",
    "is_async_img_gen_driver_registered",
    "is_async_stt_driver_registered",
    "is_async_tts_driver_registered",
    "is_driver_registered",
    "is_embedding_driver_registered",
    "is_img_gen_driver_registered",
    "is_stt_driver_registered",
    "is_tts_driver_registered",
    "list_registered_async_drivers",
    "list_registered_async_embedding_drivers",
    "list_registered_async_img_gen_drivers",
    "list_registered_async_stt_drivers",
    "list_registered_async_tts_drivers",
    "list_registered_drivers",
    "list_registered_embedding_drivers",
    "list_registered_img_gen_drivers",
    "list_registered_stt_drivers",
    "list_registered_tts_drivers",
    "load_entry_point_drivers",
    "register_all_builtin_drivers",
    "register_async_driver",
    "register_async_embedding_driver",
    "register_async_img_gen_driver",
    "register_async_stt_driver",
    "register_async_tts_driver",
    "register_azure_config",
    # Registry functions (public API)
    "register_driver",
    "register_embedding_driver",
    "register_img_gen_driver",
    "register_stt_driver",
    "register_tts_driver",
    "set_azure_config_resolver",
    "unregister_async_driver",
    "unregister_async_embedding_driver",
    "unregister_async_img_gen_driver",
    "unregister_async_stt_driver",
    "unregister_async_tts_driver",
    "unregister_azure_config",
    "unregister_driver",
    "unregister_embedding_driver",
    "unregister_img_gen_driver",
    "unregister_stt_driver",
    "unregister_tts_driver",
]
