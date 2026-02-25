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

from ..infra.provider_env import ProviderEnvironment
from ..infra.settings import settings
from .async_airllm_driver import AsyncAirLLMDriver
from .async_azure_driver import AsyncAzureDriver
from .async_cachibot_driver import AsyncCachiBotDriver
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
from .async_zai_driver import AsyncZaiDriver
from .registry import (
    _get_async_registry,
    get_async_driver_factory,
    register_async_driver,
)

# Register built-in async drivers
register_async_driver(
    "openai",
    lambda model=None: AsyncOpenAIDriver(api_key=settings.openai_api_key, model=model or settings.openai_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "ollama",
    lambda model=None: AsyncOllamaDriver(endpoint=settings.ollama_endpoint, model=model or settings.ollama_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "claude",
    lambda model=None: AsyncClaudeDriver(api_key=settings.claude_api_key, model=model or settings.claude_model),  # type: ignore[misc]
    overwrite=True,
)
# Alias: "anthropic" maps to the same Claude driver for compatibility
register_async_driver(
    "anthropic",
    lambda model=None: AsyncClaudeDriver(api_key=settings.claude_api_key, model=model or settings.claude_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "lmstudio",
    lambda model=None: AsyncLMStudioDriver(  # type: ignore[misc]
        endpoint=settings.lmstudio_endpoint,
        model=model or settings.lmstudio_model,
        api_key=settings.lmstudio_api_key,
    ),
    overwrite=True,
)
register_async_driver(
    "azure",
    lambda model=None: AsyncAzureDriver(  # type: ignore[misc]
        api_key=settings.azure_api_key,
        endpoint=settings.azure_api_endpoint,
        deployment_id=settings.azure_deployment_id,
        model=model or "gpt-4o-mini",
        claude_api_key=settings.azure_claude_api_key,
        claude_endpoint=settings.azure_claude_endpoint,
        mistral_api_key=settings.azure_mistral_api_key,
        mistral_endpoint=settings.azure_mistral_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "local_http",
    lambda model=None: AsyncLocalHTTPDriver(endpoint=getattr(settings, "local_http_endpoint", None), model=model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "google",
    lambda model=None: AsyncGoogleDriver(api_key=settings.google_api_key, model=model or settings.google_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "groq",
    lambda model=None: AsyncGroqDriver(api_key=settings.groq_api_key, model=model or settings.groq_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "openrouter",
    lambda model=None: AsyncOpenRouterDriver(  # type: ignore[misc]
        api_key=settings.openrouter_api_key, model=model or settings.openrouter_model
    ),
    overwrite=True,
)
register_async_driver(
    "grok",
    lambda model=None: AsyncGrokDriver(api_key=settings.grok_api_key, model=model or settings.grok_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "moonshot",
    lambda model=None: AsyncMoonshotDriver(  # type: ignore[misc]
        api_key=settings.moonshot_api_key,
        model=model or settings.moonshot_model,
        endpoint=settings.moonshot_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "modelscope",
    lambda model=None: AsyncModelScopeDriver(  # type: ignore[misc]
        api_key=settings.modelscope_api_key,
        model=model or settings.modelscope_model,
        endpoint=settings.modelscope_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "zai",
    lambda model=None: AsyncZaiDriver(  # type: ignore[misc]
        api_key=settings.zhipu_api_key,
        model=model or settings.zhipu_model,
        endpoint=settings.zhipu_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "cachibot",
    lambda model=None: AsyncCachiBotDriver(  # type: ignore[misc]
        api_key=settings.cachibot_api_key,
        model=model or "openai/gpt-4o-mini",
        endpoint=settings.cachibot_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "airllm",
    lambda model=None: AsyncAirLLMDriver(  # type: ignore[misc]
        model=model or settings.airllm_model,
        compression=settings.airllm_compression,
    ),
    overwrite=True,
)
register_async_driver(
    "huggingface",
    lambda model=None: AsyncHuggingFaceDriver(  # type: ignore[misc]
        endpoint=settings.hf_endpoint,
        token=settings.hf_token,
        model=model or "bert-base-uncased",
    ),
    overwrite=True,
)

# ── Aliases ────────────────────────────────────────────────────────────────
# Common alternative names so users can write e.g. "gemini/..." or "chatgpt/..."
register_async_driver(
    "gemini",
    lambda model=None: AsyncGoogleDriver(api_key=settings.google_api_key, model=model or settings.google_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "chatgpt",
    lambda model=None: AsyncOpenAIDriver(api_key=settings.openai_api_key, model=model or settings.openai_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "xai",
    lambda model=None: AsyncGrokDriver(api_key=settings.grok_api_key, model=model or settings.grok_model),  # type: ignore[misc]
    overwrite=True,
)
register_async_driver(
    "lm_studio",
    lambda model=None: AsyncLMStudioDriver(  # type: ignore[misc]
        endpoint=settings.lmstudio_endpoint,
        model=model or settings.lmstudio_model,
        api_key=settings.lmstudio_api_key,
    ),
    overwrite=True,
)
register_async_driver(
    "lm-studio",
    lambda model=None: AsyncLMStudioDriver(  # type: ignore[misc]
        endpoint=settings.lmstudio_endpoint,
        model=model or settings.lmstudio_model,
        api_key=settings.lmstudio_api_key,
    ),
    overwrite=True,
)
register_async_driver(
    "zhipu",
    lambda model=None: AsyncZaiDriver(  # type: ignore[misc]
        api_key=settings.zhipu_api_key,
        model=model or settings.zhipu_model,
        endpoint=settings.zhipu_endpoint,
    ),
    overwrite=True,
)
register_async_driver(
    "hf",
    lambda model=None: AsyncHuggingFaceDriver(  # type: ignore[misc]
        endpoint=settings.hf_endpoint,
        token=settings.hf_token,
        model=model or "bert-base-uncased",
    ),
    overwrite=True,
)

# Backwards compatibility: expose registry dict
ASYNC_DRIVER_REGISTRY = _get_async_registry()

# ── Per-environment async driver construction ──────────────────────────────
# Maps provider name → (AsyncDriverClass, {ctor_kwarg: env/settings_attr}, default_model_attr)

ASYNC_PROVIDER_DRIVER_MAP: dict[str, tuple[type, dict[str, str], str]] = {
    "openai": (AsyncOpenAIDriver, {"api_key": "openai_api_key"}, "openai_model"),
    "chatgpt": (AsyncOpenAIDriver, {"api_key": "openai_api_key"}, "openai_model"),
    "claude": (AsyncClaudeDriver, {"api_key": "claude_api_key"}, "claude_model"),
    "anthropic": (AsyncClaudeDriver, {"api_key": "claude_api_key"}, "claude_model"),
    "google": (AsyncGoogleDriver, {"api_key": "google_api_key"}, "google_model"),
    "gemini": (AsyncGoogleDriver, {"api_key": "google_api_key"}, "google_model"),
    "groq": (AsyncGroqDriver, {"api_key": "groq_api_key"}, "groq_model"),
    "grok": (AsyncGrokDriver, {"api_key": "grok_api_key"}, "grok_model"),
    "xai": (AsyncGrokDriver, {"api_key": "grok_api_key"}, "grok_model"),
    "openrouter": (AsyncOpenRouterDriver, {"api_key": "openrouter_api_key"}, "openrouter_model"),
    "moonshot": (
        AsyncMoonshotDriver,
        {"api_key": "moonshot_api_key", "endpoint": "moonshot_endpoint"},
        "moonshot_model",
    ),
    "modelscope": (
        AsyncModelScopeDriver,
        {"api_key": "modelscope_api_key", "endpoint": "modelscope_endpoint"},
        "modelscope_model",
    ),
    "zai": (AsyncZaiDriver, {"api_key": "zhipu_api_key", "endpoint": "zhipu_endpoint"}, "zhipu_model"),
    "zhipu": (AsyncZaiDriver, {"api_key": "zhipu_api_key", "endpoint": "zhipu_endpoint"}, "zhipu_model"),
    "ollama": (AsyncOllamaDriver, {"endpoint": "ollama_endpoint"}, "ollama_model"),
    "lmstudio": (
        AsyncLMStudioDriver,
        {"endpoint": "lmstudio_endpoint", "api_key": "lmstudio_api_key"},
        "lmstudio_model",
    ),
    "lm_studio": (
        AsyncLMStudioDriver,
        {"endpoint": "lmstudio_endpoint", "api_key": "lmstudio_api_key"},
        "lmstudio_model",
    ),
    "lm-studio": (
        AsyncLMStudioDriver,
        {"endpoint": "lmstudio_endpoint", "api_key": "lmstudio_api_key"},
        "lmstudio_model",
    ),
    "cachibot": (
        AsyncCachiBotDriver,
        {"api_key": "cachibot_api_key", "endpoint": "cachibot_endpoint"},
        "openai/gpt-4o-mini",
    ),
    "azure": (
        AsyncAzureDriver,
        {
            "api_key": "azure_api_key",
            "endpoint": "azure_api_endpoint",
            "deployment_id": "azure_deployment_id",
            "claude_api_key": "azure_claude_api_key",
            "claude_endpoint": "azure_claude_endpoint",
            "mistral_api_key": "azure_mistral_api_key",
            "mistral_endpoint": "azure_mistral_endpoint",
        },
        "gpt-4o-mini",
    ),
    "huggingface": (AsyncHuggingFaceDriver, {"endpoint": "hf_endpoint", "token": "hf_token"}, "bert-base-uncased"),  # nosec B105
    "hf": (AsyncHuggingFaceDriver, {"endpoint": "hf_endpoint", "token": "hf_token"}, "bert-base-uncased"),  # nosec B105
}


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


def _build_async_driver_with_env(
    provider: str,
    model_id: str | None,
    env: ProviderEnvironment,
) -> object:
    """Construct an async driver using *env* for credential resolution."""
    info = ASYNC_PROVIDER_DRIVER_MAP.get(provider)
    if info is None:
        # Provider not in the direct map — fall back to registry (global settings)
        factory = get_async_driver_factory(provider)
        return factory(model_id)

    driver_cls, kwarg_map, default_model = info
    kwargs: dict[str, object] = {}
    for ctor_kwarg, attr_name in kwarg_map.items():
        kwargs[ctor_kwarg] = env.resolve(attr_name)

    if model_id:
        kwargs["model"] = model_id
    elif default_model:
        kwargs["model"] = getattr(settings, default_model, default_model)

    return driver_cls(**kwargs)


def _build_async_driver_with_overrides(
    provider: str,
    model_id: str | None,
    api_key: str | None = None,
    env: ProviderEnvironment | None = None,
    **overrides: object,
) -> object:
    """Construct an async driver with explicit ``api_key`` and/or ``**overrides``.

    Precedence for each kwarg: ``overrides`` > ``api_key`` > ``env`` > global settings.

    Raises:
        ValueError: If *provider* is not in :data:`ASYNC_PROVIDER_DRIVER_MAP`.
    """
    info = ASYNC_PROVIDER_DRIVER_MAP.get(provider)
    if info is None:
        raise ValueError(
            f"Unknown provider '{provider}' for explicit-credential construction. "
            f"Known providers: {', '.join(sorted(ASYNC_PROVIDER_DRIVER_MAP))}"
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


def get_async_driver(
    provider_name: str | None = None,
    *,
    env: ProviderEnvironment | None = None,
    api_key: str | None = None,
    **overrides: object,
) -> object:
    """Factory to get an async driver instance based on the provider name.

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
        return _build_async_driver_with_overrides(provider, None, api_key=api_key, env=env, **overrides)
    if env is not None:
        return _build_async_driver_with_env(provider, None, env)
    factory = get_async_driver_factory(provider)
    return factory(None)


def get_async_driver_for_model(
    model_str: str,
    *,
    env: ProviderEnvironment | None = None,
    api_key: str | None = None,
    **overrides: object,
) -> object:
    """Factory to get an async driver instance based on a full model string.

    Format: ``provider/model_id``
    Example: ``"openai/gpt-4-turbo-preview"``

    Args:
        model_str: Model identifier string.
        env: Optional per-consumer environment for isolated API keys.
            When ``None``, uses the global settings singleton (current behavior).
        api_key: Explicit API key injected into the driver's credential kwarg.
            Takes precedence over *env* and global settings for that kwarg.
        **overrides: Extra kwargs forwarded to the driver constructor
            (e.g. ``endpoint=``, ``deployment_id=``). Take top precedence.
    """
    if not isinstance(model_str, str):
        raise ValueError(f"Model string must be a string, got {type(model_str)}")

    if not model_str:
        raise ValueError("Model string cannot be empty")

    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None

    if api_key is not None or overrides:
        return _build_async_driver_with_overrides(provider, model_id, api_key=api_key, env=env, **overrides)
    if env is not None:
        return _build_async_driver_with_env(provider, model_id, env)

    factory = get_async_driver_factory(provider)
    return factory(model_id)
