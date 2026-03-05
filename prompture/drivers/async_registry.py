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
from .provider_descriptors import build_provider_driver_map
from .registry import (
    _get_async_registry,
    get_async_driver_factory,
)

# Backwards compatibility: expose registry dict
ASYNC_DRIVER_REGISTRY = _get_async_registry()

# ── Per-environment async driver construction ──────────────────────────────
# Derived from ProviderDescriptor — replaces a ~60-line hardcoded dict.
ASYNC_PROVIDER_DRIVER_MAP: dict[str, tuple[type, dict[str, str], str]] = build_provider_driver_map(is_async=True)


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
