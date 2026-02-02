"""Azure per-model configuration resolution.

Supports multiple Azure endpoints, API keys, and deployment names for
different models, as well as routing to different API backends (OpenAI,
Claude, Mistral) based on the model prefix.

Usage::

    from prompture.drivers.azure_config import (
        register_azure_config,
        set_azure_config_resolver,
        resolve_config,
        classify_backend,
    )

    # Register per-model configs
    register_azure_config("gpt-4o", {
        "endpoint": "https://my-eastus.openai.azure.com/",
        "api_key": "key-eastus",
        "deployment_id": "gpt-4o",
    })

    # Or use a resolver callback
    set_azure_config_resolver(lambda model: my_db.get_config(model))
"""

from __future__ import annotations

import threading
from typing import Any, Callable

# Model prefix → backend type
AZURE_BACKEND_MAP: dict[str, str] = {
    "gpt-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "o4-": "openai",
    "claude-": "claude",
    "mistral-": "mistral",
    "mixtral-": "mistral",
}

_lock = threading.Lock()
_config_registry: dict[str, dict[str, Any]] = {}
_config_resolver: Callable[[str], dict[str, Any]] | None = None


def classify_backend(model: str) -> str:
    """Determine API backend for a model. Default: ``'openai'``."""
    model_lower = model.lower()
    for prefix, backend in AZURE_BACKEND_MAP.items():
        if model_lower.startswith(prefix):
            return backend
    return "openai"


def register_azure_config(name: str, config: dict[str, Any]) -> None:
    """Register a named Azure config (deployment name, region, etc.).

    Args:
        name: Model name key (e.g. ``"gpt-4o"``).
        config: Dict with ``endpoint``, ``api_key``, and optionally
            ``deployment_id``, ``api_version``.
    """
    with _lock:
        _config_registry[name] = config


def unregister_azure_config(name: str) -> None:
    """Remove a previously registered Azure config."""
    with _lock:
        _config_registry.pop(name, None)


def clear_azure_configs() -> None:
    """Remove all registered Azure configs."""
    with _lock:
        _config_registry.clear()


def set_azure_config_resolver(
    resolver: Callable[[str], dict[str, Any]] | None,
) -> None:
    """Set a callback that resolves config per deployment/model name.

    Pass ``None`` to clear the resolver.
    """
    global _config_resolver
    with _lock:
        _config_resolver = resolver


def has_azure_config_resolver() -> bool:
    """Return ``True`` if a config resolver callback is registered."""
    with _lock:
        return _config_resolver is not None


def has_registered_configs() -> bool:
    """Return ``True`` if any named configs are registered."""
    with _lock:
        return len(_config_registry) > 0


def resolve_config(
    model: str,
    override: dict[str, Any] | None = None,
    default_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve Azure config for a model using priority chain.

    Priority:
    1. Per-call ``override`` (highest)
    2. Resolver callback (if registered)
    3. Registry lookup (by model name)
    4. ``default_config`` (env vars fallback)

    Raises:
        ValueError: If no config could be resolved.
    """
    # 1. Per-call override
    if override:
        return override

    # 2. Resolver callback
    with _lock:
        resolver = _config_resolver
    if resolver:
        resolved = resolver(model)
        if resolved:
            return resolved

    # 3. Registry lookup (by model name)
    with _lock:
        if model in _config_registry:
            return _config_registry[model]

    # 4. Default (env vars) — only use if it has at least an endpoint or api_key
    if default_config and (default_config.get("endpoint") or default_config.get("api_key")):
        return default_config

    raise ValueError(
        f"No Azure config found for '{model}'. "
        "Set env vars, register a config with register_azure_config(), "
        "or provide azure_config in options."
    )
