"""Rerank driver factory functions.

Provides high-level factory functions for instantiating rerank drivers
by model string.  Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.

Usage:
    from prompture.drivers.rerank_registry import get_rerank_driver_for_model

    driver = get_rerank_driver_for_model("cohere/rerank-v3.5")
    results = driver.rerank("query", ["doc1", "doc2"], top_n=2)
"""

from typing import cast

# ── Public registry dicts (live views) ────────────────────────────────────
from .registry import (
    _ASYNC_RERANK_REGISTRY,
    _RERANK_REGISTRY,
    get_async_rerank_driver_factory,
    get_rerank_driver_factory,
)
from .rerank_base import AsyncRerankDriver, RerankDriver

RERANK_DRIVER_REGISTRY = _RERANK_REGISTRY
ASYNC_RERANK_DRIVER_REGISTRY = _ASYNC_RERANK_REGISTRY


# ── Factory functions ─────────────────────────────────────────────────────


def get_rerank_driver_for_model(model_str: str) -> RerankDriver:
    """Instantiate a sync rerank driver from a ``"provider/model"`` string.

    Args:
        model_str: e.g. ``"cohere/rerank-v3.5"`` or ``"voyage/rerank-2.5"``.

    Returns:
        A configured rerank driver instance.
    """
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_rerank_driver_factory(provider)
    return cast(RerankDriver, factory(model_id))


def get_async_rerank_driver_for_model(model_str: str) -> AsyncRerankDriver:
    """Instantiate an async rerank driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_rerank_driver_factory(provider)
    return cast(AsyncRerankDriver, factory(model_id))
