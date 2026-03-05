"""Embedding driver factory functions.

Provides high-level factory functions for instantiating embedding drivers
by model string.  Built-in driver registration is handled centrally by
``provider_descriptors.register_all_builtin_drivers()``.

Usage:
    from prompture.drivers.embedding_registry import get_embedding_driver_for_model

    driver = get_embedding_driver_for_model("openai/text-embedding-3-small")
    result = driver.embed(["hello world"], {})
"""

from typing import cast

from .async_embedding_base import AsyncEmbeddingDriver
from .embedding_base import EmbeddingDriver
from .registry import (
    get_async_embedding_driver_factory,
    get_embedding_driver_factory,
)

# ── Factory functions ─────────────────────────────────────────────────────


def get_embedding_driver_for_model(model_str: str) -> EmbeddingDriver:
    """Instantiate a sync embedding driver from a ``"provider/model"`` string.

    Args:
        model_str: e.g. ``"openai/text-embedding-3-small"`` or ``"ollama/nomic-embed-text"``.

    Returns:
        A configured embedding driver instance.
    """
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_embedding_driver_factory(provider)
    return cast(EmbeddingDriver, factory(model_id))


def get_async_embedding_driver_for_model(model_str: str) -> AsyncEmbeddingDriver:
    """Instantiate an async embedding driver from a ``"provider/model"`` string."""
    parts = model_str.split("/", 1)
    provider = parts[0].lower()
    model_id = parts[1] if len(parts) > 1 else None
    factory = get_async_embedding_driver_factory(provider)
    return cast(AsyncEmbeddingDriver, factory(model_id))
