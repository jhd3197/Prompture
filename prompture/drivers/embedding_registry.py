"""Embedding driver registration and factory functions.

Registers built-in embedding drivers (OpenAI, Ollama) and provides
high-level factory functions for instantiating embedding drivers by model string.

Usage:
    from prompture.drivers.embedding_registry import get_embedding_driver_for_model

    driver = get_embedding_driver_for_model("openai/text-embedding-3-small")
    result = driver.embed(["hello world"], {})
"""

from typing import cast

from ..infra.settings import settings
from .async_embedding_base import AsyncEmbeddingDriver
from .async_ollama_embedding_driver import AsyncOllamaEmbeddingDriver
from .async_openai_embedding_driver import AsyncOpenAIEmbeddingDriver
from .embedding_base import EmbeddingDriver
from .ollama_embedding_driver import OllamaEmbeddingDriver
from .openai_embedding_driver import OpenAIEmbeddingDriver
from .registry import (
    get_async_embedding_driver_factory,
    get_embedding_driver_factory,
    register_async_embedding_driver,
    register_embedding_driver,
)

# ── Register built-in OpenAI embedding drivers ────────────────────────────

register_embedding_driver(
    "openai",
    lambda model=None: OpenAIEmbeddingDriver(  # type: ignore[misc]
        api_key=settings.openai_api_key,
        model=model or "text-embedding-3-small",
    ),
    overwrite=True,
)

register_async_embedding_driver(
    "openai",
    lambda model=None: AsyncOpenAIEmbeddingDriver(  # type: ignore[misc]
        api_key=settings.openai_api_key,
        model=model or "text-embedding-3-small",
    ),
    overwrite=True,
)

# ── Register built-in Ollama embedding drivers ─────────────────────────────

register_embedding_driver(
    "ollama",
    lambda model=None: OllamaEmbeddingDriver(  # type: ignore[misc]
        endpoint=settings.ollama_endpoint,
        model=model or "nomic-embed-text",
    ),
    overwrite=True,
)

register_async_embedding_driver(
    "ollama",
    lambda model=None: AsyncOllamaEmbeddingDriver(  # type: ignore[misc]
        endpoint=settings.ollama_endpoint,
        model=model or "nomic-embed-text",
    ),
    overwrite=True,
)

# ── Aliases ────────────────────────────────────────────────────────────────
register_embedding_driver(
    "chatgpt",
    lambda model=None: OpenAIEmbeddingDriver(  # type: ignore[misc]
        api_key=settings.openai_api_key,
        model=model or "text-embedding-3-small",
    ),
    overwrite=True,
)
register_async_embedding_driver(
    "chatgpt",
    lambda model=None: AsyncOpenAIEmbeddingDriver(  # type: ignore[misc]
        api_key=settings.openai_api_key,
        model=model or "text-embedding-3-small",
    ),
    overwrite=True,
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
