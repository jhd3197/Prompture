"""Vector store registry and factory functions.

Mirrors :mod:`prompture.rag.loader_registry` and
:mod:`prompture.rag.chunkers.chunker_registry` for vector stores.  Each
vector store has a configuration footprint (connection params,
collection names, etc.), so factories accept arbitrary keyword arguments
at instantiation time.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import AsyncVectorStore, VectorStore

#: name → factory (kwargs → VectorStore)
VECTORSTORE_REGISTRY: dict[str, Callable[..., VectorStore]] = {}

#: name → factory (kwargs → AsyncVectorStore)
ASYNC_VECTORSTORE_REGISTRY: dict[str, Callable[..., AsyncVectorStore]] = {}


# ── Registration helpers ─────────────────────────────────────────────────────


def register_vectorstore(
    name: str,
    factory: Callable[..., VectorStore],
    overwrite: bool = False,
) -> None:
    """Register a sync vector store factory under ``name``."""
    key = name.lower()
    if not overwrite and key in VECTORSTORE_REGISTRY:
        raise ValueError(f"Vector store '{name}' already registered. Pass overwrite=True to replace.")
    VECTORSTORE_REGISTRY[key] = factory


def register_async_vectorstore(
    name: str,
    factory: Callable[..., AsyncVectorStore],
    overwrite: bool = False,
) -> None:
    """Register an async vector store factory under ``name``."""
    key = name.lower()
    if not overwrite and key in ASYNC_VECTORSTORE_REGISTRY:
        raise ValueError(f"Async vector store '{name}' already registered. Pass overwrite=True to replace.")
    ASYNC_VECTORSTORE_REGISTRY[key] = factory


# ── Factory functions ────────────────────────────────────────────────────────


def get_vectorstore(name: str, **kwargs: Any) -> VectorStore:
    """Instantiate a sync vector store by registered name with ``**kwargs``."""
    key = name.lower()
    if key not in VECTORSTORE_REGISTRY:
        available = ", ".join(sorted(VECTORSTORE_REGISTRY)) or "<none>"
        raise KeyError(f"No sync vector store registered under '{name}'. Available: {available}")
    return VECTORSTORE_REGISTRY[key](**kwargs)


def get_async_vectorstore(name: str, **kwargs: Any) -> AsyncVectorStore:
    """Instantiate an async vector store by registered name with ``**kwargs``."""
    key = name.lower()
    if key not in ASYNC_VECTORSTORE_REGISTRY:
        available = ", ".join(sorted(ASYNC_VECTORSTORE_REGISTRY)) or "<none>"
        raise KeyError(f"No async vector store registered under '{name}'. Available: {available}")
    return ASYNC_VECTORSTORE_REGISTRY[key](**kwargs)
