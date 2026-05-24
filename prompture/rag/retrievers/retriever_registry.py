"""Retriever registry and factory functions.

Mirrors :mod:`prompture.rag.loader_registry`,
:mod:`prompture.rag.chunkers.chunker_registry`, and
:mod:`prompture.rag.vectorstores.vectorstore_registry`.  Retriever
factories accept arbitrary keyword arguments at instantiation time
because each retriever exposes a different set of configuration options
(``vector_store``, ``k``, ``fetch_k``, ``lambda_mult``, ``corpus`` ...).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import AsyncRetriever, Retriever

#: name → factory (kwargs → Retriever)
RETRIEVER_REGISTRY: dict[str, Callable[..., Retriever]] = {}

#: name → factory (kwargs → AsyncRetriever)
ASYNC_RETRIEVER_REGISTRY: dict[str, Callable[..., AsyncRetriever]] = {}


# ── Registration helpers ─────────────────────────────────────────────────────


def register_retriever(
    name: str,
    factory: Callable[..., Retriever],
    overwrite: bool = False,
) -> None:
    """Register a sync retriever factory under ``name``."""
    key = name.lower()
    if not overwrite and key in RETRIEVER_REGISTRY:
        raise ValueError(f"Retriever '{name}' already registered. Pass overwrite=True to replace.")
    RETRIEVER_REGISTRY[key] = factory


def register_async_retriever(
    name: str,
    factory: Callable[..., AsyncRetriever],
    overwrite: bool = False,
) -> None:
    """Register an async retriever factory under ``name``."""
    key = name.lower()
    if not overwrite and key in ASYNC_RETRIEVER_REGISTRY:
        raise ValueError(f"Async retriever '{name}' already registered. Pass overwrite=True to replace.")
    ASYNC_RETRIEVER_REGISTRY[key] = factory


# ── Factory functions ────────────────────────────────────────────────────────


def get_retriever(name: str, **kwargs: Any) -> Retriever:
    """Instantiate a sync retriever by registered name with ``**kwargs``."""
    key = name.lower()
    if key not in RETRIEVER_REGISTRY:
        available = ", ".join(sorted(RETRIEVER_REGISTRY)) or "<none>"
        raise KeyError(f"No sync retriever registered under '{name}'. Available: {available}")
    return RETRIEVER_REGISTRY[key](**kwargs)


def get_async_retriever(name: str, **kwargs: Any) -> AsyncRetriever:
    """Instantiate an async retriever by registered name with ``**kwargs``."""
    key = name.lower()
    if key not in ASYNC_RETRIEVER_REGISTRY:
        available = ", ".join(sorted(ASYNC_RETRIEVER_REGISTRY)) or "<none>"
        raise KeyError(f"No async retriever registered under '{name}'. Available: {available}")
    return ASYNC_RETRIEVER_REGISTRY[key](**kwargs)
