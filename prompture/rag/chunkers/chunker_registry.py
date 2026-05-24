"""Chunker registry and factory functions.

Mirrors :mod:`prompture.rag.loader_registry` but for chunkers.  Chunker
factories accept arbitrary keyword arguments at instantiation time
because each chunker exposes a different set of configuration options
(``chunk_size``, ``separators``, ``encoding_name``, ``embedding_driver``,
``headers_to_split_on`` ...).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import AsyncTextChunker, TextChunker

#: name → factory (kwargs → TextChunker)
CHUNKER_REGISTRY: dict[str, Callable[..., TextChunker]] = {}

#: name → factory (kwargs → AsyncTextChunker)
ASYNC_CHUNKER_REGISTRY: dict[str, Callable[..., AsyncTextChunker]] = {}


# ── Registration helpers ─────────────────────────────────────────────────────


def register_chunker(
    name: str,
    factory: Callable[..., TextChunker],
    overwrite: bool = False,
) -> None:
    """Register a sync chunker factory under ``name``."""
    key = name.lower()
    if not overwrite and key in CHUNKER_REGISTRY:
        raise ValueError(f"Chunker '{name}' already registered. Pass overwrite=True to replace.")
    CHUNKER_REGISTRY[key] = factory


def register_async_chunker(
    name: str,
    factory: Callable[..., AsyncTextChunker],
    overwrite: bool = False,
) -> None:
    """Register an async chunker factory under ``name``."""
    key = name.lower()
    if not overwrite and key in ASYNC_CHUNKER_REGISTRY:
        raise ValueError(f"Async chunker '{name}' already registered. Pass overwrite=True to replace.")
    ASYNC_CHUNKER_REGISTRY[key] = factory


# ── Factory functions ────────────────────────────────────────────────────────


def get_chunker(name: str, **kwargs: Any) -> TextChunker:
    """Instantiate a sync chunker by registered name with ``**kwargs``."""
    key = name.lower()
    if key not in CHUNKER_REGISTRY:
        available = ", ".join(sorted(CHUNKER_REGISTRY)) or "<none>"
        raise KeyError(f"No sync chunker registered under '{name}'. Available: {available}")
    return CHUNKER_REGISTRY[key](**kwargs)


def get_async_chunker(name: str, **kwargs: Any) -> AsyncTextChunker:
    """Instantiate an async chunker by registered name with ``**kwargs``."""
    key = name.lower()
    if key not in ASYNC_CHUNKER_REGISTRY:
        available = ", ".join(sorted(ASYNC_CHUNKER_REGISTRY)) or "<none>"
        raise KeyError(f"No async chunker registered under '{name}'. Available: {available}")
    return ASYNC_CHUNKER_REGISTRY[key](**kwargs)
