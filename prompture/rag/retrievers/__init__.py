"""Prompture RAG retrievers.

Phase 13 adds retriever abstractions plus three concrete implementations:

* :class:`VectorStoreRetriever` — thin wrapper over
  :meth:`VectorStore.similarity_search`.
* :class:`MMRRetriever` — Maximal Marginal Relevance re-ranking for
  diverse results.
* :class:`HybridRetriever` — fuses dense (vector) + sparse (BM25)
  retrieval via Reciprocal Rank Fusion.

All retrievers share the :class:`Retriever` / :class:`AsyncRetriever`
interface defined in :mod:`prompture.rag.retrievers.base` and return
unified :class:`VectorSearchResult` objects so downstream consumers
don't need to know which retrieval strategy produced them.
"""

from __future__ import annotations

from .base import AsyncRetriever, Retriever
from .hybrid import HybridRetriever, reciprocal_rank_fusion
from .mmr import MMRRetriever
from .retriever_registry import (
    ASYNC_RETRIEVER_REGISTRY,
    RETRIEVER_REGISTRY,
    get_async_retriever,
    get_retriever,
    register_async_retriever,
    register_retriever,
)
from .similarity import VectorStoreRetriever

# ── Register built-in retrievers ─────────────────────────────────────────────


def _similarity_factory(**kwargs):
    return VectorStoreRetriever(**kwargs)


def _mmr_factory(**kwargs):
    return MMRRetriever(**kwargs)


def _hybrid_factory(**kwargs):
    return HybridRetriever(**kwargs)


register_retriever("similarity", _similarity_factory, overwrite=True)
register_retriever("vector", _similarity_factory, overwrite=True)
register_retriever("mmr", _mmr_factory, overwrite=True)
register_retriever("hybrid", _hybrid_factory, overwrite=True)


__all__ = [
    "ASYNC_RETRIEVER_REGISTRY",
    "RETRIEVER_REGISTRY",
    "AsyncRetriever",
    "HybridRetriever",
    "MMRRetriever",
    "Retriever",
    "VectorStoreRetriever",
    "get_async_retriever",
    "get_retriever",
    "reciprocal_rank_fusion",
    "register_async_retriever",
    "register_retriever",
]
