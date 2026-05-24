"""Prompture RAG retrievers.

Concrete implementations:

* :class:`VectorStoreRetriever` — thin wrapper over
  :meth:`VectorStore.similarity_search`.
* :class:`MMRRetriever` — Maximal Marginal Relevance re-ranking for
  diverse results.
* :class:`HybridRetriever` — fuses dense (vector) + sparse (BM25)
  retrieval via Reciprocal Rank Fusion.
* :class:`HyDERetriever` — LLM drafts a hypothetical answer; the base
  retriever embeds and looks up neighbours of that text.
* :class:`MultiQueryRetriever` — LLM emits N reformulations; each is
  retrieved and the rankings are RRF-fused.
* :class:`StepBackRetriever` — LLM rephrases the query into a broader
  one; original and broader rankings are RRF-fused.

All retrievers share the :class:`Retriever` / :class:`AsyncRetriever`
interface defined in :mod:`prompture.rag.retrievers.base` and return
unified :class:`VectorSearchResult` objects so downstream consumers
don't need to know which retrieval strategy produced them.
"""

from __future__ import annotations

from .base import AsyncRetriever, Retriever
from .hybrid import HybridRetriever, reciprocal_rank_fusion
from .hyde import DEFAULT_HYDE_PROMPT, HyDERetriever
from .mmr import MMRRetriever
from .multi_query import DEFAULT_MULTI_QUERY_PROMPT, MultiQueryRetriever
from .retriever_registry import (
    ASYNC_RETRIEVER_REGISTRY,
    RETRIEVER_REGISTRY,
    get_async_retriever,
    get_retriever,
    register_async_retriever,
    register_retriever,
)
from .similarity import VectorStoreRetriever
from .step_back import DEFAULT_STEP_BACK_PROMPT, StepBackRetriever

# ── Register built-in retrievers ─────────────────────────────────────────────


def _similarity_factory(**kwargs):
    return VectorStoreRetriever(**kwargs)


def _mmr_factory(**kwargs):
    return MMRRetriever(**kwargs)


def _hybrid_factory(**kwargs):
    return HybridRetriever(**kwargs)


def _hyde_factory(**kwargs):
    return HyDERetriever(**kwargs)


def _multi_query_factory(**kwargs):
    return MultiQueryRetriever(**kwargs)


def _step_back_factory(**kwargs):
    return StepBackRetriever(**kwargs)


register_retriever("similarity", _similarity_factory, overwrite=True)
register_retriever("vector", _similarity_factory, overwrite=True)
register_retriever("mmr", _mmr_factory, overwrite=True)
register_retriever("hybrid", _hybrid_factory, overwrite=True)
register_retriever("hyde", _hyde_factory, overwrite=True)
register_retriever("multi_query", _multi_query_factory, overwrite=True)
register_retriever("multi-query", _multi_query_factory, overwrite=True)
register_retriever("step_back", _step_back_factory, overwrite=True)
register_retriever("step-back", _step_back_factory, overwrite=True)


__all__ = [
    "ASYNC_RETRIEVER_REGISTRY",
    "DEFAULT_HYDE_PROMPT",
    "DEFAULT_MULTI_QUERY_PROMPT",
    "DEFAULT_STEP_BACK_PROMPT",
    "RETRIEVER_REGISTRY",
    "AsyncRetriever",
    "HyDERetriever",
    "HybridRetriever",
    "MMRRetriever",
    "MultiQueryRetriever",
    "Retriever",
    "StepBackRetriever",
    "VectorStoreRetriever",
    "get_async_retriever",
    "get_retriever",
    "reciprocal_rank_fusion",
    "register_async_retriever",
    "register_retriever",
]
