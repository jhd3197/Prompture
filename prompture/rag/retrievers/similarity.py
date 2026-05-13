"""Similarity-search retriever — thin wrapper over ``VectorStore.similarity_search``."""

from __future__ import annotations

from typing import Any

from ..vectorstores.base import VectorSearchResult, VectorStore
from .base import Retriever


class VectorStoreRetriever(Retriever):
    """Retriever backed by a :class:`VectorStore`.

    Thin wrapper over :meth:`VectorStore.similarity_search` that adds
    optional default ``k``, ``filter``, and ``score_threshold`` knobs.

    Args:
        vector_store: The backing vector store.
        k: Default top-K to return.
        filter: Default metadata filter (overridable per ``retrieve`` call).
        score_threshold: If set, results with ``score`` below this value
            are dropped after retrieval.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        k: int = 4,
        filter: dict | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.k = k
        self.filter = filter
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        effective_k = k if k is not None else self.k
        effective_filter = filter if filter is not None else self.filter
        results = self.vector_store.similarity_search(query, k=effective_k, filter=effective_filter)
        if self.score_threshold is not None:
            results = [r for r in results if r.score >= self.score_threshold]
        return results
