"""Maximal Marginal Relevance (MMR) retriever.

Uses :meth:`VectorStore.max_marginal_relevance_search` to balance
relevance and diversity in the retrieved set.
"""

from __future__ import annotations

from typing import Any

from ..vectorstores.base import VectorSearchResult, VectorStore
from .base import Retriever


class MMRRetriever(Retriever):
    """Retriever using Maximal Marginal Relevance re-ranking.

    Fetches ``fetch_k`` candidates by similarity, then re-ranks to return
    ``k`` documents that balance relevance (similarity to the query) with
    diversity (dissimilarity to already-picked results).

    Args:
        vector_store: The backing vector store (must expose an embedding
            driver — MMR re-embeds the query to compute candidate similarity).
        k: Default top-K to return after MMR re-ranking.
        fetch_k: Default candidate pool size to fetch before re-ranking.
        lambda_mult: Trade-off between relevance (``1.0``) and diversity
            (``0.0``).  Default ``0.5`` weights them equally.
        filter: Default metadata filter (overridable per ``retrieve`` call).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.filter = filter

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        effective_k = k if k is not None else self.k
        effective_filter = filter if filter is not None else self.filter
        fetch_k = options.get("fetch_k", self.fetch_k)
        lambda_mult = options.get("lambda_mult", self.lambda_mult)
        return self.vector_store.max_marginal_relevance_search(
            query,
            k=effective_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=effective_filter,
        )
