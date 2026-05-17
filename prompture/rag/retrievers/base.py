"""Core retriever abstractions.

A :class:`Retriever` abstracts the lookup step of RAG: given a query
string, return ranked :class:`~prompture.rag.vectorstores.base.VectorSearchResult`
objects.  Implementations may use a vector store, BM25 keyword search,
hybrid fusion, or any other relevance strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..vectorstores.base import VectorSearchResult


class Retriever(ABC):
    """Retrieves relevant documents/chunks for a query.

    Retrievers abstract the lookup step: given a query string, return
    ranked :class:`VectorSearchResult` objects.  Implementations may use
    a vector store, BM25 keyword search, hybrid fusion, or any other
    relevance strategy.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        """Retrieve up to ``k`` results for ``query``."""
        raise NotImplementedError

    def __call__(self, query: str, k: int = 4, **options: Any) -> list[VectorSearchResult]:
        return self.retrieve(query, k=k, **options)


class AsyncRetriever(ABC):
    """Asynchronous counterpart to :class:`Retriever`."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        """Retrieve up to ``k`` results for ``query`` asynchronously."""
        raise NotImplementedError
