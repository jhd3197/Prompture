"""Core RAG vector store abstractions.

Defines :class:`VectorStore` and :class:`AsyncVectorStore` abstract base
classes together with the :class:`VectorSearchResult` dataclass and a
thin :class:`_SyncToAsyncVectorStore` adapter that wraps a sync store in
:func:`asyncio.to_thread` to provide trivial async support.

The base class implements :meth:`VectorStore.max_marginal_relevance_search`
once so subclasses only need to implement core CRUD + similarity_search.
MMR uses ``numpy`` when available and falls back to a pure-Python
implementation otherwise.
"""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..documents import Document

# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class VectorSearchResult:
    """A single result from a vector store similarity search.

    Attributes:
        document: The retrieved :class:`Document`.
        score: Similarity score.  Higher = more similar by convention.  Each
            adapter normalizes its backend's native distance/score to follow
            this convention and documents the conversion.
        vector: The stored vector if the backend returned it; ``None``
            otherwise.  Required for MMR re-ranking.
    """

    document: Document
    score: float
    vector: list[float] | None = None


# ── Cosine / MMR helpers ─────────────────────────────────────────────────────


def _cosine_sim_py(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (fallback when numpy is missing)."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _mmr_select(
    query_vec: list[float],
    candidate_vecs: list[list[float]],
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Greedy MMR selection. Returns indices into ``candidate_vecs``.

    Uses numpy if available; otherwise falls back to a pure-Python
    implementation.
    """
    if not candidate_vecs or k <= 0:
        return []
    k = min(k, len(candidate_vecs))

    try:
        import numpy as np  # type: ignore

        q = np.array(query_vec, dtype=float)
        cand = np.array(candidate_vecs, dtype=float)
        # Cosine similarity to query
        q_norm = np.linalg.norm(q)
        cand_norms = np.linalg.norm(cand, axis=1)
        denom = q_norm * cand_norms
        denom[denom == 0] = 1.0
        sim_to_query = cand.dot(q) / denom

        selected: list[int] = []
        remaining = list(range(len(candidate_vecs)))
        # Pick most similar first
        first = int(np.argmax(sim_to_query))
        selected.append(first)
        remaining.remove(first)

        while remaining and len(selected) < k:
            best_idx = remaining[0]
            best_score = -float("inf")
            for idx in remaining:
                # Max similarity to any already-picked
                sel_vecs = cand[selected]
                sel_norms = np.linalg.norm(sel_vecs, axis=1)
                target = cand[idx]
                target_norm = np.linalg.norm(target)
                denom2 = sel_norms * target_norm
                denom2[denom2 == 0] = 1.0
                sims_to_selected = sel_vecs.dot(target) / denom2
                max_sim_to_selected = float(np.max(sims_to_selected))
                mmr_score = lambda_mult * float(sim_to_query[idx]) - (1.0 - lambda_mult) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected
    except ImportError:
        # Pure-Python fallback
        sim_to_query = [_cosine_sim_py(query_vec, v) for v in candidate_vecs]
        selected = []
        remaining = list(range(len(candidate_vecs)))
        first = max(remaining, key=lambda i: sim_to_query[i])
        selected.append(first)
        remaining.remove(first)
        while remaining and len(selected) < k:
            best_idx = remaining[0]
            best_score = -float("inf")
            for idx in remaining:
                max_sim_to_selected = max(_cosine_sim_py(candidate_vecs[idx], candidate_vecs[s]) for s in selected)
                mmr_score = lambda_mult * sim_to_query[idx] - (1.0 - lambda_mult) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)
        return selected


# ── Sync base ────────────────────────────────────────────────────────────────


class VectorStore(ABC):
    """Synchronous vector store interface.

    Implementations typically need an embedding driver (passed at
    ``__init__``) to embed query strings transparently.  Pre-embedded
    vectors can be passed via :meth:`add_vectors` and
    :meth:`similarity_search_by_vector`.
    """

    def __init__(self, embedding_driver: Any = None, **kwargs: Any):
        self.embedding_driver = embedding_driver

    @abstractmethod
    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        """Embed each ``document.content`` and store with metadata."""
        raise NotImplementedError

    @abstractmethod
    def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Store pre-computed vectors paired with Documents."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        """Embed ``query`` and search."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        """Search using a pre-computed query vector."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete entries by ID."""
        raise NotImplementedError

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
    ) -> list[VectorSearchResult]:
        """MMR re-ranking on top of similarity search.

        Fetches ``fetch_k`` candidates by similarity, then re-ranks to
        balance relevance and diversity.  Default implementation works
        for any store that implements
        :meth:`similarity_search_by_vector` and returns vectors on the
        results.
        """
        if self.embedding_driver is None:
            raise ValueError("MMR requires an embedding_driver")
        query_vector = self.embedding_driver.embed([query])[0]
        candidates = self.similarity_search_by_vector(query_vector, k=fetch_k, filter=filter)
        # Filter out candidates without vectors — MMR needs them
        with_vectors = [c for c in candidates if c.vector is not None]
        if not with_vectors:
            # Fall back to plain similarity if backend didn't return vectors
            return candidates[:k]
        vecs = [c.vector for c in with_vectors]
        picked = _mmr_select(query_vector, vecs, k=k, lambda_mult=lambda_mult)
        return [with_vectors[i] for i in picked]

    def __len__(self) -> int:
        """Override if cheap; default raises ``NotImplementedError``."""
        raise NotImplementedError


# ── Async base ───────────────────────────────────────────────────────────────


class AsyncVectorStore(ABC):
    """Asynchronous counterpart to :class:`VectorStore`."""

    def __init__(self, embedding_driver: Any = None, **kwargs: Any):
        self.embedding_driver = embedding_driver

    @abstractmethod
    async def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    async def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    async def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: dict | None = None,
    ) -> list[VectorSearchResult]:
        if self.embedding_driver is None:
            raise ValueError("MMR requires an embedding_driver")
        # embedding_driver.embed may be sync; run it in a thread.
        query_vector = (await asyncio.to_thread(self.embedding_driver.embed, [query]))[0]
        candidates = await self.similarity_search_by_vector(query_vector, k=fetch_k, filter=filter)
        with_vectors = [c for c in candidates if c.vector is not None]
        if not with_vectors:
            return candidates[:k]
        vecs = [c.vector for c in with_vectors]
        picked = _mmr_select(query_vector, vecs, k=k, lambda_mult=lambda_mult)
        return [with_vectors[i] for i in picked]


# ── Sync→Async adapter ───────────────────────────────────────────────────────


class _SyncToAsyncVectorStore(AsyncVectorStore):
    """Wraps a sync :class:`VectorStore` in :func:`asyncio.to_thread`.

    Trivial async support for backends without native async clients.
    Native-async ports for Qdrant / Pinecone v3 / Weaviate v4 are a
    follow-up.
    """

    def __init__(self, sync_store: VectorStore):
        self._sync = sync_store
        self.embedding_driver = sync_store.embedding_driver

    async def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        return await asyncio.to_thread(self._sync.add_documents, documents, ids)

    async def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        return await asyncio.to_thread(self._sync.add_vectors, vectors, documents, ids)

    async def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        return await asyncio.to_thread(self._sync.similarity_search, query, k, filter)

    async def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        return await asyncio.to_thread(self._sync.similarity_search_by_vector, vector, k, filter)

    async def delete(self, ids: list[str]) -> None:
        await asyncio.to_thread(self._sync.delete, ids)
