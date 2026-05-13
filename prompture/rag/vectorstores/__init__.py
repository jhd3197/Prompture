"""Prompture RAG vector stores.

Phase 12 adds vector store abstractions plus six concrete adapters:

* :class:`ChromaVectorStore` — local ChromaDB (in-memory or persistent).
* :class:`PineconeVectorStore` — managed Pinecone (v3 SDK).
* :class:`QdrantVectorStore` — Qdrant (local / Qdrant Cloud).
* :class:`PgVectorStore` — PostgreSQL + ``pgvector`` extension.
* :class:`FAISSVectorStore` — local FAISS index with optional disk persistence.
* :class:`WeaviateVectorStore` — Weaviate (v4 client).

All adapters share the :class:`VectorStore` / :class:`AsyncVectorStore`
interface defined in :mod:`prompture.rag.vectorstores.base` and return
unified :class:`VectorSearchResult` objects.  The base class provides a
shared :meth:`VectorStore.max_marginal_relevance_search` implementation
(numpy-accelerated, with pure-Python fallback).

Backend libraries are lazily imported — listing all stores from this
package does not require any of them to be installed.  Each store
raises :class:`ImportError` with a clear ``prompture[rag-vs-*]``
extras hint when its backend is missing.
"""

from __future__ import annotations

from .base import (
    AsyncVectorStore,
    VectorSearchResult,
    VectorStore,
    _SyncToAsyncVectorStore,
)
from .chroma import ChromaVectorStore
from .faiss import FAISSVectorStore
from .pgvector import PgVectorStore
from .pinecone import PineconeVectorStore
from .qdrant import QdrantVectorStore
from .vectorstore_registry import (
    ASYNC_VECTORSTORE_REGISTRY,
    VECTORSTORE_REGISTRY,
    get_async_vectorstore,
    get_vectorstore,
    register_async_vectorstore,
    register_vectorstore,
)
from .weaviate import WeaviateVectorStore

# ── Register built-in vector stores ──────────────────────────────────────────

_BUILTIN_VECTORSTORES: dict[str, type[VectorStore]] = {
    "chroma": ChromaVectorStore,
    "pinecone": PineconeVectorStore,
    "qdrant": QdrantVectorStore,
    "pgvector": PgVectorStore,
    "faiss": FAISSVectorStore,
    "weaviate": WeaviateVectorStore,
}


def _make_sync_factory(cls: type[VectorStore]):
    def _factory(**kwargs):
        return cls(**kwargs)

    return _factory


def _make_async_factory(cls: type[VectorStore]):
    def _factory(**kwargs):
        return _SyncToAsyncVectorStore(cls(**kwargs))

    return _factory


for _name, _cls in _BUILTIN_VECTORSTORES.items():
    register_vectorstore(_name, _make_sync_factory(_cls), overwrite=True)
    register_async_vectorstore(_name, _make_async_factory(_cls), overwrite=True)

del _name, _cls

__all__ = [
    "ASYNC_VECTORSTORE_REGISTRY",
    "VECTORSTORE_REGISTRY",
    "AsyncVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "PgVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "VectorSearchResult",
    "VectorStore",
    "WeaviateVectorStore",
    "get_async_vectorstore",
    "get_vectorstore",
    "register_async_vectorstore",
    "register_vectorstore",
]
