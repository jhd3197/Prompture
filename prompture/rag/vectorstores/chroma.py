"""Chroma vector store adapter.

Wraps the ``chromadb`` client.  Lazy-imports the backend so callers
that don't use Chroma incur no import cost and don't need the package
installed.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..documents import Document
from .base import VectorSearchResult, VectorStore

_CHROMA_HINT = "ChromaVectorStore requires 'chromadb'. Install with: pip install 'prompture[rag-vs-chroma]'"


def _require_chromadb():
    try:
        import chromadb  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised via test mocks
        raise ImportError(_CHROMA_HINT) from e
    return chromadb


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store.

    Distance → similarity convention: Chroma returns L2 / cosine
    *distances* (lower = closer); we map to similarity via
    ``1 - distance`` so callers see "higher = more similar" uniformly.
    """

    def __init__(
        self,
        embedding_driver: Any = None,
        collection_name: str = "prompture",
        persist_directory: str | None = None,
        client: Any = None,
        **kwargs: Any,
    ):
        super().__init__(embedding_driver=embedding_driver, **kwargs)
        if client is not None:
            self._client = client
        else:
            chromadb = _require_chromadb()
            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                # In-memory ephemeral client
                if hasattr(chromadb, "EphemeralClient"):
                    self._client = chromadb.EphemeralClient()
                else:  # pragma: no cover - older chromadb
                    self._client = chromadb.Client()
        self.collection_name = collection_name
        self._collection = self._client.get_or_create_collection(name=collection_name)

    # ── Writes ───────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        if self.embedding_driver is None:
            raise ValueError("ChromaVectorStore.add_documents requires an embedding_driver")
        texts = [d.content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        vectors = self.embedding_driver.embed(texts)
        self._collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        return ids

    def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        texts = [d.content for d in documents]
        metadatas = [d.metadata for d in documents]
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        self._collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        return ids

    # ── Reads ────────────────────────────────────────────────────────────────

    def _query(self, query_embeddings: list[list[float]], k: int, filter: dict | None):
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": k,
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if filter:
            kwargs["where"] = filter
        return self._collection.query(**kwargs)

    @staticmethod
    def _distance_to_similarity(d: float) -> float:
        return 1.0 - float(d)

    def _map_results(self, result: dict) -> list[VectorSearchResult]:
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        embs = (result.get("embeddings") or [[None] * len(docs)])[0]
        out: list[VectorSearchResult] = []
        for content, meta, dist, emb in zip(docs, metas, dists, embs):
            out.append(
                VectorSearchResult(
                    document=Document(content=content or "", metadata=dict(meta or {})),
                    score=self._distance_to_similarity(dist),
                    vector=list(emb) if emb is not None else None,
                )
            )
        return out

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        if self.embedding_driver is None:
            raise ValueError("ChromaVectorStore.similarity_search requires an embedding_driver")
        query_vec = self.embedding_driver.embed([query])[0]
        return self.similarity_search_by_vector(query_vec, k=k, filter=filter)

    def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        result = self._query([vector], k=k, filter=filter)
        return self._map_results(result)

    # ── Deletes ──────────────────────────────────────────────────────────────

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    def __len__(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as e:  # pragma: no cover
            raise NotImplementedError from e
