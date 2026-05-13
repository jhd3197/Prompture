"""Qdrant vector store adapter."""

from __future__ import annotations

import uuid
from typing import Any

from ..documents import Document
from .base import VectorSearchResult, VectorStore

_QDRANT_HINT = "QdrantVectorStore requires 'qdrant-client'. Install with: pip install 'prompture[rag-vs-qdrant]'"


def _require_qdrant():
    try:
        import qdrant_client  # type: ignore
        from qdrant_client.http import models as qmodels  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_QDRANT_HINT) from e
    return qdrant_client, qmodels


_DISTANCE_MAP = {
    "cosine": "Cosine",
    "euclid": "Euclid",
    "l2": "Euclid",
    "dot": "Dot",
}


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store.

    Qdrant returns similarity scores per the configured metric (e.g.
    cosine similarity for ``distance="cosine"``); higher = more similar.
    """

    def __init__(
        self,
        embedding_driver: Any = None,
        collection_name: str = "prompture",
        url: str | None = None,
        api_key: str | None = None,
        host: str = "localhost",
        port: int = 6333,
        prefer_grpc: bool = False,
        vector_size: int | None = None,
        distance: str = "cosine",
        client: Any = None,
        **kwargs: Any,
    ):
        super().__init__(embedding_driver=embedding_driver, **kwargs)
        self.collection_name = collection_name
        self.distance = distance
        self.vector_size = vector_size
        if client is not None:
            self._client = client
            self._qmodels = None
        else:
            qdrant_client, qmodels = _require_qdrant()
            self._qmodels = qmodels
            if url:
                self._client = qdrant_client.QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
            else:
                self._client = qdrant_client.QdrantClient(
                    host=host, port=port, api_key=api_key, prefer_grpc=prefer_grpc
                )
        self._collection_ready = False

    # ── Collection bootstrap ─────────────────────────────────────────────────

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_ready:
            return
        try:
            existing = self._client.get_collection(self.collection_name)
            if existing is not None:
                self._collection_ready = True
                return
        except Exception:
            pass
        # Need to create
        if self._qmodels is not None:
            self._client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=self._qmodels.VectorParams(
                    size=vector_size,
                    distance=getattr(
                        self._qmodels.Distance,
                        _DISTANCE_MAP.get(self.distance.lower(), "Cosine").upper(),
                    ),
                ),
            )
        else:
            # Caller passed a mock client; fall back to a plain dict spec.
            self._client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": vector_size, "distance": self.distance},
            )
        self._collection_ready = True

    def _infer_vector_size(self, vectors: list[list[float]]) -> int:
        if self.vector_size is not None:
            return self.vector_size
        if vectors:
            self.vector_size = len(vectors[0])
            return self.vector_size
        raise ValueError("Cannot determine vector_size; pass it to the constructor.")

    # ── Writes ───────────────────────────────────────────────────────────────

    def _build_points(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[Document],
    ):
        points = []
        for _id, vec, doc in zip(ids, vectors, documents):
            payload = dict(doc.metadata)
            payload["_content"] = doc.content
            if self._qmodels is not None:
                points.append(self._qmodels.PointStruct(id=_id, vector=list(vec), payload=payload))
            else:
                points.append({"id": _id, "vector": list(vec), "payload": payload})
        return points

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        if self.embedding_driver is None:
            raise ValueError("QdrantVectorStore.add_documents requires an embedding_driver")
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        vectors = self.embedding_driver.embed([d.content for d in documents])
        size = self._infer_vector_size(vectors)
        self._ensure_collection(size)
        points = self._build_points(ids, vectors, documents)
        self._client.upsert(collection_name=self.collection_name, points=points)
        return ids

    def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        size = self._infer_vector_size(vectors)
        self._ensure_collection(size)
        points = self._build_points(ids, vectors, documents)
        self._client.upsert(collection_name=self.collection_name, points=points)
        return ids

    # ── Reads ────────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        if self.embedding_driver is None:
            raise ValueError("QdrantVectorStore.similarity_search requires an embedding_driver")
        vec = self.embedding_driver.embed([query])[0]
        return self.similarity_search_by_vector(vec, k=k, filter=filter)

    def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=list(vector),
            limit=k,
            query_filter=filter,
            with_payload=True,
            with_vectors=True,
        )
        out: list[VectorSearchResult] = []
        for pt in results or []:
            payload = getattr(pt, "payload", None) or (pt.get("payload") if isinstance(pt, dict) else {}) or {}
            score = getattr(pt, "score", None)
            if score is None and isinstance(pt, dict):
                score = pt.get("score", 0.0)
            stored_vec = getattr(pt, "vector", None)
            if stored_vec is None and isinstance(pt, dict):
                stored_vec = pt.get("vector")
            content = payload.pop("_content", "") if isinstance(payload, dict) else ""
            out.append(
                VectorSearchResult(
                    document=Document(content=content, metadata=dict(payload)),
                    score=float(score or 0.0),
                    vector=list(stored_vec) if stored_vec is not None else None,
                )
            )
        return out

    # ── Deletes ──────────────────────────────────────────────────────────────

    def delete(self, ids: list[str]) -> None:
        if self._qmodels is not None:
            selector = self._qmodels.PointIdsList(points=ids)
            self._client.delete(collection_name=self.collection_name, points_selector=selector)
        else:
            self._client.delete(collection_name=self.collection_name, points_selector={"points": ids})
