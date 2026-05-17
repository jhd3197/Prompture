"""Pinecone (v3 SDK) vector store adapter."""

from __future__ import annotations

import os
import uuid
from typing import Any

from ..documents import Document
from .base import VectorSearchResult, VectorStore

_PINECONE_HINT = (
    "PineconeVectorStore requires 'pinecone-client>=3'. Install with: pip install 'prompture[rag-vs-pinecone]'"
)


def _require_pinecone():
    try:
        import pinecone  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_PINECONE_HINT) from e
    return pinecone


class PineconeVectorStore(VectorStore):
    """Pinecone-backed vector store using the v3 SDK.

    Pinecone returns cosine similarity directly (or dot product / Euclidean
    depending on the index's metric); the score is forwarded as-is.
    """

    def __init__(
        self,
        embedding_driver: Any = None,
        index_name: str | None = None,
        api_key: str | None = None,
        environment: str | None = None,
        namespace: str | None = None,
        client: Any = None,
        **kwargs: Any,
    ):
        super().__init__(embedding_driver=embedding_driver, **kwargs)
        if index_name is None:
            raise ValueError("PineconeVectorStore requires an 'index_name'")
        self.index_name = index_name
        self.namespace = namespace
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        if client is not None:
            self._client = client
        else:
            pinecone = _require_pinecone()
            # v3 SDK: pinecone.Pinecone(api_key=..)
            self._client = pinecone.Pinecone(api_key=api_key, environment=environment)
        self._index = self._client.Index(index_name)

    # ── Writes ───────────────────────────────────────────────────────────────

    def _build_vectors(
        self,
        ids: list[str],
        vectors: list[list[float]],
        documents: list[Document],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for _id, vec, doc in zip(ids, vectors, documents):
            md = dict(doc.metadata)
            md.setdefault("_content", doc.content)
            items.append({"id": _id, "values": list(vec), "metadata": md})
        return items

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        if self.embedding_driver is None:
            raise ValueError("PineconeVectorStore.add_documents requires an embedding_driver")
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        vectors = self.embedding_driver.embed([d.content for d in documents])
        items = self._build_vectors(ids, vectors, documents)
        self._index.upsert(vectors=items, namespace=self.namespace)
        return ids

    def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        items = self._build_vectors(ids, vectors, documents)
        self._index.upsert(vectors=items, namespace=self.namespace)
        return ids

    # ── Reads ────────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        if self.embedding_driver is None:
            raise ValueError("PineconeVectorStore.similarity_search requires an embedding_driver")
        vec = self.embedding_driver.embed([query])[0]
        return self.similarity_search_by_vector(vec, k=k, filter=filter)

    def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        kwargs: dict[str, Any] = {
            "vector": list(vector),
            "top_k": k,
            "include_metadata": True,
            "include_values": True,
        }
        if self.namespace:
            kwargs["namespace"] = self.namespace
        if filter:
            kwargs["filter"] = filter
        resp = self._index.query(**kwargs)
        matches = resp.get("matches") if isinstance(resp, dict) else getattr(resp, "matches", [])
        out: list[VectorSearchResult] = []
        for m in matches or []:
            md = (m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", {})) or {}
            score = m.get("score") if isinstance(m, dict) else getattr(m, "score", 0.0)
            values = m.get("values") if isinstance(m, dict) else getattr(m, "values", None)
            content = md.pop("_content", "") if isinstance(md, dict) else ""
            out.append(
                VectorSearchResult(
                    document=Document(content=content, metadata=dict(md)),
                    score=float(score) if score is not None else 0.0,
                    vector=list(values) if values is not None else None,
                )
            )
        return out

    # ── Deletes ──────────────────────────────────────────────────────────────

    def delete(self, ids: list[str]) -> None:
        if self.namespace:
            self._index.delete(ids=ids, namespace=self.namespace)
        else:
            self._index.delete(ids=ids)
