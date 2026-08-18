"""Weaviate (v4 client) vector store adapter."""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

from ..documents import Document
from .base import VectorSearchResult, VectorStore, embed_texts

_WEAVIATE_HINT = (
    "WeaviateVectorStore requires 'weaviate-client>=4'. Install with: pip install 'prompture[rag-vs-weaviate]'"
)


def _require_weaviate():
    try:
        import weaviate  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_WEAVIATE_HINT) from e
    return weaviate


class WeaviateVectorStore(VectorStore):
    """Weaviate-backed vector store using the v4 client.

    Weaviate returns a ``distance`` (lower = closer) for vector
    searches; we convert to similarity via ``1 - distance``.
    """

    def __init__(
        self,
        embedding_driver: Any = None,
        class_name: str = "PromptureDoc",
        url: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        **kwargs: Any,
    ):
        super().__init__(embedding_driver=embedding_driver, **kwargs)
        self.class_name = class_name
        if client is not None:
            self._client = client
        else:
            weaviate = _require_weaviate()
            if url:
                # connect_to_wcs / connect_to_local depending on URL — defer to
                # user-supplied client for non-default deployments.
                try:
                    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
                    self._client = weaviate.connect_to_wcs(cluster_url=url, auth_credentials=auth)
                except Exception:
                    self._client = weaviate.connect_to_local()
            else:
                self._client = weaviate.connect_to_local()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        import contextlib

        with contextlib.suppress(Exception):
            if self._client.collections.exists(self.class_name):
                return
        # If the mock or backend doesn't support create, ignore — the
        # caller is expected to have set up the schema.
        with contextlib.suppress(Exception):
            self._client.collections.create(name=self.class_name)

    def _coll(self):
        return self._client.collections.get(self.class_name)

    # ── Writes ───────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        if self.embedding_driver is None:
            raise ValueError("WeaviateVectorStore.add_documents requires an embedding_driver")
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        vectors = embed_texts(self.embedding_driver, [d.content for d in documents])
        return self.add_vectors(vectors, documents, ids=ids)

    def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        coll = self._coll()
        for _id, vec, doc in zip(ids, vectors, documents, strict=False):
            props = {"content": doc.content, "metadata": doc.metadata}
            coll.data.insert(properties=props, uuid=_id, vector=list(vec))
        return ids

    # ── Reads ────────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        if self.embedding_driver is None:
            raise ValueError("WeaviateVectorStore.similarity_search requires an embedding_driver")
        vec = embed_texts(self.embedding_driver, [query])[0]
        return self.similarity_search_by_vector(vec, k=k, filter=filter)

    def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        coll = self._coll()
        kwargs: dict[str, Any] = {"near_vector": list(vector), "limit": k}
        if filter:
            kwargs["filters"] = filter
        resp = coll.query.near_vector(**kwargs)
        objects = getattr(resp, "objects", None)
        if objects is None and isinstance(resp, dict):
            objects = resp.get("objects", [])
        out: list[VectorSearchResult] = []
        for obj in objects or []:
            props = getattr(obj, "properties", None)
            if props is None and isinstance(obj, dict):
                props = obj.get("properties", {})
            props = props or {}
            metadata = props.get("metadata", {}) if isinstance(props, dict) else {}
            content = props.get("content", "") if isinstance(props, dict) else ""
            md_obj = getattr(obj, "metadata", None)
            distance = None
            if md_obj is not None:
                distance = getattr(md_obj, "distance", None)
            if distance is None and isinstance(obj, dict):
                distance = (obj.get("metadata") or {}).get("distance")
            score = 1.0 - float(distance) if distance is not None else 0.0
            stored_vec = getattr(obj, "vector", None)
            if stored_vec is None and isinstance(obj, dict):
                stored_vec = obj.get("vector")
            out.append(
                VectorSearchResult(
                    document=Document(content=content or "", metadata=dict(metadata or {})),
                    score=score,
                    vector=list(stored_vec) if stored_vec else None,
                )
            )
        return out

    # ── Deletes ──────────────────────────────────────────────────────────────

    def delete(self, ids: list[str]) -> None:
        coll = self._coll()
        for _id in ids:
            with contextlib.suppress(Exception):
                coll.data.delete_by_id(_id)
