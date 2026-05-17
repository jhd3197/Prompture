"""FAISS vector store adapter.

FAISS itself stores only vectors + integer IDs; we maintain a sidecar
dict ``{int_id: (Document, vector, string_id)}`` to retain Documents
and string IDs.
"""

from __future__ import annotations

import pickle  # nosec B403 - pickle used for own sidecar files, not user input
import uuid
from pathlib import Path
from typing import Any

from ..documents import Document
from .base import VectorSearchResult, VectorStore

_FAISS_HINT = (
    "FAISSVectorStore requires 'faiss-cpu' (or 'faiss-gpu'). Install with: pip install 'prompture[rag-vs-faiss]'"
)


def _require_faiss():
    try:
        import faiss  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_FAISS_HINT) from e
    return faiss


def _l2_normalize(v: list[float]) -> list[float]:
    s = sum(x * x for x in v) ** 0.5
    if s == 0:
        return list(v)
    return [x / s for x in v]


class FAISSVectorStore(VectorStore):
    """In-memory FAISS vector store with optional disk persistence.

    ``index_type``:
      * ``"L2"``  → ``IndexFlatL2`` (distance, similarity = ``-distance``)
      * ``"IP"`` / ``"cosine"`` → ``IndexFlatIP`` (vectors normalized).
    """

    def __init__(
        self,
        embedding_driver: Any = None,
        index: Any = None,
        index_type: str = "L2",
        vector_size: int | None = None,
        persist_directory: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(embedding_driver=embedding_driver, **kwargs)
        self.index_type = index_type.upper()
        self.vector_size = vector_size
        self.persist_directory = persist_directory
        # int_id → (string_id, Document, vector)
        self._docstore: dict[int, tuple[str, Document, list[float]]] = {}
        self._next_int_id = 0
        self._faiss = _require_faiss()
        if index is not None:
            self._index = index
        elif vector_size is not None:
            self._index = self._build_index(vector_size)
        else:
            self._index = None  # built lazily on first add
        # Try restore
        if persist_directory and self._index is None:
            self._try_restore()

    # ── Index lifecycle ──────────────────────────────────────────────────────

    def _build_index(self, dim: int):
        faiss = self._faiss
        if self.index_type in ("IP", "COSINE"):
            base = faiss.IndexFlatIP(dim)
        else:
            base = faiss.IndexFlatL2(dim)
        return faiss.IndexIDMap(base)

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self.vector_size = dim
            self._index = self._build_index(dim)

    def _prep_vec(self, v: list[float]) -> list[float]:
        if self.index_type in ("IP", "COSINE"):
            return _l2_normalize(v)
        return list(v)

    # ── Writes ───────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        if self.embedding_driver is None:
            raise ValueError("FAISSVectorStore.add_documents requires an embedding_driver")
        vectors = self.embedding_driver.embed([d.content for d in documents])
        return self.add_vectors(vectors, documents, ids=ids)

    def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        if not vectors:
            return []
        dim = len(vectors[0])
        self._ensure_index(dim)
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        int_ids: list[int] = []
        prepped: list[list[float]] = []
        for sid, vec, doc in zip(ids, vectors, documents):
            int_id = self._next_int_id
            self._next_int_id += 1
            pvec = self._prep_vec(vec)
            self._docstore[int_id] = (sid, doc, pvec)
            int_ids.append(int_id)
            prepped.append(pvec)
        # FAISS needs numpy arrays
        try:
            import numpy as np  # type: ignore

            arr = np.array(prepped, dtype="float32")
            id_arr = np.array(int_ids, dtype="int64")
            self._index.add_with_ids(arr, id_arr)
        except ImportError as e:  # pragma: no cover
            raise ImportError("FAISSVectorStore requires numpy") from e
        if self.persist_directory:
            self._persist()
        return ids

    # ── Reads ────────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        if self.embedding_driver is None:
            raise ValueError("FAISSVectorStore.similarity_search requires an embedding_driver")
        vec = self.embedding_driver.embed([query])[0]
        return self.similarity_search_by_vector(vec, k=k, filter=filter)

    def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import numpy as np  # type: ignore

        q = np.array([self._prep_vec(vector)], dtype="float32")
        scores, ids = self._index.search(q, min(k, self._index.ntotal))
        out: list[VectorSearchResult] = []
        for raw_score, raw_id in zip(scores[0].tolist(), ids[0].tolist()):
            if raw_id == -1:
                continue
            entry = self._docstore.get(int(raw_id))
            if entry is None:
                continue
            _sid, doc, stored_vec = entry
            if filter and not all(doc.metadata.get(k2) == v for k2, v in filter.items()):
                continue
            # L2 → similarity = -distance; IP → already similarity.
            score = float(raw_score) if self.index_type in ("IP", "COSINE") else -float(raw_score)
            out.append(VectorSearchResult(document=doc, score=score, vector=list(stored_vec)))
        return out

    # ── Deletes ──────────────────────────────────────────────────────────────

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        ids_set = set(ids)
        to_remove_int = [iid for iid, (sid, _d, _v) in self._docstore.items() if sid in ids_set]
        if not to_remove_int:
            return
        import numpy as np  # type: ignore

        if self._index is not None and hasattr(self._index, "remove_ids"):
            self._index.remove_ids(np.array(to_remove_int, dtype="int64"))
        for iid in to_remove_int:
            self._docstore.pop(iid, None)
        if self.persist_directory:
            self._persist()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self) -> None:
        if not self.persist_directory or self._index is None:
            return
        p = Path(self.persist_directory)
        p.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(p / "index.faiss"))
        with open(p / "docstore.pkl", "wb") as f:
            pickle.dump(
                {
                    "docstore": self._docstore,
                    "next_int_id": self._next_int_id,
                    "index_type": self.index_type,
                    "vector_size": self.vector_size,
                },
                f,
            )

    def _try_restore(self) -> None:
        if not self.persist_directory:
            return
        p = Path(self.persist_directory)
        idx_path = p / "index.faiss"
        ds_path = p / "docstore.pkl"
        if not idx_path.exists() or not ds_path.exists():
            return
        self._index = self._faiss.read_index(str(idx_path))
        with open(ds_path, "rb") as f:
            data = pickle.load(f)  # nosec B301 - loading own persisted state
        self._docstore = data.get("docstore", {})
        self._next_int_id = data.get("next_int_id", 0)
        self.index_type = data.get("index_type", self.index_type)
        self.vector_size = data.get("vector_size", self.vector_size)

    def __len__(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)
