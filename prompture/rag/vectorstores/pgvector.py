"""pgvector (PostgreSQL) vector store adapter."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..documents import Document
from .base import VectorSearchResult, VectorStore, embed_texts

_PGVECTOR_HINT = (
    "PgVectorStore requires 'psycopg2-binary' and 'pgvector'. Install with: pip install 'prompture[rag-vs-pgvector]'"
)


def _require_psycopg2():
    try:
        import psycopg2  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_PGVECTOR_HINT) from e
    return psycopg2


_DISTANCE_OPERATORS = {
    "cosine": "<=>",
    "l2": "<->",
    "euclid": "<->",
    "ip": "<#>",
    "dot": "<#>",
}


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class PgVectorStore(VectorStore):
    """PostgreSQL + pgvector-backed vector store.

    Auto-creates the table on first write if it doesn't exist.  For
    ``distance="cosine"`` we return ``1 - (embedding <=> q)`` as the
    similarity score (higher = more similar).
    """

    def __init__(
        self,
        embedding_driver: Any = None,
        connection_string: str | None = None,
        table_name: str = "prompture_vectors",
        vector_size: int | None = None,
        distance: str = "cosine",
        connection: Any = None,
        **kwargs: Any,
    ):
        super().__init__(embedding_driver=embedding_driver, **kwargs)
        if connection is None and connection_string is None:
            raise ValueError("PgVectorStore requires 'connection_string' or 'connection'")
        self.table_name = table_name
        self.vector_size = vector_size
        self.distance = distance.lower()
        if self.distance not in _DISTANCE_OPERATORS:
            raise ValueError(f"Unknown distance metric: {distance}")
        if connection is not None:
            self._conn = connection
        else:
            psycopg2 = _require_psycopg2()
            self._conn = psycopg2.connect(connection_string)
        self._table_ready = False

    # ── Schema bootstrap ─────────────────────────────────────────────────────

    def _ensure_table(self, vector_size: int) -> None:
        if self._table_ready:
            return
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector({vector_size})
                );
                """
            )
        self._conn.commit()
        self._table_ready = True

    def _infer_vector_size(self, vectors: list[list[float]]) -> int:
        if self.vector_size is not None:
            return self.vector_size
        if vectors:
            self.vector_size = len(vectors[0])
            return self.vector_size
        raise ValueError("Cannot determine vector_size; pass it to the constructor.")

    # ── Writes ───────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        if self.embedding_driver is None:
            raise ValueError("PgVectorStore.add_documents requires an embedding_driver")
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
        size = self._infer_vector_size(vectors)
        self._ensure_table(size)
        with self._conn.cursor() as cur:
            for _id, vec, doc in zip(ids, vectors, documents, strict=False):
                cur.execute(
                    f"""
                    INSERT INTO {self.table_name} (id, content, metadata, embedding)
                    VALUES (%s, %s, %s::jsonb, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                    """,  # nosec B608 - table_name is library-controlled config, values parameterized
                    (_id, doc.content, json.dumps(doc.metadata), _vec_literal(vec)),
                )
        self._conn.commit()
        return ids

    # ── Reads ────────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 4, filter: dict | None = None) -> list[VectorSearchResult]:
        if self.embedding_driver is None:
            raise ValueError("PgVectorStore.similarity_search requires an embedding_driver")
        vec = embed_texts(self.embedding_driver, [query])[0]
        return self.similarity_search_by_vector(vec, k=k, filter=filter)

    def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, filter: dict | None = None
    ) -> list[VectorSearchResult]:
        op = _DISTANCE_OPERATORS[self.distance]
        # For cosine, similarity = 1 - distance.  For L2 / ip we leave the raw
        # operator value but invert the sign for ip so "higher = better".
        score_expr = f"1 - (embedding {op} %s)" if self.distance == "cosine" else f"-(embedding {op} %s)"
        where_clause = ""
        params: list[Any] = [_vec_literal(vector)]
        if filter:
            conditions = []
            for key, val in filter.items():
                conditions.append("metadata ->> %s = %s")
                params.extend([key, str(val)])
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
        params.append(_vec_literal(vector))  # for ORDER BY
        params.append(k)
        # nosec B608 - table_name/op/score_expr are library-controlled; user values parameterized
        sql = (
            f"SELECT id, content, metadata, embedding, {score_expr} AS score "  # nosec B608
            f"FROM {self.table_name} {where_clause} "  # nosec B608
            f"ORDER BY embedding {op} %s LIMIT %s"  # nosec B608
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        out: list[VectorSearchResult] = []
        for row in rows or []:
            # row: (id, content, metadata, embedding, score)
            _id, content, metadata, embedding, score = row
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except Exception:
                    metadata = {}
            elif metadata is None:
                metadata = {}
            # embedding may come back as a string literal "[..]" — keep as-is or None.
            vec_out: list[float] | None = None
            if isinstance(embedding, (list, tuple)):
                vec_out = [float(x) for x in embedding]
            out.append(
                VectorSearchResult(
                    document=Document(content=content or "", metadata=dict(metadata)),
                    score=float(score) if score is not None else 0.0,
                    vector=vec_out,
                )
            )
        return out

    # ── Deletes ──────────────────────────────────────────────────────────────

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",  # nosec B608 - table_name is library-controlled config
                (list(ids),),
            )
        self._conn.commit()
