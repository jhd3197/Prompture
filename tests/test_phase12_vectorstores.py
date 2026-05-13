"""Phase 12 — RAG vector stores."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import MagicMock, patch

import pytest

from prompture.rag import (
    ASYNC_VECTORSTORE_REGISTRY,
    VECTORSTORE_REGISTRY,
    ChromaVectorStore,
    Document,
    FAISSVectorStore,
    PgVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    VectorSearchResult,
    VectorStore,
    WeaviateVectorStore,
    get_vectorstore,
)
from prompture.rag.vectorstores.base import (
    _cosine_sim_py,
    _mmr_select,
    _SyncToAsyncVectorStore,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """Deterministic embedder: maps each text to a fixed vector by lookup."""

    def __init__(self, mapping: dict[str, list[float]] | None = None, dim: int = 3):
        self.mapping = mapping or {}
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for t in texts:
            if t in self.mapping:
                out.append(list(self.mapping[t]))
            else:
                # Hash-derived deterministic vector
                h = abs(hash(t))
                vec = [(h >> (i * 4)) % 7 / 7.0 for i in range(self.dim)]
                out.append(vec)
        return out


# ── VectorSearchResult ───────────────────────────────────────────────────────


def test_vector_search_result_dataclass():
    doc = Document(content="hello", metadata={"a": 1})
    r = VectorSearchResult(document=doc, score=0.9, vector=[0.1, 0.2])
    assert r.document is doc
    assert r.score == 0.9
    assert r.vector == [0.1, 0.2]
    r2 = VectorSearchResult(document=doc, score=0.5)
    assert r2.vector is None


# ── MMR algorithm ────────────────────────────────────────────────────────────


def test_cosine_sim_py():
    assert _cosine_sim_py([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine_sim_py([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _cosine_sim_py([], [1.0]) == 0.0


def test_mmr_selects_diverse_results():
    # Query points along x-axis. Three candidates: two very close along x,
    # one somewhat-similar-but-diverse. With a low lambda_mult
    # (diversity-weighted), the second pick should be the diverse one,
    # not the near-duplicate.
    query = [1.0, 0.0]
    cands = [
        [1.0, 0.0],  # very close to query
        [0.99, 0.05],  # nearly identical to first cand
        [0.2, 1.0],  # somewhat similar to query but mostly diverse
    ]
    picked = _mmr_select(query, cands, k=2, lambda_mult=0.2)
    assert picked[0] == 0  # most similar first
    assert picked[1] == 2  # diverse second, not the near-duplicate


def test_mmr_pure_python_fallback():
    """Force ImportError on numpy to exercise the pure-Python branch."""
    query = [1.0, 0.0]
    cands = [[1.0, 0.0], [0.99, 0.05], [0.2, 1.0]]
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        picked = _mmr_select(query, cands, k=2, lambda_mult=0.2)
    assert picked == [0, 2]


def test_base_mmr_via_store():
    """MMR through the base class against a mock store."""

    class _MockStore(VectorStore):
        def __init__(self, embedder, candidates):
            super().__init__(embedding_driver=embedder)
            self._candidates = candidates

        def add_documents(self, documents, ids=None):
            return []

        def add_vectors(self, vectors, documents, ids=None):
            return []

        def similarity_search(self, query, k=4, filter=None):
            return self._candidates[:k]

        def similarity_search_by_vector(self, vector, k=4, filter=None):
            return self._candidates[:k]

        def delete(self, ids):
            pass

    embedder = _FakeEmbedder({"q": [1.0, 0.0]})
    cands = [
        VectorSearchResult(Document("A"), 0.99, [1.0, 0.0]),
        VectorSearchResult(Document("B"), 0.95, [0.99, 0.05]),
        VectorSearchResult(Document("C"), 0.30, [0.2, 1.0]),
    ]
    store = _MockStore(embedder, cands)
    out = store.max_marginal_relevance_search("q", k=2, fetch_k=3, lambda_mult=0.2)
    assert len(out) == 2
    assert out[0].document.content == "A"
    assert out[1].document.content == "C"


def test_base_mmr_requires_embedding_driver():
    class _Empty(VectorStore):
        def add_documents(self, documents, ids=None):
            return []

        def add_vectors(self, vectors, documents, ids=None):
            return []

        def similarity_search(self, query, k=4, filter=None):
            return []

        def similarity_search_by_vector(self, vector, k=4, filter=None):
            return []

        def delete(self, ids):
            pass

    store = _Empty()
    with pytest.raises(ValueError, match="embedding_driver"):
        store.max_marginal_relevance_search("q")


# ── Chroma ───────────────────────────────────────────────────────────────────


def _mock_chroma_client():
    client = MagicMock()
    collection = MagicMock()
    collection.add = MagicMock()
    collection.query = MagicMock(
        return_value={
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"a": 1}, {"a": 2}]],
            "distances": [[0.1, 0.4]],
            "embeddings": [[[0.1, 0.2], [0.3, 0.4]]],
        }
    )
    collection.delete = MagicMock()
    collection.count = MagicMock(return_value=2)
    client.get_or_create_collection = MagicMock(return_value=collection)
    return client, collection


def test_chroma_add_and_search():
    client, collection = _mock_chroma_client()
    embedder = _FakeEmbedder({"q": [1.0, 0.0]}, dim=2)
    store = ChromaVectorStore(embedding_driver=embedder, client=client)
    ids = store.add_documents([Document("hi", {"a": 1}), Document("yo", {"a": 2})])
    assert len(ids) == 2
    assert collection.add.called
    call_kwargs = collection.add.call_args.kwargs
    assert "embeddings" in call_kwargs
    assert len(call_kwargs["embeddings"]) == 2

    results = store.similarity_search("q", k=2)
    assert collection.query.called
    assert len(results) == 2
    # Chroma distance 0.1 → similarity 0.9
    assert results[0].score == pytest.approx(0.9)
    assert results[0].document.content == "doc1"
    assert results[0].vector == [0.1, 0.2]


def test_chroma_delete_and_len():
    client, collection = _mock_chroma_client()
    store = ChromaVectorStore(embedding_driver=_FakeEmbedder(), client=client)
    store.delete(["a", "b"])
    collection.delete.assert_called_once_with(ids=["a", "b"])
    assert len(store) == 2


def test_chroma_add_documents_requires_embedder():
    client, _ = _mock_chroma_client()
    store = ChromaVectorStore(client=client)
    with pytest.raises(ValueError, match="embedding_driver"):
        store.add_documents([Document("hi")])


def test_chroma_import_error_when_backend_missing():
    """Patch sys.modules so 'chromadb' import fails."""
    with patch.dict(sys.modules, {"chromadb": None}):
        with pytest.raises(ImportError, match="rag-vs-chroma"):
            ChromaVectorStore(embedding_driver=_FakeEmbedder())


# ── Pinecone ─────────────────────────────────────────────────────────────────


def _mock_pinecone_client():
    client = MagicMock()
    index = MagicMock()
    index.upsert = MagicMock()
    index.query = MagicMock(
        return_value={
            "matches": [
                {
                    "id": "i1",
                    "score": 0.92,
                    "values": [0.1, 0.2],
                    "metadata": {"_content": "hello", "a": 1},
                },
                {
                    "id": "i2",
                    "score": 0.55,
                    "values": [0.3, 0.4],
                    "metadata": {"_content": "world", "a": 2},
                },
            ]
        }
    )
    index.delete = MagicMock()
    client.Index = MagicMock(return_value=index)
    return client, index


def test_pinecone_add_and_search():
    client, index = _mock_pinecone_client()
    embedder = _FakeEmbedder({"q": [1.0, 0.0]}, dim=2)
    store = PineconeVectorStore(embedding_driver=embedder, index_name="idx", client=client)
    ids = store.add_documents([Document("hello", {"a": 1}), Document("world", {"a": 2})])
    assert len(ids) == 2
    index.upsert.assert_called_once()
    upsert_kwargs = index.upsert.call_args.kwargs
    assert "vectors" in upsert_kwargs
    assert len(upsert_kwargs["vectors"]) == 2

    results = store.similarity_search("q", k=2)
    assert results[0].score == pytest.approx(0.92)
    assert results[0].document.content == "hello"
    assert results[0].document.metadata.get("a") == 1
    assert "_content" not in results[0].document.metadata


def test_pinecone_requires_index_name():
    with pytest.raises(ValueError, match="index_name"):
        PineconeVectorStore(embedding_driver=_FakeEmbedder(), client=MagicMock())


def test_pinecone_delete():
    client, index = _mock_pinecone_client()
    store = PineconeVectorStore(embedding_driver=_FakeEmbedder(), index_name="idx", client=client)
    store.delete(["i1", "i2"])
    index.delete.assert_called_once()


def test_pinecone_import_error():
    with patch.dict(sys.modules, {"pinecone": None}):
        with pytest.raises(ImportError, match="rag-vs-pinecone"):
            PineconeVectorStore(embedding_driver=_FakeEmbedder(), index_name="idx")


# ── Qdrant ───────────────────────────────────────────────────────────────────


def _mock_qdrant_client():
    client = MagicMock()
    client.get_collection = MagicMock(side_effect=Exception("missing"))
    client.recreate_collection = MagicMock()
    client.upsert = MagicMock()
    point = MagicMock()
    point.payload = {"_content": "doc1", "a": 1}
    point.score = 0.88
    point.vector = [0.1, 0.2]
    point2 = MagicMock()
    point2.payload = {"_content": "doc2", "a": 2}
    point2.score = 0.50
    point2.vector = [0.3, 0.4]
    client.search = MagicMock(return_value=[point, point2])
    client.delete = MagicMock()
    return client


def test_qdrant_add_infers_vector_size_and_creates_collection():
    client = _mock_qdrant_client()
    embedder = _FakeEmbedder({"hello": [1.0, 0.0], "world": [0.0, 1.0]}, dim=2)
    store = QdrantVectorStore(embedding_driver=embedder, client=client)
    ids = store.add_documents([Document("hello", {"a": 1}), Document("world", {"a": 2})])
    assert len(ids) == 2
    assert store.vector_size == 2
    client.recreate_collection.assert_called_once()
    client.upsert.assert_called_once()


def test_qdrant_similarity_search():
    client = _mock_qdrant_client()
    embedder = _FakeEmbedder({"q": [1.0, 0.0]}, dim=2)
    store = QdrantVectorStore(embedding_driver=embedder, client=client, vector_size=2)
    results = store.similarity_search("q", k=2)
    assert client.search.called
    assert results[0].score == pytest.approx(0.88)
    assert results[0].document.content == "doc1"
    assert "_content" not in results[0].document.metadata


def test_qdrant_delete():
    client = _mock_qdrant_client()
    store = QdrantVectorStore(embedding_driver=_FakeEmbedder(), client=client, vector_size=2)
    store.delete(["a", "b"])
    client.delete.assert_called_once()


def test_qdrant_import_error():
    with patch.dict(sys.modules, {"qdrant_client": None}):
        with pytest.raises(ImportError, match="rag-vs-qdrant"):
            QdrantVectorStore(embedding_driver=_FakeEmbedder(), vector_size=4)


# ── pgvector ─────────────────────────────────────────────────────────────────


def _mock_pg_connection():
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    cur.execute = MagicMock()
    cur.fetchall = MagicMock(
        return_value=[
            ("id1", "doc1", {"a": 1}, "[0.1,0.2]", 0.95),
            ("id2", "doc2", '{"a": 2}', "[0.3,0.4]", 0.60),
        ]
    )
    conn.cursor = MagicMock(return_value=cur)
    return conn, cur


def test_pgvector_add_creates_table():
    conn, cur = _mock_pg_connection()
    embedder = _FakeEmbedder({"hello": [1.0, 0.0]}, dim=2)
    store = PgVectorStore(embedding_driver=embedder, connection=conn)
    store.add_documents([Document("hello", {"a": 1})])
    # CREATE EXTENSION + CREATE TABLE + INSERT
    executed = [c.args[0] for c in cur.execute.call_args_list]
    assert any("CREATE EXTENSION" in s for s in executed)
    assert any("CREATE TABLE IF NOT EXISTS" in s for s in executed)
    assert any("INSERT INTO" in s for s in executed)
    conn.commit.assert_called()


def test_pgvector_similarity_search_cosine_operator():
    conn, cur = _mock_pg_connection()
    embedder = _FakeEmbedder({"q": [1.0, 0.0]}, dim=2)
    store = PgVectorStore(embedding_driver=embedder, connection=conn, vector_size=2, distance="cosine")
    results = store.similarity_search("q", k=2)
    # Find the SELECT among execute calls
    select_calls = [c for c in cur.execute.call_args_list if "SELECT" in str(c.args[0])]
    assert select_calls, "Expected a SELECT call"
    sql = select_calls[-1].args[0]
    assert "<=>" in sql  # cosine operator
    assert "1 - (embedding <=>" in sql
    assert len(results) == 2
    assert results[0].score == pytest.approx(0.95)
    assert results[0].document.content == "doc1"
    assert results[1].document.metadata == {"a": 2}


def test_pgvector_requires_connection():
    with pytest.raises(ValueError, match="connection"):
        PgVectorStore(embedding_driver=_FakeEmbedder())


def test_pgvector_invalid_distance():
    conn, _ = _mock_pg_connection()
    with pytest.raises(ValueError, match="distance"):
        PgVectorStore(embedding_driver=_FakeEmbedder(), connection=conn, distance="weird")


def test_pgvector_delete():
    conn, cur = _mock_pg_connection()
    store = PgVectorStore(embedding_driver=_FakeEmbedder(), connection=conn, vector_size=2)
    store.delete(["a", "b"])
    deletes = [c for c in cur.execute.call_args_list if "DELETE" in str(c.args[0])]
    assert deletes


def test_pgvector_import_error():
    with patch.dict(sys.modules, {"psycopg2": None}):
        with pytest.raises(ImportError, match="rag-vs-pgvector"):
            PgVectorStore(
                embedding_driver=_FakeEmbedder(),
                connection_string="postgresql://x",
            )


# ── FAISS (real backend if available) ────────────────────────────────────────


def test_faiss_end_to_end():
    pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    embedder = _FakeEmbedder(
        {
            "apple": [1.0, 0.0, 0.0],
            "banana": [0.0, 1.0, 0.0],
            "cherry": [0.0, 0.0, 1.0],
            "fruit?": [1.0, 0.1, 0.0],
        },
        dim=3,
    )
    store = FAISSVectorStore(embedding_driver=embedder, index_type="IP")
    docs = [Document("apple"), Document("banana"), Document("cherry")]
    ids = store.add_documents(docs)
    assert len(ids) == 3
    assert len(store) == 3
    results = store.similarity_search("fruit?", k=3)
    assert results[0].document.content == "apple"

    # delete one
    store.delete([ids[0]])
    assert len(store) == 2


def test_faiss_import_error():
    # Save real faiss if present, then mask
    saved = sys.modules.pop("faiss", None)
    try:
        with patch.dict(sys.modules, {"faiss": None}):
            with pytest.raises(ImportError, match="rag-vs-faiss"):
                FAISSVectorStore(embedding_driver=_FakeEmbedder(), vector_size=3)
    finally:
        if saved is not None:
            sys.modules["faiss"] = saved


# ── Weaviate ─────────────────────────────────────────────────────────────────


def _mock_weaviate_client():
    client = MagicMock()
    collections = MagicMock()
    coll = MagicMock()
    coll.data.insert = MagicMock()
    coll.data.delete_by_id = MagicMock()
    obj1 = MagicMock()
    obj1.properties = {"content": "doc1", "metadata": {"a": 1}}
    md1 = MagicMock()
    md1.distance = 0.1
    obj1.metadata = md1
    obj1.vector = [0.1, 0.2]
    obj2 = MagicMock()
    obj2.properties = {"content": "doc2", "metadata": {"a": 2}}
    md2 = MagicMock()
    md2.distance = 0.4
    obj2.metadata = md2
    obj2.vector = [0.3, 0.4]
    resp = MagicMock()
    resp.objects = [obj1, obj2]
    coll.query.near_vector = MagicMock(return_value=resp)
    collections.get = MagicMock(return_value=coll)
    collections.exists = MagicMock(return_value=True)
    collections.create = MagicMock()
    client.collections = collections
    return client, coll


def test_weaviate_add_and_search():
    client, coll = _mock_weaviate_client()
    embedder = _FakeEmbedder({"q": [1.0, 0.0]}, dim=2)
    store = WeaviateVectorStore(embedding_driver=embedder, client=client)
    ids = store.add_documents([Document("hello", {"a": 1}), Document("world", {"a": 2})])
    assert len(ids) == 2
    assert coll.data.insert.call_count == 2
    results = store.similarity_search("q", k=2)
    assert results[0].score == pytest.approx(0.9)
    assert results[0].document.content == "doc1"


def test_weaviate_delete():
    client, coll = _mock_weaviate_client()
    store = WeaviateVectorStore(embedding_driver=_FakeEmbedder(), client=client)
    store.delete(["a", "b"])
    assert coll.data.delete_by_id.call_count == 2


def test_weaviate_import_error():
    with patch.dict(sys.modules, {"weaviate": None}):
        with pytest.raises(ImportError, match="rag-vs-weaviate"):
            WeaviateVectorStore(embedding_driver=_FakeEmbedder())


# ── Registry ─────────────────────────────────────────────────────────────────


def test_registry_contains_all_six():
    for name in ("chroma", "pinecone", "qdrant", "pgvector", "faiss", "weaviate"):
        assert name in VECTORSTORE_REGISTRY
        assert name in ASYNC_VECTORSTORE_REGISTRY


def test_get_vectorstore_chroma_with_mock_client():
    client, _ = _mock_chroma_client()
    store = get_vectorstore("chroma", embedding_driver=_FakeEmbedder(), client=client)
    assert isinstance(store, ChromaVectorStore)


def test_get_vectorstore_unknown():
    with pytest.raises(KeyError):
        get_vectorstore("nonesuch")


# ── Async wrapper smoke test ─────────────────────────────────────────────────


def test_async_wrapper_smoke():
    client, _collection = _mock_chroma_client()
    embedder = _FakeEmbedder({"q": [1.0, 0.0]}, dim=2)
    sync_store = ChromaVectorStore(embedding_driver=embedder, client=client)
    async_store = _SyncToAsyncVectorStore(sync_store)

    async def _run():
        ids = await async_store.add_documents([Document("hi", {"a": 1})])
        assert len(ids) == 1
        results = await async_store.similarity_search("q", k=2)
        assert len(results) == 2
        await async_store.delete(["x"])

    asyncio.run(_run())


def test_async_registry_factory():
    factory = ASYNC_VECTORSTORE_REGISTRY["chroma"]
    client, _ = _mock_chroma_client()
    async_store = factory(embedding_driver=_FakeEmbedder(), client=client)
    assert isinstance(async_store, _SyncToAsyncVectorStore)
