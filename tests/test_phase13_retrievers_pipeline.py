"""Phase 13 — RAG retrievers and end-to-end pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from prompture.rag import (
    RETRIEVER_REGISTRY,
    AsyncRAGPipeline,
    Document,
    HybridRetriever,
    MMRRetriever,
    RAGAnswer,
    RAGPipeline,
    Retriever,
    VectorSearchResult,
    VectorStoreRetriever,
    get_retriever,
)
from prompture.rag.retrievers.hybrid import _tokenize, reciprocal_rank_fusion

# ── Helpers ──────────────────────────────────────────────────────────────────


def _result(content: str, score: float = 1.0, **md) -> VectorSearchResult:
    return VectorSearchResult(document=Document(content=content, metadata=md), score=score)


def _fake_vector_store(results: list[VectorSearchResult]) -> MagicMock:
    store = MagicMock()
    store.similarity_search.return_value = list(results)
    store.max_marginal_relevance_search.return_value = list(results)
    store.add_documents.return_value = [f"id-{i}" for i in range(len(results))]
    return store


# ── VectorStoreRetriever ─────────────────────────────────────────────────────


def test_vector_store_retriever_passes_through():
    results = [_result("a", 0.9), _result("b", 0.5)]
    store = _fake_vector_store(results)
    retriever = VectorStoreRetriever(store, k=2)

    out = retriever.retrieve("query")

    store.similarity_search.assert_called_once_with("query", k=2, filter=None)
    assert out == results


def test_vector_store_retriever_score_threshold_filters():
    results = [_result("a", 0.9), _result("b", 0.2), _result("c", 0.5)]
    store = _fake_vector_store(results)
    retriever = VectorStoreRetriever(store, k=10, score_threshold=0.4)

    out = retriever.retrieve("q")
    assert [r.score for r in out] == [0.9, 0.5]


def test_vector_store_retriever_call_overrides_k_and_filter():
    store = _fake_vector_store([])
    retriever = VectorStoreRetriever(store, k=4, filter={"a": 1})

    retriever.retrieve("q", k=7, filter={"b": 2})
    store.similarity_search.assert_called_with("q", k=7, filter={"b": 2})


def test_vector_store_retriever_callable_alias():
    results = [_result("x", 1.0)]
    store = _fake_vector_store(results)
    retriever = VectorStoreRetriever(store, k=1)
    assert retriever("query") == results


# ── MMRRetriever ─────────────────────────────────────────────────────────────


def test_mmr_retriever_calls_max_marginal_relevance_search():
    results = [_result("a"), _result("b")]
    store = _fake_vector_store(results)
    retriever = MMRRetriever(store, k=2, fetch_k=10, lambda_mult=0.7)

    out = retriever.retrieve("q")
    store.max_marginal_relevance_search.assert_called_once_with("q", k=2, fetch_k=10, lambda_mult=0.7, filter=None)
    assert out == results


# ── HybridRetriever / RRF ────────────────────────────────────────────────────


def test_rrf_basic_math():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a"]], rrf_k=60)
    # a: 1/61 + 1/62, b: 1/62 + 1/61, c: 1/63
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    # a and b tied (different rank in each list), both above c
    assert fused["a"] > fused["c"]


def test_rrf_handles_unique_doc():
    fused = reciprocal_rank_fusion([["a"], ["b"]], rrf_k=60)
    # Each in only one list — both at rank 1 → same score
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 61)


def test_hybrid_retriever_fuses_dense_and_sparse():
    pytest.importorskip("rank_bm25")
    corpus = [
        Document(content="the cat sat on the mat", metadata={"id": "1"}),
        Document(content="dogs love bones", metadata={"id": "2"}),
        Document(content="cats are mammals", metadata={"id": "3"}),
        Document(content="bones are calcium", metadata={"id": "4"}),
    ]
    # Dense returns docs in one order
    dense_results = [
        _result(corpus[3].content, 0.9, id="4"),
        _result(corpus[1].content, 0.8, id="2"),
    ]
    store = _fake_vector_store(dense_results)

    retriever = HybridRetriever(store, corpus=corpus, k=3, alpha=0.5)
    out = retriever.retrieve("cats")
    # BM25 ranking for "cats" should surface doc 1 / 3.
    contents = [r.document.content for r in out]
    assert any("cat" in c.lower() for c in contents)
    # All returned items must be VectorSearchResult
    assert all(isinstance(r, VectorSearchResult) for r in out)
    # k respected
    assert len(out) <= 3


def test_hybrid_retriever_requires_corpus_or_index():
    store = _fake_vector_store([])
    with pytest.raises(ValueError, match="corpus"):
        HybridRetriever(store)


def test_hybrid_retriever_empty_corpus_still_returns_dense():
    pytest.importorskip("rank_bm25")
    dense_results = [_result("x", 0.9, id="x")]
    store = _fake_vector_store(dense_results)
    # Empty corpus: BM25 index is None internally
    retriever = HybridRetriever(store, corpus=[], k=1, alpha=0.5)
    out = retriever.retrieve("anything")
    # Dense-only fallback path
    assert len(out) == 1
    assert out[0].document.content == "x"


def test_hybrid_retriever_doc_in_only_one_ranking():
    pytest.importorskip("rank_bm25")
    # Use a larger corpus so BM25 IDF is non-zero for the matching term.
    corpus = [
        Document(content="alpha beta gamma", metadata={"id": "1"}),
        Document(content="completely unrelated text", metadata={"id": "2"}),
        Document(content="more unrelated content", metadata={"id": "3"}),
        Document(content="yet more random words", metadata={"id": "4"}),
        Document(content="and still more filler", metadata={"id": "5"}),
    ]
    # Dense returns only doc 2 (some embedding artifact)
    store = _fake_vector_store([_result(corpus[1].content, 0.7, id="2")])
    retriever = HybridRetriever(store, corpus=corpus, k=2, alpha=0.5)
    out = retriever.retrieve("alpha")
    # Doc 1 from sparse-only, doc 2 from dense-only — both should appear
    contents = {r.document.content for r in out}
    assert "alpha beta gamma" in contents
    assert "completely unrelated text" in contents


def test_tokenize_is_lowercase_whitespace():
    assert _tokenize("Hello WORLD foo") == ["hello", "world", "foo"]


# ── Registry ─────────────────────────────────────────────────────────────────


def test_retriever_registry_get_similarity():
    store = _fake_vector_store([])
    r = get_retriever("similarity", vector_store=store, k=5)
    assert isinstance(r, VectorStoreRetriever)
    assert r.k == 5


def test_retriever_registry_get_mmr():
    store = _fake_vector_store([])
    r = get_retriever("mmr", vector_store=store, k=3, fetch_k=15)
    assert isinstance(r, MMRRetriever)
    assert r.fetch_k == 15


def test_retriever_registry_known_keys():
    assert "similarity" in RETRIEVER_REGISTRY
    assert "mmr" in RETRIEVER_REGISTRY
    assert "hybrid" in RETRIEVER_REGISTRY
    assert "vector" in RETRIEVER_REGISTRY


def test_retriever_registry_unknown_raises():
    with pytest.raises(KeyError, match="bogus"):
        get_retriever("bogus")


# ── RAGPipeline.query ────────────────────────────────────────────────────────


class _FakeLLM:
    """Minimal LLM driver double."""

    model_name = "fake/model"

    def __init__(self, text: str = "the answer", usage: dict | None = None):
        self.text = text
        self.usage_blob = usage or {"prompt_tokens": 10, "completion_tokens": 5}
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, options: dict):
        self.calls.append((prompt, options))
        return {
            "text": self.text,
            "usage": self.usage_blob,
        }


class _FakeRetriever(Retriever):
    def __init__(self, results: list[VectorSearchResult]):
        self.results = results
        self.calls = 0

    def retrieve(self, query, k=4, filter=None, **options):
        self.calls += 1
        self.last_query = query
        self.last_k = k
        return list(self.results)


def test_rag_pipeline_query_basic():
    results = [_result("doc A", 0.9), _result("doc B", 0.8)]
    retriever = _FakeRetriever(results)
    llm = _FakeLLM(text="hello world")

    pipeline = RAGPipeline(retriever=retriever, llm=llm)
    answer = pipeline.query("what is X?", k=2)

    assert isinstance(answer, RAGAnswer)
    assert answer.answer == "hello world"
    assert len(answer.sources) == 2
    assert answer.retrieval_results == results
    assert answer.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    # Prompt template should have both context and question filled in.
    prompt, _opts = llm.calls[0]
    assert "doc A" in prompt
    assert "doc B" in prompt
    assert "what is X?" in prompt


def test_rag_pipeline_query_return_sources_false():
    retriever = _FakeRetriever([_result("doc A")])
    pipeline = RAGPipeline(retriever=retriever, llm=_FakeLLM())
    answer = pipeline.query("q", return_sources=False)
    assert answer.sources == []
    # retrieval_results still populated
    assert len(answer.retrieval_results) == 1


def test_rag_pipeline_query_custom_template_and_system_prompt():
    retriever = _FakeRetriever([_result("ctx-1")])
    llm = _FakeLLM()
    template = "SYS\nCtx:{context}\nQ:{question}"
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=llm,
        prompt_template=template,
        system_prompt="be terse",
    )
    pipeline.query("why?", k=1)
    prompt, opts = llm.calls[0]
    assert prompt.startswith("SYS")
    assert "ctx-1" in prompt
    assert opts == {"system_prompt": "be terse"}


# ── Reranker integration ────────────────────────────────────────────────────


def test_rag_pipeline_reranker_reorders_results():
    results = [
        _result("first", 0.9),
        _result("second", 0.8),
        _result("third", 0.7),
        _result("fourth", 0.6),
    ]
    retriever = _FakeRetriever(results)

    # Reranker returns RerankResult-like objects with .index pointing to inputs
    class _RR:
        def __init__(self, index, score):
            self.index = index
            self.relevance_score = score

    reranker = MagicMock()
    reranker.rerank.return_value = [_RR(2, 0.99), _RR(0, 0.5)]  # reverse-ish

    llm = _FakeLLM(text="ans")
    pipeline = RAGPipeline(retriever=retriever, llm=llm, reranker=reranker, top_n_after_rerank=2)

    answer = pipeline.query("q", k=4)
    # Reranker called with original doc contents
    call_kwargs = reranker.rerank.call_args.kwargs
    assert call_kwargs["query"] == "q"
    assert call_kwargs["documents"] == ["first", "second", "third", "fourth"]
    assert call_kwargs["top_n"] == 2

    # After rerank, results are reordered: index 2 then 0
    assert [r.document.content for r in answer.retrieval_results] == ["third", "first"]


# ── RAGPipeline.extract ──────────────────────────────────────────────────────


class _Person(BaseModel):
    name: str
    age: int


def test_rag_pipeline_extract_returns_tuple(monkeypatch):
    retriever = _FakeRetriever([_result("Maria is 32 years old")])
    llm = _FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    fake_instance = _Person(name="Maria", age=32)
    fake_result = {
        "model": fake_instance,
        "usage": {"total_tokens": 42, "cost": 0.001},
    }

    captured = {}

    def _fake_extract(model_cls, text, **kwargs):
        captured["model_cls"] = model_cls
        captured["text"] = text
        captured["kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr("prompture.extraction.core.extract_with_model", _fake_extract)

    instance, rag_answer = pipeline.extract("Who is mentioned?", _Person, k=1)

    assert instance is fake_instance
    assert isinstance(rag_answer, RAGAnswer)
    assert "Maria" in rag_answer.answer  # JSON serialization
    assert rag_answer.usage == {"total_tokens": 42, "cost": 0.001}
    assert captured["model_cls"] is _Person
    assert "Maria is 32" in captured["text"]
    assert captured["kwargs"]["model_name"] == "fake/model"
    assert captured["kwargs"]["instruction_template"] == "Who is mentioned?"


def test_rag_pipeline_extract_requires_model_name():
    retriever = _FakeRetriever([_result("ctx")])

    class _NoModel:
        def generate(self, p, o):  # pragma: no cover - never called
            return {"text": ""}

    pipeline = RAGPipeline(retriever=retriever, llm=_NoModel())
    with pytest.raises(ValueError, match="model_name"):
        pipeline.extract("q", _Person)


# ── RAGPipeline.ingest ───────────────────────────────────────────────────────


def test_rag_pipeline_ingest_chains_calls():
    docs_loaded = [Document(content="raw doc", metadata={"source": "x"})]
    chunks = [
        Document(content="raw", metadata={"source": "x"}),
        Document(content="doc", metadata={"source": "x"}),
    ]

    loader = MagicMock()
    loader.load.return_value = docs_loaded
    chunker = MagicMock()
    chunker.split_documents.return_value = chunks
    vector_store = MagicMock()
    vector_store.add_documents.return_value = ["id-0", "id-1"]

    retriever = _FakeRetriever([])
    pipeline = RAGPipeline(retriever=retriever, llm=_FakeLLM())

    out = pipeline.ingest("something.txt", loader=loader, chunker=chunker, vector_store=vector_store)

    loader.load.assert_called_once_with("something.txt")
    chunker.split_documents.assert_called_once_with(docs_loaded)
    vector_store.add_documents.assert_called_once_with(chunks, ids=None)
    assert out == ["id-0", "id-1"]


def test_rag_pipeline_ingest_uses_retriever_vector_store():
    docs_loaded = [Document(content="raw doc")]
    chunks = [Document(content="raw doc")]

    loader = MagicMock()
    loader.load.return_value = docs_loaded
    chunker = MagicMock()
    chunker.split_documents.return_value = chunks

    vector_store = _fake_vector_store([])
    retriever = VectorStoreRetriever(vector_store, k=4)
    pipeline = RAGPipeline(retriever=retriever, llm=_FakeLLM())

    out = pipeline.ingest("any.txt", loader=loader, chunker=chunker)
    vector_store.add_documents.assert_called_once_with(chunks, ids=None)
    assert out == ["id-0"] or out == []  # MagicMock-style default


def test_rag_pipeline_ingest_no_vector_store_available_raises():
    retriever = _FakeRetriever([])  # no .vector_store attr
    pipeline = RAGPipeline(retriever=retriever, llm=_FakeLLM())
    loader = MagicMock()
    loader.load.return_value = []
    chunker = MagicMock()
    chunker.split_documents.return_value = []
    with pytest.raises(ValueError, match="vector_store"):
        pipeline.ingest("x.txt", loader=loader, chunker=chunker)


# ── AsyncRAGPipeline ─────────────────────────────────────────────────────────


def test_async_rag_pipeline_aquery_smoke():
    retriever = _FakeRetriever([_result("ctx", 0.9)])
    llm = _FakeLLM(text="async-answer")
    pipeline = AsyncRAGPipeline(retriever=retriever, llm=llm)

    answer = asyncio.run(pipeline.aquery("hello?"))
    assert isinstance(answer, RAGAnswer)
    assert answer.answer == "async-answer"
    assert len(answer.sources) == 1
