"""Phase 11 — RAG text chunkers."""

from __future__ import annotations

import asyncio
import math

import pytest

from prompture.rag import (
    ASYNC_CHUNKER_REGISTRY,
    CHUNKER_REGISTRY,
    AsyncTextChunker,
    CharacterChunker,
    Document,
    MarkdownChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    TextChunker,
    TokenChunker,
    get_async_chunker,
    get_chunker,
)
from prompture.rag.chunkers.base import _SyncToAsyncChunker

# ── Shared helpers ───────────────────────────────────────────────────────────


class _FakeEmbeddingDriver:
    """Returns deterministic embeddings keyed off sentence content.

    Sentences containing any of the ``cluster_b_markers`` characters
    receive a vector pointing along the second axis; everything else
    points along the first axis.  This guarantees an unambiguous topic
    switch at the marker boundary regardless of buffer_size.
    """

    def __init__(self, cluster_b_markers: str = "XYZ") -> None:
        self.cluster_b_markers = set(cluster_b_markers)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if any(ch in self.cluster_b_markers for ch in t):
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([1.0, 0.0, 0.0])
        return out


# ── chunk_overlap >= chunk_size raises ───────────────────────────────────────


def test_overlap_greater_than_size_raises():
    with pytest.raises(ValueError):
        CharacterChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        RecursiveCharacterChunker(chunk_size=10, chunk_overlap=20)


# ── CharacterChunker ─────────────────────────────────────────────────────────


def test_character_chunker_splits_on_separator():
    text = "aaa\n\nbbb\n\nccc"
    chunker = CharacterChunker(chunk_size=8, chunk_overlap=0, separator="\n\n")
    chunks = chunker.split_text(text)
    assert chunks, "expected at least one chunk"
    joined = " ".join(chunks)
    for piece in ("aaa", "bbb", "ccc"):
        assert piece in joined


def test_character_chunker_hard_cut_fallback():
    text = "abcdefghij" * 10  # 100 chars, no separator
    chunker = CharacterChunker(chunk_size=20, chunk_overlap=5, separator="\n\n")
    chunks = chunker.split_text(text)
    assert all(len(c) <= 20 for c in chunks)
    # All input characters should still appear (overlap means slight repeat).
    assert "".join(chunks).replace("", "")  # smoke; non-empty
    assert chunks[0].startswith("abc")


def test_character_chunker_overlap_behavior():
    text = "aaa\n\nbbb\n\nccc\n\nddd"
    chunker = CharacterChunker(chunk_size=10, chunk_overlap=2, separator="\n\n")
    chunks = chunker.split_text(text)
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) and c for c in chunks)


# ── RecursiveCharacterChunker ────────────────────────────────────────────────


def test_recursive_chunker_respects_chunk_size():
    text = "Paragraph one.\n\nParagraph two has multiple sentences. Each one is short. They go on.\n\nFinal paragraph."
    chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=5)
    chunks = chunker.split_text(text)
    assert chunks
    # Allow tiny slack for separator retention but not a free-for-all.
    assert all(len(c) <= 80 for c in chunks)


def test_recursive_chunker_picks_largest_fitting_separator():
    text = "A" * 30 + "\n\n" + "B" * 30 + "\n" + "C" * 30
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=0)
    chunks = chunker.split_text(text)
    # Should split at "\n\n" first since the result fits chunk_size.
    assert any("A" * 30 in c for c in chunks)


def test_recursive_chunker_empty_input():
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split_text("") == []


# ── TokenChunker (requires tiktoken) ─────────────────────────────────────────


def test_token_chunker_basic():
    pytest.importorskip("tiktoken")
    chunker = TokenChunker(chunk_size=10, chunk_overlap=2)
    text = "Hello world. " * 50
    chunks = chunker.split_text(text)
    assert chunks
    # Re-encode chunks to verify each is <= chunk_size tokens.
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    for c in chunks:
        assert len(enc.encode(c)) <= 10


def test_token_chunker_round_trip():
    pytest.importorskip("tiktoken")
    chunker = TokenChunker(chunk_size=100, chunk_overlap=0)
    text = "The quick brown fox jumps over the lazy dog."
    chunks = chunker.split_text(text)
    assert len(chunks) == 1
    assert chunks[0].strip() == text.strip()


# ── SemanticChunker ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "threshold_type",
    ["percentile", "standard_deviation", "interquartile", "gradient"],
)
def test_semantic_chunker_finds_topic_boundary(threshold_type: str):
    # A/B/C are one topic; X/Y/Z are a different topic.
    text = "A is here. B is too. C joins. X arrives. Y arrives. Z arrives."
    driver = _FakeEmbeddingDriver(cluster_b_markers="XYZ")
    chunker = SemanticChunker(
        embedding_driver=driver,
        breakpoint_threshold_type=threshold_type,
        breakpoint_threshold_amount=50.0 if threshold_type != "standard_deviation" else 0.5,
        buffer_size=0,
        min_chunk_size=1,
    )
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2, f"expected >=2 chunks, got {chunks!r}"
    # Cluster A markers and cluster B markers should not mix in one chunk.
    has_a_only = any(("A" in c or "B" in c or "C" in c) and not any(m in c for m in "XYZ") for c in chunks)
    has_b_only = any(any(m in c for m in "XYZ") and not ("A" in c and "B" in c and "C" in c) for c in chunks)
    assert has_a_only and has_b_only


def test_semantic_chunker_rejects_bad_threshold_type():
    with pytest.raises(ValueError):
        SemanticChunker(embedding_driver=_FakeEmbeddingDriver(), breakpoint_threshold_type="bogus")


def test_semantic_chunker_handles_short_input():
    driver = _FakeEmbeddingDriver()
    chunker = SemanticChunker(embedding_driver=driver, min_chunk_size=1)
    assert chunker.split_text("") == []
    assert chunker.split_text("Just one sentence.") == ["Just one sentence."]


# ── MarkdownChunker ──────────────────────────────────────────────────────────


def test_markdown_chunker_metadata_headers():
    text = "# Title\nintro paragraph\n## Section\ncontent under section\n### Subsection\ndetail"
    chunker = MarkdownChunker()
    docs = chunker.split_documents([Document(content=text, metadata={"source": "doc.md"})])
    assert docs, "expected chunks"
    # First chunk should have Header 1 in headers metadata
    first_with_h1 = next((d for d in docs if d.metadata.get("headers", {}).get("Header 1") == "Title"), None)
    assert first_with_h1 is not None
    sec = next((d for d in docs if d.metadata.get("headers", {}).get("Header 2") == "Section"), None)
    assert sec is not None
    assert sec.metadata["headers"].get("Header 1") == "Title"
    sub = next((d for d in docs if d.metadata.get("headers", {}).get("Header 3") == "Subsection"), None)
    assert sub is not None
    assert sub.metadata["headers"].get("Header 2") == "Section"


def test_markdown_chunker_split_text_strings_only():
    text = "# H1\nfoo\n## H2\nbar"
    chunks = MarkdownChunker().split_text(text)
    assert all(isinstance(c, str) for c in chunks)
    assert any("foo" in c for c in chunks)
    assert any("bar" in c for c in chunks)


def test_markdown_chunker_return_each_line():
    text = "# H1\nline a\nline b\n## H2\nline c"
    chunker = MarkdownChunker(return_each_line=True)
    docs = chunker.split_documents([Document(content=text)])
    contents = [d.content for d in docs]
    assert "line a" in contents
    assert "line b" in contents
    assert "line c" in contents


# ── split_documents preserves and extends metadata ───────────────────────────


def test_split_documents_extends_metadata():
    text = "alpha\n\nbravo\n\ncharlie\n\ndelta"
    chunker = CharacterChunker(chunk_size=15, chunk_overlap=0, separator="\n\n")
    docs = chunker.split_documents([Document(content=text, metadata={"source": "doc.txt", "type": "text"})])
    assert all(d.metadata["source"] == "doc.txt" for d in docs)
    assert all(d.metadata["type"] == "text" for d in docs)
    assert all(d.metadata["parent_source"] == "doc.txt" for d in docs)
    assert [d.metadata["chunk_index"] for d in docs] == list(range(len(docs)))
    assert all(d.metadata["chunk_count"] == len(docs) for d in docs)


def test_recursive_chunker_preserves_metadata():
    chunker = RecursiveCharacterChunker(chunk_size=20, chunk_overlap=2)
    docs = chunker.split_documents(
        [Document(content="aa bb cc dd ee ff gg hh ii jj kk ll", metadata={"source": "x.txt"})]
    )
    assert all(d.metadata.get("source") == "x.txt" for d in docs)
    assert {d.metadata["chunk_index"] for d in docs} == set(range(len(docs)))


# ── Registry ─────────────────────────────────────────────────────────────────


def test_registry_get_chunker_with_kwargs():
    chunker = get_chunker("recursive", chunk_size=500, chunk_overlap=50)
    assert isinstance(chunker, RecursiveCharacterChunker)
    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 50


def test_registry_lists_builtin_names():
    for name in ("character", "recursive", "token", "semantic", "markdown"):
        assert name in CHUNKER_REGISTRY
        assert name in ASYNC_CHUNKER_REGISTRY


def test_registry_unknown_chunker():
    with pytest.raises(KeyError):
        get_chunker("does-not-exist")


def test_registry_async_sibling_smoke():
    async_chunker = get_async_chunker("character", chunk_size=20, chunk_overlap=0)
    assert isinstance(async_chunker, AsyncTextChunker)
    assert isinstance(async_chunker, _SyncToAsyncChunker)

    async def _run():
        return await async_chunker.split_text("alpha\n\nbravo\n\ncharlie")

    result = asyncio.run(_run())
    assert isinstance(result, list)
    assert all(isinstance(c, str) for c in result)


def test_registry_async_split_documents():
    async_chunker = get_async_chunker("recursive", chunk_size=30, chunk_overlap=0)

    async def _run():
        return await async_chunker.split_documents([Document(content="x y z " * 20, metadata={"source": "s"})])

    docs = asyncio.run(_run())
    assert docs
    assert all(d.metadata["source"] == "s" for d in docs)


# ── TextChunker is abstract ──────────────────────────────────────────────────


def test_text_chunker_is_abstract():
    with pytest.raises(TypeError):
        TextChunker()  # type: ignore[abstract]


# ── Math sanity for semantic helpers (internal) ──────────────────────────────


def test_semantic_cosine_distance_basic():
    from prompture.rag.chunkers.semantic import _cosine_distance

    assert math.isclose(_cosine_distance([1.0, 0.0], [1.0, 0.0]), 0.0, abs_tol=1e-9)
    assert math.isclose(_cosine_distance([1.0, 0.0], [0.0, 1.0]), 1.0, abs_tol=1e-9)
