"""RAG with Reranker Example

Demonstrates the ``reranker`` parameter on :class:`RAGPipeline` /
:class:`AsyncRAGPipeline`. Pass a ``"provider/model"`` string and Prompture
resolves it to the right rerank driver via
:func:`prompture.drivers.rerank_registry.get_rerank_driver_for_model`.

The reranker runs **after** the base retriever returns top-K candidates and
re-orders them by relevance to the query. This typically adds 5–15 nDCG
points over dense-only retrieval.

Supported reranker providers (set the appropriate API key):

* ``cohere/rerank-v3.5``   — ``COHERE_API_KEY``
* ``voyage/rerank-2.5``    — ``VOYAGE_API_KEY``
* ``jina/jina-reranker-v2-base-multilingual`` — ``JINA_API_KEY``
* ``mxbai/mxbai-rerank-large-v1`` — ``MIXEDBREAD_API_KEY``

This example uses a tiny in-memory vector store so it runs without any
external infrastructure. Only the reranker call hits a real API (and the
example skips that section if the corresponding key isn't set).
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.rag import RAGPipeline
from prompture.rag.documents import Document
from prompture.rag.retrievers import VectorStoreRetriever
from prompture.rag.vectorstores.base import VectorSearchResult, VectorStore

# =============================================================================
# Minimal in-memory vector store — keeps the example self-contained
# =============================================================================


class _CharNGramVectorStore(VectorStore):
    """Trivial in-memory store that scores docs by char-trigram Jaccard overlap.

    Real applications use a proper embedding model + Chroma/FAISS/etc., but
    this stand-in lets the example run with zero external dependencies and
    keeps the focus on the reranker integration.
    """

    def __init__(self) -> None:
        self._docs: list[Document] = []
        self._ids: list[str] = []

    @staticmethod
    def _trigrams(text: str) -> set[str]:
        s = f"  {text.lower()}  "
        return {s[i : i + 3] for i in range(len(s) - 2)}

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        new_ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in documents]
        self._docs.extend(documents)
        self._ids.extend(new_ids)
        return new_ids

    def delete(self, ids: list[str]) -> None:
        kept_docs = []
        kept_ids = []
        for doc, did in zip(self._docs, self._ids, strict=False):
            if did not in ids:
                kept_docs.append(doc)
                kept_ids.append(did)
        self._docs, self._ids = kept_docs, kept_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **kwargs: Any,
    ) -> list[VectorSearchResult]:
        q = self._trigrams(query)
        if not q:
            return []
        scored: list[tuple[float, Document]] = []
        for doc in self._docs:
            d = self._trigrams(doc.content)
            inter = len(q & d)
            union = len(q | d)
            score = inter / union if union else 0.0
            scored.append((score, doc))
        scored.sort(key=lambda t: t[0], reverse=True)
        results = [VectorSearchResult(document=doc, score=s) for s, doc in scored[: k]]
        return results

    # Base class requires these abstracts; trivial passthroughs.
    def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        # No embeddings here; just delegate to add_documents (vectors ignored).
        return self.add_documents(documents, ids=ids)

    def similarity_search_by_vector(self, vector: list[float], k: int = 4, **kwargs: Any) -> list[VectorSearchResult]:
        raise NotImplementedError("char-trigram store has no embeddings")


def _build_sample_store() -> _CharNGramVectorStore:
    """Seed the store with a handful of facts where reranker order matters."""
    store = _CharNGramVectorStore()
    store.add_documents(
        [
            Document(content="Paris is the capital of France and home to the Eiffel Tower."),
            Document(content="Berlin is the capital of Germany; its population is around 3.6 million."),
            Document(content="The French Republic uses the euro and operates on Central European Time."),
            Document(content="Rome is the capital of Italy and was once the center of the Roman Empire."),
            Document(content="Madrid is Spain's capital and largest city."),
            Document(content="Lyon, Marseille, and Toulouse are major French cities (not the capital)."),
        ]
    )
    return store


# =============================================================================
# A fake LLM driver so the generation step doesn't need any API key
# =============================================================================


class _EchoLLM:
    """Minimal LLM stub: just echoes the question back with the retrieved context."""

    model = "fake/echo"
    model_name = "fake/echo"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": f"(echo-llm) {prompt[:200]}{'...' if len(prompt) > 200 else ''}",
            "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
        }


# =============================================================================
# Examples
# =============================================================================


def example_without_reranker() -> None:
    """Baseline: dense retrieval only, no reranker."""
    print("=" * 70)
    print("EXAMPLE 1: Baseline (no reranker)")
    print("=" * 70)

    store = _build_sample_store()
    retriever = VectorStoreRetriever(store, k=4)
    pipeline = RAGPipeline(retriever=retriever, llm=_EchoLLM())

    answer = pipeline.query("What is the capital of France?", k=4)
    print("Top retrieval results in original order:")
    for i, r in enumerate(answer.retrieval_results, 1):
        print(f"  [{i}] score={r.score:.3f}  {r.document.content}")
    print()


def example_with_reranker_string() -> None:
    """Pass a 'provider/model' string and Prompture instantiates the driver."""
    print("=" * 70)
    print("EXAMPLE 2: With Cohere reranker (string form)")
    print("=" * 70)

    if not os.getenv("COHERE_API_KEY"):
        print("Skipped: set COHERE_API_KEY to run this example.")
        print("Once set, this example would call:")
        print('    RAGPipeline(retriever=..., llm=..., reranker="cohere/rerank-v3.5")')
        print()
        return

    store = _build_sample_store()
    retriever = VectorStoreRetriever(store, k=6)

    # NEW IN PHASE 2: pass a string, RAGPipeline resolves it for you.
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=_EchoLLM(),
        reranker="cohere/rerank-v3.5",
        top_n_after_rerank=3,
    )

    answer = pipeline.query("What is the capital of France?", k=6)
    print("After reranker, kept top-3:")
    for i, r in enumerate(answer.retrieval_results, 1):
        print(f"  [{i}] {r.document.content}")
    print()


def example_with_pre_instantiated_driver() -> None:
    """For when you need custom rerank-driver construction."""
    print("=" * 70)
    print("EXAMPLE 3: With a pre-instantiated rerank driver")
    print("=" * 70)

    if not os.getenv("COHERE_API_KEY"):
        print("Skipped: set COHERE_API_KEY to run this example.")
        print()
        return

    from prompture.drivers.rerank_registry import get_rerank_driver_for_model

    # Hand-construct (e.g. if you need a custom API key, base_url, etc.)
    reranker = get_rerank_driver_for_model("cohere/rerank-v3.5")

    store = _build_sample_store()
    retriever = VectorStoreRetriever(store, k=6)
    pipeline = RAGPipeline(
        retriever=retriever,
        llm=_EchoLLM(),
        reranker=reranker,
        top_n_after_rerank=3,
    )

    answer = pipeline.query("Which European capital uses the euro?", k=6)
    print("Reranked results:")
    for i, r in enumerate(answer.retrieval_results, 1):
        print(f"  [{i}] {r.document.content}")
    print()


def example_error_for_bad_string() -> None:
    """Show what happens with a malformed reranker string."""
    print("=" * 70)
    print("EXAMPLE 4: Error path — malformed reranker string")
    print("=" * 70)

    store = _build_sample_store()
    retriever = VectorStoreRetriever(store, k=4)

    try:
        RAGPipeline(
            retriever=retriever,
            llm=_EchoLLM(),
            reranker="cohere",  # missing /model — invalid
        )
    except ValueError as e:
        print(f"Got expected ValueError: {e}")
    print()


def main() -> None:
    example_without_reranker()
    example_with_reranker_string()
    example_with_pre_instantiated_driver()
    example_error_for_bad_string()


if __name__ == "__main__":
    main()
