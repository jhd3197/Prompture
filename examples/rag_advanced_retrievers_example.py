"""RAG Advanced Retrievers Example

Demonstrates the three query-side retriever wrappers introduced in Phase 3:

* :class:`HyDERetriever` — LLM drafts a hypothetical answer; the base
  retriever embeds + retrieves on that.
* :class:`MultiQueryRetriever` — LLM generates N reformulations; results
  from each are fused via Reciprocal Rank Fusion.
* :class:`StepBackRetriever` — LLM produces a broader / more abstract
  query; original and broader rankings are fused.

The wrappers compose naturally — each one decorates any base retriever
(including another wrapper) — so you can stack ``HyDE(MultiQuery(...))``
or any other combination.

Run it
~~~~~~

Set ``OLLAMA_ENDPOINT`` (default ``http://localhost:11434``) and pull a
small instruction-tuned model like ``llama3.1:8b``. The example uses a
tiny in-memory char-trigram store so no embedding model is required.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.rag.documents import Document
from prompture.rag.retrievers import (
    HyDERetriever,
    MultiQueryRetriever,
    StepBackRetriever,
    VectorStoreRetriever,
)
from prompture.rag.vectorstores.base import VectorSearchResult, VectorStore

# =============================================================================
# Minimal in-memory vector store (char-trigram Jaccard — no embeddings)
# =============================================================================


class _CharNGramVectorStore(VectorStore):
    """Same toy store used in ``rag_with_rerank_example.py``."""

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

    def add_vectors(
        self,
        vectors: list[list[float]],
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        return self.add_documents(documents, ids=ids)

    def delete(self, ids: list[str]) -> None:
        kept_docs, kept_ids = [], []
        for doc, did in zip(self._docs, self._ids):
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
        scored: list[tuple[float, Document, str]] = []
        for doc, did in zip(self._docs, self._ids):
            d = self._trigrams(doc.content)
            inter = len(q & d)
            union = len(q | d)
            score = inter / union if union else 0.0
            scored.append((score, doc, did))
        scored.sort(key=lambda t: t[0], reverse=True)
        # Stamp id into metadata so the fusion helper can dedupe across variants.
        out: list[VectorSearchResult] = []
        for s, doc, did in scored[:k]:
            doc_with_id = Document(content=doc.content, metadata={**doc.metadata, "id": did})
            out.append(VectorSearchResult(document=doc_with_id, score=s))
        return out

    def similarity_search_by_vector(
        self, vector: list[float], k: int = 4, **kwargs: Any
    ) -> list[VectorSearchResult]:
        raise NotImplementedError("char-trigram store has no embeddings")


def _build_sample_store() -> _CharNGramVectorStore:
    store = _CharNGramVectorStore()
    store.add_documents(
        [
            Document(content="Paris is the capital of France and home to the Eiffel Tower."),
            Document(content="Berlin is the capital of Germany; population around 3.6M."),
            Document(content="The Louvre Museum sits along the Seine in central Paris."),
            Document(content="France uses the euro and operates on Central European Time."),
            Document(content="Rome is the capital of Italy and was the heart of the Roman Empire."),
            Document(content="Madrid is Spain's capital and its largest city."),
            Document(content="The European Union has 27 member states using a common market."),
            Document(content="Lyon, Marseille, and Toulouse are major French cities (not the capital)."),
        ]
    )
    return store


# =============================================================================
# LLM choice — tries Ollama if available, otherwise uses a scripted stub
# =============================================================================


class _ScriptedLLM:
    """Fallback for when Ollama isn't reachable. Keeps the demo runnable."""

    _SCRIPT = {
        "HyDE": (
            "Paris is the capital city of France. It is located on the river Seine "
            "and is the country's largest city, with iconic landmarks including the "
            "Eiffel Tower and the Louvre."
        ),
        "MultiQuery": (
            "Which city serves as France's capital?\n"
            "What is the name of the capital of France?\n"
            "What city is the seat of the French government?"
        ),
        "StepBack": "What are the major capital cities of European countries?",
    }

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": self._SCRIPT[self.kind],
            "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0},
        }


def _ollama_or_scripted(kind: str):
    """Return an Ollama driver if reachable, else a scripted fallback."""
    try:
        from prompture.drivers.ollama_driver import OllamaDriver

        driver = OllamaDriver(
            endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        )
        # Probe with a tiny call to confirm reachability.
        driver.generate("ping", {"temperature": 0.0, "max_tokens": 4})
        print(f"  [LLM: ollama/{driver.model}]")
        return driver
    except Exception as e:
        print(f"  [LLM: scripted fallback — Ollama not reachable: {type(e).__name__}]")
        return _ScriptedLLM(kind)


# =============================================================================
# Pretty-print helpers
# =============================================================================


def _print_results(results: list[VectorSearchResult], label: str) -> None:
    print(f"  {label}:")
    if not results:
        print("    (no results)")
        return
    for i, r in enumerate(results, 1):
        snippet = r.document.content[:90].replace("\n", " ")
        print(f"    [{i}] {snippet}")


# =============================================================================
# Examples
# =============================================================================


QUERY = "Where is the Eiffel Tower?"


def example_baseline() -> None:
    print("=" * 70)
    print("EXAMPLE 0: Baseline (similarity retriever, no wrapping)")
    print("=" * 70)
    store = _build_sample_store()
    base = VectorStoreRetriever(store, k=4)
    _print_results(base.retrieve(QUERY, k=4), "baseline")
    print()


def example_hyde() -> None:
    print("=" * 70)
    print(f"EXAMPLE 1: HyDE — LLM drafts a hypothetical answer for {QUERY!r}")
    print("=" * 70)
    store = _build_sample_store()
    base = VectorStoreRetriever(store, k=4)
    llm = _ollama_or_scripted("HyDE")

    ret = HyDERetriever(base=base, llm=llm)
    _print_results(ret.retrieve(QUERY, k=4), "after HyDE")
    print()


def example_multi_query() -> None:
    print("=" * 70)
    print("EXAMPLE 2: MultiQuery — 3 LLM-generated reformulations, RRF-fused")
    print("=" * 70)
    store = _build_sample_store()
    base = VectorStoreRetriever(store, k=4)
    llm = _ollama_or_scripted("MultiQuery")

    ret = MultiQueryRetriever(base=base, llm=llm, n_queries=3, fetch_k=6)
    _print_results(ret.retrieve(QUERY, k=4), "after MultiQuery")
    print()


def example_step_back() -> None:
    print("=" * 70)
    print("EXAMPLE 3: StepBack — LLM produces a broader query, RRF-fused")
    print("=" * 70)
    store = _build_sample_store()
    base = VectorStoreRetriever(store, k=4)
    llm = _ollama_or_scripted("StepBack")

    ret = StepBackRetriever(base=base, llm=llm, fetch_k=6)
    _print_results(ret.retrieve(QUERY, k=4), "after StepBack")
    print()


def example_stacked() -> None:
    """Wrappers compose — wrap a wrapper to combine techniques."""
    print("=" * 70)
    print("EXAMPLE 4: Stacked — HyDE(MultiQuery(base))")
    print("=" * 70)
    store = _build_sample_store()
    base = VectorStoreRetriever(store, k=4)
    llm = _ollama_or_scripted("MultiQuery")

    # MultiQuery runs first (its base IS the similarity retriever); HyDE
    # then wraps the whole thing so the *outer* call swaps the user query
    # for a hypothetical answer before fan-out.
    inner = MultiQueryRetriever(base=base, llm=llm, n_queries=2, fetch_k=4)
    outer = HyDERetriever(base=inner, llm=_ollama_or_scripted("HyDE"))

    _print_results(outer.retrieve(QUERY, k=4), "after HyDE(MultiQuery(...))")
    print()


def main() -> None:
    example_baseline()
    example_hyde()
    example_multi_query()
    example_step_back()
    example_stacked()


if __name__ == "__main__":
    main()
