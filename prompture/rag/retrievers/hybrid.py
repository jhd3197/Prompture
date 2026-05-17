"""Hybrid retriever — dense (vector) + sparse (BM25) fused via RRF."""

from __future__ import annotations

from typing import Any

from ..documents import Document
from ..vectorstores.base import VectorSearchResult, VectorStore
from .base import Retriever


def _tokenize(text: str) -> list[str]:
    """Cheap default tokenizer.

    Lowercased whitespace split.  No stemming, no stopword removal —
    follow-up phases can swap in a richer tokenizer if needed.
    """
    return text.lower().split()


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    rrf_k: int = 60,
) -> dict[str, float]:
    """Combine multiple ranked lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is an ordered sequence of document
            identifiers (highest-ranked first).
        rrf_k: The RRF constant (default ``60`` per the original paper).

    Returns:
        A mapping ``{doc_id: fused_score}`` where the score is the sum
        of ``1 / (rrf_k + rank_i)`` across all input lists.  Documents
        present in only some lists still receive a score from those they
        appear in.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


class HybridRetriever(Retriever):
    """Fuses dense (vector similarity) and sparse (BM25) retrieval.

    Combines two ranked lists via Reciprocal Rank Fusion (RRF).  Either
    pass a pre-built ``rank_bm25.BM25Okapi`` instance through
    ``bm25_index`` along with the ``corpus`` it was built on, or pass
    only ``corpus`` and the retriever builds one internally.

    Args:
        vector_store: The dense backend used for vector similarity.
        corpus: The list of :class:`Document` objects to BM25-index.
            Required unless ``bm25_index`` AND ``bm25_corpus`` are passed.
        bm25_index: Optional pre-built ``BM25Okapi`` instance.
        bm25_corpus: Optional list of :class:`Document` objects aligned
            with a pre-built ``bm25_index``.  Required when supplying
            ``bm25_index``.
        k: Default top-K to return after fusion.
        alpha: Weight of dense vs sparse contribution (``0.0`` = pure
            sparse, ``1.0`` = pure dense).  Applied as a multiplier on
            the per-list RRF contribution.
        rrf_k: The RRF constant.
        filter: Default metadata filter, forwarded to the vector store.

    Requires ``prompture[rag-hybrid]`` (the ``rank-bm25`` package).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        corpus: list[Document] | None = None,
        bm25_index: Any = None,
        bm25_corpus: list[Document] | None = None,
        k: int = 4,
        alpha: float = 0.5,
        rrf_k: int = 60,
        filter: dict | None = None,
    ) -> None:
        if bm25_index is not None:
            if bm25_corpus is None:
                raise ValueError("When passing a pre-built bm25_index you must also pass bm25_corpus.")
            self._bm25 = bm25_index
            self._corpus = list(bm25_corpus)
        else:
            if corpus is None:
                raise ValueError("HybridRetriever requires either `corpus` or `bm25_index` + `bm25_corpus`.")
            try:
                from rank_bm25 import BM25Okapi  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "HybridRetriever requires the `rank-bm25` package. "
                    "Install with: pip install 'prompture[rag-hybrid]'"
                ) from e
            self._corpus = list(corpus)
            tokenized = [_tokenize(doc.content) for doc in self._corpus]
            # rank_bm25 crashes on an empty corpus; guard against it.
            self._bm25 = BM25Okapi(tokenized) if tokenized else None

        self.vector_store = vector_store
        self.k = k
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.filter = filter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_key(doc: Document, idx_hint: int | None = None) -> str:
        """Stable identifier for a document used in fusion.

        Prefers ``metadata["id"]`` then ``metadata["source"] + page`` then
        a positional fallback.  The fallback is acceptable because
        identifiers only need to be unique *within a single call*.
        """
        md = doc.metadata or {}
        if "id" in md:
            return f"id:{md['id']}"
        if "source" in md:
            return f"src:{md.get('source')}|p:{md.get('page', '')}|h:{hash(doc.content)}"
        return f"hash:{hash(doc.content)}|pos:{idx_hint}"

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Return up to ``top_k`` (Document, score) ranked by BM25."""
        if self._bm25 is None or not self._corpus:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [(self._corpus[i], float(s)) for i, s in ranked if s > 0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        effective_k = k if k is not None else self.k
        effective_filter = filter if filter is not None else self.filter
        fetch_k = max(effective_k * 3, effective_k)

        # 1. Dense candidates
        dense = self.vector_store.similarity_search(query, k=fetch_k, filter=effective_filter)
        # 2. Sparse candidates
        sparse = self._bm25_search(query, top_k=fetch_k)

        # 3. Build id → result map (dense results win on metadata + score
        # surface; sparse-only matches are wrapped fresh).
        result_by_id: dict[str, VectorSearchResult] = {}
        dense_ids: list[str] = []
        for i, r in enumerate(dense):
            key = self._doc_key(r.document, idx_hint=i)
            dense_ids.append(key)
            result_by_id.setdefault(key, r)

        sparse_ids: list[str] = []
        for i, (doc, score) in enumerate(sparse):
            key = self._doc_key(doc, idx_hint=10_000 + i)
            sparse_ids.append(key)
            if key not in result_by_id:
                result_by_id[key] = VectorSearchResult(document=doc, score=score, vector=None)

        # 4. RRF fusion with alpha weighting.
        dense_contrib = reciprocal_rank_fusion([dense_ids], rrf_k=self.rrf_k)
        sparse_contrib = reciprocal_rank_fusion([sparse_ids], rrf_k=self.rrf_k)
        fused: dict[str, float] = {}
        for key, s in dense_contrib.items():
            fused[key] = fused.get(key, 0.0) + self.alpha * s
        for key, s in sparse_contrib.items():
            fused[key] = fused.get(key, 0.0) + (1.0 - self.alpha) * s

        # 5. Pick top-k.
        ranked_keys = sorted(fused, key=lambda x: fused[x], reverse=True)[:effective_k]
        out: list[VectorSearchResult] = []
        for key in ranked_keys:
            r = result_by_id[key]
            # Surface the fused score so downstream consumers can sort/filter.
            out.append(VectorSearchResult(document=r.document, score=fused[key], vector=r.vector))
        return out
