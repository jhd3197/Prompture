"""Internal fusion helpers shared by the query-side retriever wrappers.

These keep :mod:`multi_query` and :mod:`step_back` from re-implementing
the RRF + per-doc max-score bookkeeping. Not part of the public API.
"""

from __future__ import annotations

import hashlib

from ..vectorstores.base import VectorSearchResult
from .hybrid import reciprocal_rank_fusion


def _document_id(result: VectorSearchResult) -> str:
    """Return a stable identifier for a search result.

    Prefer ``document.metadata['id']`` when the store provided one;
    otherwise hash the content. This lets us fuse rankings across
    multiple retrieval calls even when the backend doesn't surface IDs.
    """
    meta_id = result.document.metadata.get("id")
    if meta_id is not None:
        return str(meta_id)
    return "sha1:" + hashlib.sha1(result.document.content.encode("utf-8")).hexdigest()


def fuse_search_results(
    result_lists: list[list[VectorSearchResult]],
    *,
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[VectorSearchResult]:
    """Reciprocal-Rank-Fuse multiple ranked ``VectorSearchResult`` lists.

    Args:
        result_lists: One ranked list per upstream retrieval call. Empty
            lists are tolerated.
        rrf_k: RRF damping constant (default 60, per the original paper).
        top_n: Cap the output at the top ``top_n`` fused results. ``None``
            returns the full fused ranking.

    Returns:
        A single ranked list of :class:`VectorSearchResult` objects. Each
        document appears once with whichever copy carried the highest
        upstream score (so downstream ``score`` reflects the best
        evidence seen, not the fused RRF score).
    """
    id_lists: list[list[str]] = []
    best_result: dict[str, VectorSearchResult] = {}

    for results in result_lists:
        ids: list[str] = []
        for r in results:
            rid = _document_id(r)
            ids.append(rid)
            current = best_result.get(rid)
            if current is None or r.score > current.score:
                best_result[rid] = r
        id_lists.append(ids)

    if not best_result:
        return []

    fused = reciprocal_rank_fusion(id_lists, rrf_k=rrf_k)
    ranked_ids = sorted(fused.keys(), key=lambda i: (-fused[i], i))
    out = [best_result[i] for i in ranked_ids if i in best_result]
    if top_n is not None:
        out = out[:top_n]
    return out
