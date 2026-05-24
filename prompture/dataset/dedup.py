"""Dataset deduplication — exact, shingle, and semantic strategies.

Three modes (cheapest first, swap up as the dataset grows or precision
matters):

* :func:`dedup_exact` — normalised string equality on the question field.
  O(n). Drops only carbon-copy duplicates. The legacy behaviour.
* :func:`dedup_shingle` — Jaccard similarity on character k-shingles.
  O(n²) with an early-exit threshold; fast enough for ~50k items.
  Catches paraphrased duplicates that exact-matching misses
  ("What is the capital of France?" vs "What is France's capital?").
* :func:`dedup_semantic` — embedding-cosine threshold using any embedder
  that matches the :func:`prompture.extraction.few_shot.FewShotExampleStore`
  contract. Catches deep paraphrases that share little surface form.

All three preserve insertion order — the first occurrence wins.
``dedup_*`` functions return ``(kept_pairs, removed_count)`` so
pipelines can log a drop rate.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared key extraction
# ---------------------------------------------------------------------------


def _default_key(pair: Any) -> str:
    """Return the canonical 'identity' text for a pair.

    Uses the question / instruction field — that's typically what
    duplicates collide on. Override via the ``key`` argument when
    you want answer-side dedup or a custom key.
    """
    if hasattr(pair, "question"):
        return str(pair.question or "")
    if hasattr(pair, "instruction"):
        return str(pair.instruction or "")
    if isinstance(pair, dict):
        return str(pair.get("question") or pair.get("instruction") or "")
    return str(pair)


_NORMALISE_PUNCT = re.compile(r"[^\w\s]")


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Used by exact and shingle dedup so trivial cosmetic variants
    bucket together.
    """
    text = text.casefold()
    text = _NORMALISE_PUNCT.sub(" ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# 1. Exact dedup
# ---------------------------------------------------------------------------


def dedup_exact(
    pairs: list[Any],
    *,
    key: Callable[[Any], str] = _default_key,
) -> tuple[list[Any], int]:
    """Drop pairs whose normalised key has been seen.

    Same algorithm as the legacy ``_dedupe`` in ``synth.py`` but
    promoted to a public function so it composes with the other
    strategies.
    """
    seen: set[str] = set()
    kept: list[Any] = []
    removed = 0
    for pair in pairs:
        k = _normalise(key(pair))
        if not k:
            continue
        if k in seen:
            removed += 1
            continue
        seen.add(k)
        kept.append(pair)
    return kept, removed


# ---------------------------------------------------------------------------
# 2. Shingle / Jaccard near-dup
# ---------------------------------------------------------------------------


def _shingles(text: str, k: int) -> set[str]:
    """Return the set of overlapping character k-shingles for ``text``.

    Pads with two leading/trailing spaces so the first/last few
    characters get full coverage.
    """
    if k <= 0:
        return set()
    padded = f"  {text}  "
    if len(padded) < k:
        return {padded}
    return {padded[i : i + k] for i in range(len(padded) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def dedup_shingle(
    pairs: list[Any],
    *,
    key: Callable[[Any], str] = _default_key,
    threshold: float = 0.8,
    shingle_size: int = 5,
) -> tuple[list[Any], int]:
    """Drop pairs whose key shingle-Jaccard exceeds ``threshold`` vs. any prior kept pair.

    Args:
        pairs: Items to deduplicate.
        key: Extracts the comparison string from each item.
        threshold: Jaccard similarity above which two items are
            considered duplicates. 0.8 catches paraphrases with strong
            shared phrasing; 0.6 catches looser similarity at the cost
            of some false positives.
        shingle_size: Character window size for k-shingles. Larger
            values are more conservative (require longer shared
            substrings); 5 is a reasonable default.

    Returns:
        ``(kept, removed_count)``. ``kept`` preserves the original
        order — the first occurrence of any near-duplicate wins.
    """
    kept_shingles: list[set[str]] = []
    kept: list[Any] = []
    removed = 0
    for pair in pairs:
        norm_key = _normalise(key(pair))
        if not norm_key:
            continue
        shings = _shingles(norm_key, shingle_size)
        is_dup = False
        for prior in kept_shingles:
            if _jaccard(shings, prior) >= threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(pair)
            kept_shingles.append(shings)
    return kept, removed


# ---------------------------------------------------------------------------
# 3. Semantic / embedding dedup
# ---------------------------------------------------------------------------


class _Embedder(Protocol):
    """Minimal embedder contract — matches :class:`EmbeddingDriver`."""

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        ...


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def dedup_semantic(
    pairs: list[Any],
    *,
    embedder: _Embedder,
    key: Callable[[Any], str] = _default_key,
    threshold: float = 0.92,
    embed_options: dict[str, Any] | None = None,
) -> tuple[list[Any], int]:
    """Drop pairs whose key embeddings are within cosine ``threshold`` of any prior kept pair.

    Uses one batched call to ``embedder.embed`` for the full set, then
    runs O(n²) cosine comparisons with an early exit per pair. For
    very large datasets (>50k pairs), swap in a vector store + ANN
    index instead.

    Args:
        pairs: Items to deduplicate.
        embedder: Anything with ``embed(texts, options) -> {"embeddings": ...}``.
            ``FewShotExampleStore``'s default ``_NgramEmbedder`` works
            for prototyping; use a real embedding driver in production.
        key: Extracts the comparison string from each item.
        threshold: Cosine similarity above which two items are
            duplicates. 0.92 is conservative (catches deep paraphrases
            without merging genuinely-different-but-related questions);
            lower for more aggressive dedup.
        embed_options: Forwarded to ``embedder.embed``.

    Returns:
        ``(kept, removed_count)``. Order preserved.

    Raises:
        ValueError: when ``embedder`` returns the wrong number of
            vectors or an unexpected shape.
    """
    if not pairs:
        return [], 0
    texts = [key(p) for p in pairs]
    # Drop pairs with an empty key first — they'd otherwise force a
    # zero-vector compare that always reports cosine 0.
    indexed: list[tuple[int, str]] = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not indexed:
        return [], len(pairs)

    response = embedder.embed([t for _, t in indexed], embed_options or {})
    if not isinstance(response, dict) or "embeddings" not in response:
        raise ValueError(
            f"Embedder returned an unexpected shape; expected {{'embeddings': [...]}}, got {type(response).__name__}."
        )
    vectors = response["embeddings"]
    if len(vectors) != len(indexed):
        raise ValueError(f"Embedder returned {len(vectors)} vectors for {len(indexed)} inputs.")

    embed_by_idx: dict[int, list[float]] = {idx: list(vec) for (idx, _), vec in zip(indexed, vectors, strict=False)}

    kept: list[Any] = []
    kept_vecs: list[list[float]] = []
    removed = 0
    for i, pair in enumerate(pairs):
        vec = embed_by_idx.get(i)
        if vec is None:
            # Empty key — drop quietly (also matches dedup_exact behaviour).
            removed += 1
            continue
        is_dup = False
        for prior in kept_vecs:
            if _cosine(vec, prior) >= threshold:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(pair)
            kept_vecs.append(vec)
    return kept, removed


# ---------------------------------------------------------------------------
# Unified config
# ---------------------------------------------------------------------------


@dataclass
class DedupConfig:
    """Pipeline-friendly dedup configuration.

    Pass an instance to :func:`apply_dedup` (or to the synth pipeline's
    ``dedup`` parameter) instead of remembering which function to call.

    Attributes:
        strategy: ``"exact"`` | ``"shingle"`` | ``"semantic"`` | ``"none"``.
        threshold: Similarity threshold (ignored for ``exact`` / ``none``).
        shingle_size: Only used when ``strategy == "shingle"``.
        embedder: Required when ``strategy == "semantic"``.
        embed_options: Forwarded to the embedder.
        key: Override the default key extractor (question / instruction).
    """

    strategy: str = "exact"
    threshold: float = 0.8
    shingle_size: int = 5
    embedder: Any = None
    embed_options: dict[str, Any] | None = None
    key: Callable[[Any], str] = _default_key


def apply_dedup(pairs: list[Any], config: DedupConfig) -> tuple[list[Any], int]:
    """Run ``config.strategy`` against ``pairs``; return ``(kept, removed_count)``.

    Use this when you've stored a :class:`DedupConfig` and want a
    single entry point. ``strategy="none"`` is a no-op pass-through.
    """
    if config.strategy == "none":
        return list(pairs), 0
    if config.strategy == "exact":
        return dedup_exact(pairs, key=config.key)
    if config.strategy == "shingle":
        return dedup_shingle(
            pairs,
            key=config.key,
            threshold=config.threshold,
            shingle_size=config.shingle_size,
        )
    if config.strategy == "semantic":
        if config.embedder is None:
            raise ValueError("DedupConfig(strategy='semantic') requires an `embedder`.")
        return dedup_semantic(
            pairs,
            embedder=config.embedder,
            key=config.key,
            threshold=config.threshold,
            embed_options=config.embed_options,
        )
    raise ValueError(f"Unknown dedup strategy {config.strategy!r}. Choose 'exact', 'shingle', 'semantic', or 'none'.")


__all__ = [
    "DedupConfig",
    "apply_dedup",
    "dedup_exact",
    "dedup_semantic",
    "dedup_shingle",
]
