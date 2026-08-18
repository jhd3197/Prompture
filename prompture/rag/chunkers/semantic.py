"""Semantic chunker: group sentences by embedding similarity.

Splits text into sentences, embeds them via an
:class:`~prompture.drivers.embedding_base.EmbeddingDriver`, and inserts
chunk boundaries where the cosine distance between adjacent sentence
embeddings exceeds a configurable threshold.

This is the only chunker in the package that hits an external API at
split time.  ``numpy`` is preferred for percentile / stddev / gradient
math but the implementation falls back to pure Python if numpy is
unavailable.
"""

from __future__ import annotations

import math
import re
from typing import Any

from ..documents import Document
from ..vectorstores.base import embed_texts
from .base import TextChunker

# Best-effort numpy import; the chunker also has pure-python fallbacks.
try:  # pragma: no cover - exercised only when numpy is installed
    import numpy as _np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - exercised only when numpy is missing
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker(TextChunker):
    """Group adjacent sentences with similar embeddings into one chunk.

    Unlike the other chunkers, ``SemanticChunker`` is **driven by
    similarity, not character/token windows** — ``chunk_size`` and
    ``chunk_overlap`` are intentionally ignored.  It needs an
    ``embedding_driver`` instance (any object exposing a synchronous
    ``embed(list[str]) -> list[list[float]]`` method).

    Args:
        embedding_driver: Driver used to embed candidate sentences.
        breakpoint_threshold_type: One of ``"percentile"``,
            ``"standard_deviation"``, ``"interquartile"``, ``"gradient"``.
        breakpoint_threshold_amount: Threshold magnitude (interpreted
            according to ``breakpoint_threshold_type``).
        buffer_size: Number of surrounding sentences combined before
            embedding.  Larger values yield smoother similarity curves.
        min_chunk_size: Drop chunks shorter than this many characters.
    """

    _VALID_TYPES = ("percentile", "standard_deviation", "interquartile", "gradient")

    def __init__(
        self,
        embedding_driver: Any,
        *,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        buffer_size: int = 1,
        min_chunk_size: int = 100,
        **kwargs: Any,
    ):
        # Bypass chunk_size/chunk_overlap validation — this chunker is
        # driven by similarity, not fixed windows.  We still expose
        # placeholders so consumer code reading the attributes doesn't crash.
        self.chunk_size = 0
        self.chunk_overlap = 0
        if breakpoint_threshold_type not in self._VALID_TYPES:
            raise ValueError(
                f"breakpoint_threshold_type must be one of {self._VALID_TYPES}, got {breakpoint_threshold_type!r}"
            )
        self.embedding_driver = embedding_driver
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = float(breakpoint_threshold_amount)
        self.buffer_size = int(buffer_size)
        self.min_chunk_size = int(min_chunk_size)
        # Stash any extra kwargs but don't forward to the base init.
        self._extra: dict[str, Any] = dict(kwargs)

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
        if len(sentences) <= 1:
            return sentences

        combined = self._combine(sentences)
        embeddings = embed_texts(self.embedding_driver, combined)
        if len(embeddings) != len(combined):
            raise RuntimeError(
                f"Embedding driver returned {len(embeddings)} vectors for "
                f"{len(combined)} inputs; expected equal counts."
            )

        distances = [_cosine_distance(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]
        if not distances:
            return [" ".join(sentences)]

        breakpoints = self._find_breakpoints(distances)
        chunks: list[str] = []
        start = 0
        for bp in breakpoints:
            chunk = " ".join(sentences[start : bp + 1]).strip()
            if chunk:
                chunks.append(chunk)
            start = bp + 1
        tail = " ".join(sentences[start:]).strip()
        if tail:
            chunks.append(tail)

        if self.min_chunk_size > 0:
            chunks = [c for c in chunks if len(c) >= self.min_chunk_size] or chunks[:1]
        return chunks

    # ------------------------------------------------------------------
    def _combine(self, sentences: list[str]) -> list[str]:
        b = self.buffer_size
        result: list[str] = []
        for i in range(len(sentences)):
            lo = max(0, i - b)
            hi = min(len(sentences), i + b + 1)
            result.append(" ".join(sentences[lo:hi]))
        return result

    def _find_breakpoints(self, distances: list[float]) -> list[int]:
        kind = self.breakpoint_threshold_type
        amt = self.breakpoint_threshold_amount

        if kind == "percentile":
            threshold = _percentile(distances, amt)
            return [i for i, d in enumerate(distances) if d > threshold]

        if kind == "standard_deviation":
            mean = sum(distances) / len(distances)
            var = sum((d - mean) ** 2 for d in distances) / len(distances)
            std = math.sqrt(var)
            threshold = mean + amt * std
            return [i for i, d in enumerate(distances) if d > threshold]

        if kind == "interquartile":
            q1 = _percentile(distances, 25.0)
            q3 = _percentile(distances, 75.0)
            iqr = q3 - q1
            threshold = q3 + amt * iqr
            return [i for i, d in enumerate(distances) if d > threshold]

        # gradient
        if _HAS_NUMPY:
            grads = list(_np.gradient(_np.array(distances)))  # type: ignore[union-attr]
        else:
            grads = _simple_gradient(distances)
        # Treat the gradient series itself as a distance series and pick the
        # top `amt` percentile of gradient magnitudes as breakpoints.
        mags = [abs(g) for g in grads]
        threshold = _percentile(mags, amt)
        return [i for i, g in enumerate(mags) if g > threshold]

    # ------------------------------------------------------------------
    def split_documents(self, documents: list[Document]) -> list[Document]:
        # Identical semantics to TextChunker.split_documents, but we
        # repeat it here because we bypass the base __init__ validation.
        result: list[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            count = len(chunks)
            parent_source = doc.metadata.get("source")
            for i, chunk in enumerate(chunks):
                new_meta = dict(doc.metadata)
                new_meta["chunk_index"] = i
                new_meta["chunk_count"] = count
                if parent_source is not None and "parent_source" not in new_meta:
                    new_meta["parent_source"] = parent_source
                result.append(Document(content=chunk, metadata=new_meta))
        return result


# ── Math helpers ─────────────────────────────────────────────────────────────


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding vectors must have the same dimensionality")
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    similarity = dot / (na * nb)
    # Clamp to [-1, 1] before computing distance to avoid float drift.
    similarity = max(-1.0, min(1.0, similarity))
    return 1.0 - similarity


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if _HAS_NUMPY:
        return float(_np.percentile(_np.array(values), pct))  # type: ignore[union-attr]
    # Pure-python linear-interpolated percentile (matches numpy default).
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def _simple_gradient(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    grads = [0.0] * n
    grads[0] = values[1] - values[0]
    grads[-1] = values[-1] - values[-2]
    for i in range(1, n - 1):
        grads[i] = (values[i + 1] - values[i - 1]) / 2.0
    return grads
