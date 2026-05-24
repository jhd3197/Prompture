"""Semantic few-shot example selection for structured extraction.

Stores (input, expected_output) example pairs and serves the most
similar examples to a new query by cosine similarity in embedding space.
Lifts extraction accuracy by injecting K nearest exemplars into the
prompt automatically — published numbers from few-shot literature show
5–20 percentage-point gains on structured-output benchmarks.

Architecture
~~~~~~~~~~~~

* :class:`FewShotExampleStore` — in-memory store. ``add()`` /
  ``add_many()`` to populate, ``select()`` to retrieve top-K.
* :class:`Example` — the dataclass carried inside the store.
* :func:`format_examples_for_prompt` — turns selected examples into
  the prompt fragment :func:`prompture.extraction.core.extract_with_model`
  injects.

The store accepts any embedder with the standard
``embed(texts, options) -> {"embeddings": list[list[float]]}`` contract
used across :class:`prompture.drivers.embedding_base.EmbeddingDriver`.
When ``embedder=None`` a deterministic char-trigram fallback embedder
is used so the store is usable for prototyping / tests with zero
external dependencies.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from .._internal.json_encoder import PromptureJSONEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder protocol + tiny fallback
# ---------------------------------------------------------------------------


class _Embedder(Protocol):
    """Minimal embedder contract — matches :class:`EmbeddingDriver`."""

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        ...


class _NgramEmbedder:
    """Char-trigram hashed embedder. Zero-dependency fallback for testing.

    Produces a fixed-size vector by hashing every char-trigram of the
    input into a bucket and counting occurrences. Cosine similarity on
    these vectors approximates Jaccard-like content overlap. Good
    enough to demonstrate top-K selection in tests; replace with a
    real embedding model in production.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str], options: dict[str, Any]) -> dict[str, Any]:
        vectors: list[list[float]] = []
        for t in texts:
            padded = f"  {t.lower()}  "
            vec = [0.0] * self.dimensions
            for i in range(len(padded) - 2):
                tri = padded[i : i + 3]
                vec[hash(tri) % self.dimensions] += 1.0
            vectors.append(vec)
        return {"embeddings": vectors, "meta": {"model_name": "ngram-fallback", "dimensions": self.dimensions}}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """A single ``(input, expected_output)`` example with its cached embedding."""

    input_text: str
    expected_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class FewShotExampleStore:
    """In-memory semantic example bank for structured extraction.

    Args:
        embedder: Any embedding driver exposing the standard
            ``embed(texts, options) -> {"embeddings": ...}`` contract.
            When ``None`` (the default), a deterministic char-trigram
            fallback is used — fine for prototyping and tests but you
            want a real embedding model for production accuracy.
        embed_options: Forwarded verbatim on every embed call. Useful
            for picking a specific ``dimensions`` on OpenAI embeddings,
            or routing to a specific model.

    Example::

        from prompture.extraction.few_shot import FewShotExampleStore

        store = FewShotExampleStore()  # uses fallback embedder
        store.add("Maria is 32, a developer.", {"name": "Maria", "age": 32})
        store.add("Bob, 47, accountant.", {"name": "Bob", "age": 47})

        hits = store.select("Alice is 25, designer.", k=2)
        for ex in hits:
            print(ex.input_text, ex.expected_output)
    """

    def __init__(
        self,
        embedder: _Embedder | None = None,
        *,
        embed_options: dict[str, Any] | None = None,
    ) -> None:
        self.embedder = embedder if embedder is not None else _NgramEmbedder()
        self.embed_options = embed_options or {}
        self._examples: list[Example] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        input_text: str,
        expected_output: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Example:
        """Embed and store a single example. Returns the stored :class:`Example`."""
        ex = Example(input_text=input_text, expected_output=expected_output, metadata=metadata or {})
        ex.embedding = self._embed_single(input_text)
        self._examples.append(ex)
        return ex

    def add_many(self, examples: list[tuple[str, Any]]) -> list[Example]:
        """Batch-embed and store multiple examples. Saves embedding calls."""
        if not examples:
            return []
        inputs = [t for t, _ in examples]
        vectors = self._embed_batch(inputs)
        out: list[Example] = []
        for (input_text, expected_output), vec in zip(examples, vectors, strict=False):
            ex = Example(input_text=input_text, expected_output=expected_output)
            ex.embedding = vec
            self._examples.append(ex)
            out.append(ex)
        return out

    def clear(self) -> None:
        self._examples.clear()

    def __len__(self) -> int:
        return len(self._examples)

    @property
    def examples(self) -> list[Example]:
        """Read-only view over the stored examples (returned as a copy)."""
        return list(self._examples)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def select(self, query: str, k: int = 3) -> list[Example]:
        """Return the top-``k`` examples most similar to ``query``.

        Ties are broken by insertion order (stable sort). When ``k``
        exceeds the store size, all stored examples are returned.
        """
        if k <= 0 or not self._examples:
            return []
        q_vec = self._embed_single(query)
        scored: list[tuple[float, int, Example]] = []
        for idx, ex in enumerate(self._examples):
            if ex.embedding is None:
                continue
            scored.append((_cosine(q_vec, ex.embedding), idx, ex))
        # Sort by descending similarity, breaking ties on insertion order.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [ex for _, _, ex in scored[:k]]

    # ------------------------------------------------------------------
    # Embedder plumbing
    # ------------------------------------------------------------------

    def _embed_single(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.embedder.embed(texts, self.embed_options)
        if not isinstance(response, dict) or "embeddings" not in response:
            raise ValueError(
                "Embedder returned an unexpected shape; expected a dict with an "
                f"'embeddings' key, got: {type(response).__name__}"
            )
        vectors = response["embeddings"]
        if len(vectors) != len(texts):
            raise ValueError(f"Embedder returned {len(vectors)} vectors for {len(texts)} inputs.")
        return [list(v) for v in vectors]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_examples_for_prompt(
    examples: list[Example],
    *,
    header: str = "Here are similar examples you should use as a guide:",
    footer: str = "Now extract from the input below using the same conventions:",
    output_indent: int = 2,
) -> str:
    """Render a list of :class:`Example` objects as a single prompt fragment.

    Output format::

        {header}

        Input: <example 1 input>
        Output:
        {<example 1 output as JSON>}

        Input: <example 2 input>
        Output:
        {<example 2 output as JSON>}

        {footer}

    The output of each example is serialised via
    :class:`prompture._internal.json_encoder.PromptureJSONEncoder` so
    Pydantic models, datetimes, Decimals, UUIDs, etc. all serialise
    deterministically.

    Returns an empty string when ``examples`` is empty so callers can
    safely splice the result into any prompt without adding stray
    whitespace.
    """
    if not examples:
        return ""
    parts: list[str] = [header.strip(), ""]
    for ex in examples:
        rendered_output = _render_output(ex.expected_output, indent=output_indent)
        parts.append(f"Input: {ex.input_text}")
        parts.append(f"Output:\n{rendered_output}")
        parts.append("")  # blank line between examples
    parts.append(footer.strip())
    return "\n".join(parts).strip() + "\n"


def _render_output(value: Any, *, indent: int) -> str:
    """Serialise an example's expected output to a stable text representation."""
    # Pydantic BaseModel? Use model_dump.
    try:
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
    except ImportError:  # pragma: no cover - pydantic is a hard dep
        pass

    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=indent, cls=PromptureJSONEncoder)
    if isinstance(value, str):
        return value
    # Fallback: best-effort stringify via the shared encoder.
    return json.dumps(value, indent=indent, cls=PromptureJSONEncoder)


__all__ = [
    "Example",
    "FewShotExampleStore",
    "format_examples_for_prompt",
]
