"""Multi-Query retriever — LLM-generated reformulations fused via RRF.

For a given user query, asks an LLM to produce ``n_queries`` distinct
reformulations, runs the base retriever once per reformulation (the
original is included by default), and fuses the resulting ranked lists
via Reciprocal Rank Fusion (see :func:`reciprocal_rank_fusion`).

Multi-Query helps when:

* The user's wording is non-canonical relative to the corpus (e.g. uses
  jargon the corpus doesn't, or vice-versa).
* Single-query embedding-space retrieval has high variance — averaging
  across multiple paraphrases stabilises the top-k.

Cost trade: 1 extra LLM call (the reformulation step) + N base-retriever
calls. The reformulation call is small (returns N short strings); the
extra retrievals are cheap if the vector store is local.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from ..vectorstores.base import VectorSearchResult
from ._fusion import fuse_search_results
from .base import Retriever

logger = logging.getLogger(__name__)


#: Default reformulation prompt. ``{query}`` and ``{n}`` are substituted.
DEFAULT_MULTI_QUERY_PROMPT = (
    "You generate alternative phrasings of a user's question to help a "
    "search engine find relevant documents.\n\n"
    "Generate exactly {n} alternative versions of the question below. "
    "Vary the wording, synonyms, and level of abstraction — but preserve "
    "the original intent. Each version should be a complete, standalone "
    "question on a single line. Do NOT echo the original question. Do NOT "
    "number the alternatives.\n\n"
    "Question: {query}\n\n"
    "Alternatives:"
)


class _LLM(Protocol):
    """Minimal LLM contract used by the MultiQuery wrapper."""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


_LEADING_NUMBER = re.compile(r"^\s*(?:\d+[.)]|\*|-)\s+")


def _parse_alternatives(raw: str, max_count: int) -> list[str]:
    """Pull individual reformulation lines out of an LLM response.

    Handles common shapes:

    * one per line, no numbering — emitted directly
    * ``1. Foo``, ``1) Foo``, ``- Foo``, ``* Foo`` — leading marker stripped
    * blank lines and surrounding whitespace are dropped
    * de-duplicates (case-insensitive)
    * caps the result at ``max_count`` lines
    """
    seen: set[str] = set()
    out: list[str] = []
    for line in raw.splitlines():
        stripped = _LEADING_NUMBER.sub("", line.strip()).strip(" \"'")
        if not stripped:
            continue
        key = stripped.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(stripped)
        if len(out) >= max_count:
            break
    return out


class MultiQueryRetriever(Retriever):
    """Wraps a base retriever, RRF-fusing results across LLM reformulations.

    Args:
        base: The underlying retriever called once per query variant.
        llm: LLM driver exposing ``generate(prompt, options) -> {"text": ...}``.
        n_queries: How many *additional* reformulations to generate. Total
            retrieval calls = ``n_queries + 1`` when ``include_original`` is
            true. Defaults to 3.
        include_original: Include the user's original query in the fan-out.
            Recommended (cheap safety net). Defaults to True.
        prompt: Template containing ``{query}`` and ``{n}`` placeholders.
            Defaults to :data:`DEFAULT_MULTI_QUERY_PROMPT`.
        llm_options: Options forwarded to the LLM. Higher temperature gives
            more diversity in reformulations.
        fetch_k: Per-variant retrieval size. Defaults to ``k`` from the
            ``retrieve()`` call when ``None``; oversample (e.g. 2× the final
            ``k``) when you need wider coverage before fusion.
        rrf_k: Reciprocal Rank Fusion damping constant. Defaults to 60.
    """

    def __init__(
        self,
        base: Retriever,
        llm: _LLM,
        *,
        n_queries: int = 3,
        include_original: bool = True,
        prompt: str = DEFAULT_MULTI_QUERY_PROMPT,
        llm_options: dict[str, Any] | None = None,
        fetch_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        if n_queries < 1:
            raise ValueError(f"n_queries must be >= 1, got {n_queries}")
        if "{query}" not in prompt or "{n}" not in prompt:
            raise ValueError(
                "MultiQuery prompt must include '{query}' and '{n}' placeholders; "
                f"got: {prompt[:60]!r}..."
            )
        self.base = base
        self.llm = llm
        self.n_queries = n_queries
        self.include_original = include_original
        self.prompt = prompt
        self.llm_options = llm_options or {"temperature": 0.7, "max_tokens": 512}
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def _generate_alternatives(self, query: str) -> list[str]:
        rendered = self.prompt.format(query=query, n=self.n_queries)
        try:
            response = self.llm.generate(rendered, self.llm_options)
        except Exception:
            logger.warning(
                "MultiQueryRetriever: LLM call failed; falling back to original-query only.",
                exc_info=True,
            )
            return []
        text = (response.get("text") if isinstance(response, dict) else str(response)) or ""
        alternatives = _parse_alternatives(text, max_count=self.n_queries)
        # Drop any reformulation that turned out identical to the user's query.
        q = query.casefold().strip()
        return [a for a in alternatives if a.casefold().strip() != q]

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        alternatives = self._generate_alternatives(query)
        # Always start with the original (when requested) so a degenerate LLM
        # output can never make this retriever worse than the base.
        variants: list[str] = []
        if self.include_original or not alternatives:
            variants.append(query)
        variants.extend(alternatives)

        per_query_k = self.fetch_k or k
        result_lists: list[list[VectorSearchResult]] = []
        for v in variants:
            try:
                result_lists.append(self.base.retrieve(v, k=per_query_k, filter=filter, **options))
            except Exception:
                logger.warning(
                    "MultiQueryRetriever: base.retrieve failed for variant %r; skipping.",
                    v[:80],
                    exc_info=True,
                )

        return fuse_search_results(result_lists, rrf_k=self.rrf_k, top_n=k)
