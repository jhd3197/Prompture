"""Step-Back retriever — broader-query abstraction fused with the original.

Based on the Step-Back Prompting technique
(`Zheng et al. 2023 <https://arxiv.org/abs/2310.06117>`_): for queries
that are very specific, the relevant evidence often only appears when
the retriever sees the *broader concept* the query is about. Step-Back
asks an LLM to rephrase the query into a more general one, retrieves
both, and RRF-fuses the two ranked lists.

Useful when:

* Questions are narrow / detail-oriented and the corpus contains
  background context the query doesn't lexically overlap with.
* You want to surface "supporting evidence" passages, not just direct
  answer passages, before the generation step.

Cost trade: 1 extra LLM call + 1 extra retrieval call per query.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..vectorstores.base import VectorSearchResult
from ._fusion import fuse_search_results
from .base import Retriever

logger = logging.getLogger(__name__)


#: Default step-back prompt. ``{query}`` is substituted with the user's
#: question. Adapted from the original paper's exemplar instruction
#: (without the few-shot examples).
DEFAULT_STEP_BACK_PROMPT = (
    "You are an expert at rephrasing questions to be more general.\n\n"
    "Given a specific question, write a single broader, more abstract "
    "question that asks about the underlying concept or topic. The broader "
    "question should make it easier to retrieve background context that "
    "could help answer the original. Output only the broader question, on "
    "one line, with no prefix.\n\n"
    "Original: {query}\n\n"
    "Broader:"
)


class _LLM(Protocol):
    """Minimal LLM contract used by the StepBack wrapper."""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


class StepBackRetriever(Retriever):
    """Retriever that fuses results from the original query and a step-back query.

    Args:
        base: The underlying retriever called twice (once per variant).
        llm: LLM driver exposing ``generate(prompt, options) -> {"text": ...}``.
        prompt: Template containing a ``{query}`` placeholder. Defaults to
            :data:`DEFAULT_STEP_BACK_PROMPT`.
        llm_options: Options forwarded to the LLM. Defaults to a low-temperature
            generation since we want a single deterministic rephrasing.
        fetch_k: Per-variant retrieval size. Defaults to ``k`` from the
            ``retrieve()`` call when ``None``.
        rrf_k: Reciprocal Rank Fusion damping constant. Defaults to 60.
    """

    def __init__(
        self,
        base: Retriever,
        llm: _LLM,
        *,
        prompt: str = DEFAULT_STEP_BACK_PROMPT,
        llm_options: dict[str, Any] | None = None,
        fetch_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        if "{query}" not in prompt:
            raise ValueError(
                "StepBack prompt must include a '{query}' placeholder; got: "
                f"{prompt[:60]!r}..."
            )
        self.base = base
        self.llm = llm
        self.prompt = prompt
        self.llm_options = llm_options or {"temperature": 0.0, "max_tokens": 128}
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def _draft_step_back(self, query: str) -> str | None:
        """Ask the LLM for a broader version of ``query``.

        Returns ``None`` when the LLM produces nothing useful or when its
        output is identical to the original (case-insensitive). In either
        case the wrapper falls back to original-only retrieval.
        """
        rendered = self.prompt.format(query=query)
        try:
            response = self.llm.generate(rendered, self.llm_options)
        except Exception:
            logger.warning(
                "StepBackRetriever: LLM call failed; falling back to original-only retrieval.",
                exc_info=True,
            )
            return None
        text = (response.get("text") if isinstance(response, dict) else str(response)) or ""
        # Some models emit a label-y prefix ("Broader question:") or quote
        # the answer. Strip both before comparing to the original.
        cleaned = text.strip().strip(" \"'")
        # If the model returned multiple lines, keep the first non-empty one.
        for line in cleaned.splitlines():
            line = line.strip(" \"'")
            if line:
                cleaned = line
                break
        if not cleaned or cleaned.casefold() == query.casefold():
            return None
        return cleaned

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        broader = self._draft_step_back(query)
        per_query_k = self.fetch_k or k
        # Original retrieval is always done. Broader is added only when the
        # LLM produced something distinct.
        result_lists: list[list[VectorSearchResult]] = [
            self.base.retrieve(query, k=per_query_k, filter=filter, **options)
        ]
        if broader:
            try:
                result_lists.append(self.base.retrieve(broader, k=per_query_k, filter=filter, **options))
            except Exception:
                logger.warning(
                    "StepBackRetriever: base.retrieve failed for step-back variant %r; ignoring.",
                    broader[:80],
                    exc_info=True,
                )
        return fuse_search_results(result_lists, rrf_k=self.rrf_k, top_n=k)
