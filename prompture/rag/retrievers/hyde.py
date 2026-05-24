"""HyDE (Hypothetical Document Embeddings) retriever wrapper.

HyDE — `Gao et al. 2022 <https://arxiv.org/abs/2212.10496>`_ — observes
that an LLM-generated hypothetical answer to the user query is often a
better embedding-space neighbour of the *real* answer than the question
itself. The wrapper:

1. Asks an LLM to draft a hypothetical answer to ``query``.
2. Passes that hypothetical answer through to a base retriever (which
   embeds and looks up nearest neighbours as usual).
3. Returns the base retriever's results.

Trades one extra LLM call per query for a measurable lift in retrieval
quality. Particularly effective when queries are short / under-specified
and the corpus contains long-form passages.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..vectorstores.base import VectorSearchResult
from .base import Retriever

logger = logging.getLogger(__name__)


#: Default prompt used to draft the hypothetical answer. ``{query}`` is
#: substituted with the user's question. Override via the ``prompt``
#: constructor argument.
DEFAULT_HYDE_PROMPT = (
    "Write a concise, factual answer to the question below. Include enough "
    "specific detail that the answer could plausibly appear verbatim in a "
    "reference document. Do not say you are unsure — write the most likely "
    "answer.\n\n"
    "Question: {query}\n\n"
    "Answer:"
)


class _LLM(Protocol):
    """Minimal LLM contract used by the HyDE wrapper."""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


class HyDERetriever(Retriever):
    """Retriever that swaps the user query for an LLM-drafted hypothetical answer.

    Args:
        base: The underlying retriever (typically a
            :class:`VectorStoreRetriever`) that does the actual lookup.
        llm: Any LLM driver exposing ``generate(prompt, options) -> {"text": ...}``.
        prompt: Template containing a ``{query}`` placeholder. Defaults to
            :data:`DEFAULT_HYDE_PROMPT`.
        llm_options: Options forwarded to the LLM (e.g. temperature). Defaults
            to a low-temperature deterministic generation.
        max_hypothetical_chars: Truncate the LLM output to this many characters
            before forwarding to the base retriever. Long hypotheticals waste
            embedding context. Defaults to 2000.
    """

    def __init__(
        self,
        base: Retriever,
        llm: _LLM,
        *,
        prompt: str = DEFAULT_HYDE_PROMPT,
        llm_options: dict[str, Any] | None = None,
        max_hypothetical_chars: int = 2000,
    ) -> None:
        if "{query}" not in prompt:
            raise ValueError(
                "HyDE prompt must include a '{query}' placeholder; got: "
                f"{prompt[:60]!r}..."
            )
        self.base = base
        self.llm = llm
        self.prompt = prompt
        self.llm_options = llm_options or {"temperature": 0.0, "max_tokens": 256}
        self.max_hypothetical_chars = max_hypothetical_chars

    def _draft_hypothetical(self, query: str) -> str:
        """Call the LLM to produce a hypothetical answer for ``query``.

        Falls back to the original query when the LLM returns nothing
        useful so HyDE can never make a query *worse* than the baseline.
        """
        rendered = self.prompt.format(query=query)
        try:
            response = self.llm.generate(rendered, self.llm_options)
        except Exception:
            logger.warning(
                "HyDERetriever: LLM call failed; falling back to the original query.",
                exc_info=True,
            )
            return query
        text = (response.get("text") if isinstance(response, dict) else str(response)) or ""
        text = text.strip()
        if not text:
            logger.debug("HyDERetriever: LLM returned empty text; using original query.")
            return query
        if len(text) > self.max_hypothetical_chars:
            text = text[: self.max_hypothetical_chars]
        return text

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
        **options: Any,
    ) -> list[VectorSearchResult]:
        hypothetical = self._draft_hypothetical(query)
        # The base retriever handles embedding + search. Reuse its k/filter
        # contract verbatim — HyDE only swaps the *query text*.
        return self.base.retrieve(hypothetical, k=k, filter=filter, **options)
