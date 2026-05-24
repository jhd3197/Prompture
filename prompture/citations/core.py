"""Citation tracker and parser.

The tracker is intentionally retriever-agnostic: pass any iterable of
:class:`~prompture.rag.Document`, :class:`~prompture.rag.VectorSearchResult`,
or already-built :class:`Source` records.  It assigns stable ids,
formats a citation-instructed prompt, and parses the LLM output back
into typed :class:`Citation` records.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .types import Citation, CitedAnswer, Source

if TYPE_CHECKING:
    from ..rag.documents import Document
    from ..rag.vectorstores.base import VectorSearchResult


CITATION_INSTRUCTION = (
    "Answer the question using ONLY the numbered sources below. "
    "Cite every factual claim by appending the source number(s) in square "
    "brackets immediately after the sentence — e.g. 'Paris is the capital "
    "of France [1].' or 'Both routes are viable [2, 5].'. "
    "If the sources do not contain enough information, say so explicitly "
    "and do not cite anything for that sentence."
)

_BRACKET_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _coerce_sources(items: Iterable[Any], *, start: int = 1) -> list[Source]:
    """Coerce mixed inputs into a normalised :class:`Source` list.

    Accepted item types:

    - :class:`Source` — used as-is.
    - :class:`~prompture.rag.documents.Document` — id assigned in order,
      score left ``None``.
    - :class:`~prompture.rag.vectorstores.base.VectorSearchResult` —
      score preserved.
    - Any object exposing ``content`` and a ``metadata`` mapping — duck
      typed; matches the loose ``Document``-shaped objects used in
      tests.

    Sources keep stable string ids starting at *start* (default 1).
    """
    out: list[Source] = []
    next_id = start
    for item in items:
        if isinstance(item, Source):
            out.append(item)
            continue

        text: str | None = None
        title: str | None = None
        url: str | None = None
        score: float | None = None
        metadata: dict[str, Any] = {}

        # VectorSearchResult shape: document + score
        if hasattr(item, "document") and hasattr(item, "score"):
            doc = item.document
            text = getattr(doc, "content", None)
            metadata = dict(getattr(doc, "metadata", {}) or {})
            score = float(item.score) if item.score is not None else None
        else:
            text = getattr(item, "content", None)
            metadata = dict(getattr(item, "metadata", {}) or {})

        if text is None:
            raise TypeError(
                f"Cannot derive source text from {type(item).__name__}; "
                "supply a Source, Document, or VectorSearchResult."
            )

        title = metadata.get("title") or metadata.get("source")
        url = metadata.get("url")

        out.append(
            Source(
                id=str(next_id),
                text=text,
                title=title,
                url=url,
                score=score,
                metadata=metadata,
            )
        )
        next_id += 1
    return out


def _split_source_ids(group: str) -> list[str]:
    """Parse the comma-separated id list inside a ``[...]`` marker."""
    return [s.strip() for s in group.split(",") if s.strip()]


def _claim_before(text: str, marker_start: int) -> tuple[str, int]:
    """Return the sentence (or fragment) ending immediately before *marker_start*.

    Scans backwards from the bracket position to the previous sentence
    boundary (``.``, ``!``, ``?``, or a newline), returning that span
    and its start offset in the original text.
    """
    upto = text[:marker_start]
    boundaries = list(_SENTENCE_END_RE.finditer(upto))
    if boundaries:
        last = boundaries[-1]
        claim_start = last.end()
    else:
        claim_start = 0
    claim = upto[claim_start:].strip()
    return claim, claim_start


def extract_citations(text: str, sources: Iterable[Source]) -> tuple[list[Citation], str, set[str]]:
    """Pull bracketed citation markers out of *text*.

    Args:
        text: Raw LLM output containing ``[n]`` or ``[n,m]`` markers.
        sources: The sources made available to the LLM.  Markers
            referencing ids outside this set are still recorded (they
            indicate a hallucinated reference) but those ids appear in
            the returned :class:`Citation.source_ids` unchanged so
            callers can detect the divergence.

    Returns:
        A tuple ``(citations, clean_text, cited_ids)`` where
        ``clean_text`` is *text* with every ``[...]`` marker stripped
        and ``cited_ids`` is the set of ids that actually appear in at
        least one valid marker.
    """
    sources_by_id = {s.id: s for s in sources}
    citations: list[Citation] = []
    cited: set[str] = set()

    for match in _BRACKET_RE.finditer(text):
        ids = _split_source_ids(match.group(1))
        claim, claim_start = _claim_before(text, match.start())
        citations.append(
            Citation(
                text=claim,
                source_ids=ids,
                start=claim_start,
                end=match.end(),
                marker=match.group(0),
            )
        )
        for sid in ids:
            if sid in sources_by_id:
                cited.add(sid)

    clean = _BRACKET_RE.sub("", text)
    # Collapse the double spaces left where ``foo [1] bar`` becomes ``foo  bar``.
    clean = re.sub(r"[ \t]+", " ", clean).strip()
    return citations, clean, cited


class CitationTracker:
    """Assigns ids to sources, formats a prompt, and parses citations back out.

    Use directly when you want fine control over how the LLM is invoked
    (e.g. plugging into :class:`~prompture.agents.Agent`,
    :class:`~prompture.agents.Conversation`, or a custom driver loop).
    For end-to-end RAG with citations, compose with
    :class:`~prompture.rag.RAGPipeline`.

    Args:
        sources: Any iterable of :class:`Source`, :class:`Document`, or
            :class:`VectorSearchResult` items.  Ids are assigned in
            iteration order starting at 1.
        instruction: Citation directive prepended to the formatted
            context.  Defaults to :data:`CITATION_INSTRUCTION`.
        question: Optional question text appended after the context.
            When ``None`` the prompt ends after the context block — the
            caller is responsible for adding the question.

    Example::

        tracker = CitationTracker.from_retrieval_results(results)
        prompt = tracker.build_prompt(question="Who founded the city?")
        response = driver.generate(prompt, {})
        answer = tracker.parse_response(response["text"])
        for c in answer.citations:
            print(c.text, "→", c.source_ids)
    """

    def __init__(
        self,
        sources: Iterable[Source | Document | VectorSearchResult | Any] | None = None,
        *,
        instruction: str = CITATION_INSTRUCTION,
        question: str | None = None,
    ) -> None:
        self._sources: list[Source] = _coerce_sources(sources) if sources is not None else []
        self._instruction = instruction
        self._question = question

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_retrieval_results(
        cls,
        results: Iterable[VectorSearchResult],
        *,
        instruction: str = CITATION_INSTRUCTION,
        question: str | None = None,
    ) -> CitationTracker:
        """Build a tracker from raw retriever output."""
        return cls(sources=results, instruction=instruction, question=question)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        instruction: str = CITATION_INSTRUCTION,
        question: str | None = None,
    ) -> CitationTracker:
        """Build a tracker from :class:`~prompture.rag.Document` objects."""
        return cls(sources=documents, instruction=instruction, question=question)

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def sources(self) -> list[Source]:
        """Return a copy of the source list (ids are stable strings)."""
        return list(self._sources)

    def add_source(self, source: Source | Document | VectorSearchResult | Any) -> Source:
        """Append a new source and return the typed :class:`Source` record."""
        coerced = _coerce_sources([source], start=len(self._sources) + 1)[0]
        self._sources.append(coerced)
        return coerced

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_context(self) -> str:
        """Render the numbered source block fed to the LLM."""
        if not self._sources:
            return "(no sources available)"
        return "\n\n".join(s.as_context_line() for s in self._sources)

    def build_prompt(self, question: str | None = None) -> str:
        """Render the full citation-instructed prompt.

        Layout::

            <instruction>

            Sources:
            [1] title
            text...

            [2] title
            text...

            Question: <question>

        When neither *question* nor the tracker's stored question is
        set, the ``Question:`` line is omitted.
        """
        q = question if question is not None else self._question
        parts: list[str] = [self._instruction.strip(), "", "Sources:", self.build_context()]
        if q:
            parts.extend(["", f"Question: {q}"])
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def parse_response(self, text: str, *, usage: dict[str, Any] | None = None) -> CitedAnswer:
        """Parse *text* into a :class:`CitedAnswer` using this tracker's sources."""
        citations, clean, cited = extract_citations(text, self._sources)
        return CitedAnswer(
            answer=text,
            clean_answer=clean,
            citations=citations,
            sources=list(self._sources),
            cited_source_ids=cited,
            usage=dict(usage) if usage else {},
        )
