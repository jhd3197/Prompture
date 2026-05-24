"""Types for citation tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Source:
    """A single citable passage.

    Attributes:
        id: Short identifier the LLM is asked to cite (e.g. ``"1"``,
            ``"2"``).  Stable for the lifetime of a single
            :class:`~prompture.citations.CitationTracker` invocation.
        text: The full passage text fed to the LLM.
        title: Optional human-readable title (filename, document title).
        url: Optional URL of the source document.
        score: Optional retrieval score, ``None`` when the source did
            not come from a scored retriever.
        metadata: Free-form metadata copied from the underlying
            :class:`~prompture.rag.Document` (page numbers, section
            headings, etc.).
    """

    id: str
    text: str
    title: str | None = None
    url: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_context_line(self) -> str:
        """Render this source as a single ``[id] title — text`` line."""
        head = f"[{self.id}]"
        if self.title:
            head += f" {self.title}"
        return f"{head}\n{self.text}"


@dataclass
class Citation:
    """A span of generated text linked to one or more :class:`Source` ids.

    Citations are produced by :func:`~prompture.citations.extract_citations`
    from bracketed references in the LLM's output.

    Attributes:
        text: The cited claim — the sentence (or fragment) immediately
            preceding the bracket reference.
        source_ids: List of source identifiers cited at this span.
        start: Character offset of the claim's start in the original
            answer text.
        end: Character offset (exclusive) of the bracket marker's end.
        marker: The literal bracket text from the LLM output
            (e.g. ``"[1]"``, ``"[1,3]"``).
    """

    text: str
    source_ids: list[str]
    start: int
    end: int
    marker: str = ""

    def first_source(self) -> str | None:
        """Convenience accessor for the first cited source id."""
        return self.source_ids[0] if self.source_ids else None


@dataclass
class CitedAnswer:
    """A citation-grounded answer.

    Sits next to :class:`~prompture.rag.RAGAnswer` and is the recommended
    return type from any flow where the user-facing claim should be
    traceable to retrieved passages.

    Attributes:
        answer: The final answer text as written by the LLM, with the
            bracketed citation markers preserved.
        clean_answer: ``answer`` with bracket markers stripped — what
            you'd render in a UI when citations are surfaced separately.
        citations: Ordered list of :class:`Citation` records.
        sources: Every source supplied to the LLM (cited or not).
        cited_source_ids: Set of source ids that were referenced at
            least once.  Useful for "footnotes" UI rendering.
        usage: Optional token / cost summary from the LLM call.
    """

    answer: str
    clean_answer: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    cited_source_ids: set[str] = field(default_factory=set)
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of supplied sources that the answer actually cited.

        Returns 0.0 when there are no sources.  Useful for evaluating
        whether the LLM grounded its answer or hallucinated.
        """
        if not self.sources:
            return 0.0
        return len(self.cited_source_ids) / len(self.sources)

    @property
    def is_grounded(self) -> bool:
        """``True`` when the answer contains at least one citation."""
        return bool(self.citations)

    def uncited_sources(self) -> list[Source]:
        """Return sources that were supplied but never cited."""
        return [s for s in self.sources if s.id not in self.cited_source_ids]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "answer": self.answer,
            "clean_answer": self.clean_answer,
            "citations": [
                {
                    "text": c.text,
                    "source_ids": c.source_ids,
                    "start": c.start,
                    "end": c.end,
                    "marker": c.marker,
                }
                for c in self.citations
            ],
            "sources": [
                {
                    "id": s.id,
                    "title": s.title,
                    "url": s.url,
                    "score": s.score,
                    "text": s.text,
                    "metadata": s.metadata,
                }
                for s in self.sources
            ],
            "cited_source_ids": sorted(self.cited_source_ids),
            "coverage": self.coverage,
            "is_grounded": self.is_grounded,
            "usage": self.usage,
        }
