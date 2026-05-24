"""Citation grounding for Prompture.

Adds claim-level provenance on top of the existing
:mod:`prompture.rag` retrievers and :class:`~prompture.rag.RAGAnswer`:

- :class:`Source` — a single retrieved passage with a stable id.
- :class:`Citation` — a span of generated text linked to one or more
  :class:`Source` ids.
- :class:`CitedAnswer` — final answer text + sources + citation list
  + aggregated coverage statistics.
- :class:`CitationTracker` — wraps a retriever, formats a citation-aware
  context for the LLM, and parses the response back into typed
  citations.
- :func:`extract_citations` — standalone parser for ``[n]``-style
  bracketed references against a list of sources.

Drops into existing code:

- Wrap any :class:`~prompture.rag.Retriever` to produce ``Source`` ids.
- Use :meth:`CitationTracker.build_prompt` to assemble a
  citation-instructed prompt.
- Parse the LLM response with :meth:`CitationTracker.parse_response` or
  the free function :func:`extract_citations`.
"""

from .core import (
    CITATION_INSTRUCTION,
    CitationTracker,
    extract_citations,
)
from .types import (
    Citation,
    CitedAnswer,
    Source,
)

__all__ = [
    "CITATION_INSTRUCTION",
    "Citation",
    "CitationTracker",
    "CitedAnswer",
    "Source",
    "extract_citations",
]
