"""Synthesis: turn opened sources into a cited answer, conflicts and gaps.

Only sources whose content was actually read reach the model, numbered as
:class:`~prompture.citations.Source` records. After the call, every ``[n]``
marker is checked against that set; markers pointing anywhere else are
removed, so the final answer can only cite pages that were opened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..citations import CITATION_INSTRUCTION, CitationTracker, CitedAnswer, Source
from ..infra.compression import compress_messages

_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SECTION_RE = re.compile(
    r"^#{1,4}\s*(answer|conflicting claims|conflicts|coverage gaps|gaps|open questions)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_NONE_RE = re.compile(
    r"^(none|n/?a|nothing|no (conflicts?|gaps?|conflicting claims?|disagreements?))"
    r"( (found|identified|noted|significant))?[.!]?$",
    re.IGNORECASE,
)


@dataclass
class OpenedSource:
    """Content read from one URL, ready for synthesis."""

    n: int
    url: str
    title: str
    content: str
    reader: str | None = None
    kind: str | None = None
    sub_questions: list[int] = field(default_factory=list)

    def to_citation_source(self, text: str | None = None) -> Source:
        return Source(
            id=str(self.n),
            text=self.content if text is None else text,
            title=self.title or self.url,
            url=self.url,
            metadata={"reader": self.reader, "kind": self.kind},
        )


@dataclass
class SourceBundle:
    """The formatted source block plus what it took to build it."""

    text: str
    format: str  # "markdown" | "toon"
    sources: list[Source]
    chars_before: int = 0
    chars_after: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "sources": len(self.sources),
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
        }


def trim_content(text: str, limit: int) -> str:
    """Tidy whitespace and keep a head + tail window of *text* within *limit* chars."""
    msgs, _ = compress_messages(
        [{"role": "tool", "content": text or ""}],
        max_tool_chars=max(200, limit),
        trim_latest_tool=True,
    )
    return str(msgs[0]["content"])


def build_source_bundle(sources: list[OpenedSource], *, chars_per_source: int, compress: bool = False) -> SourceBundle:
    """Format opened sources for the synthesis prompt.

    With ``compress=True`` the bundle is a TOON table (``id, title, url,
    content``) — the uniform rows compress well against repeated keys. Falls
    back to the plain numbered layout when ``python-toon`` isn't installed.
    """
    before = sum(len(s.content) for s in sources)
    trimmed = [(s, trim_content(s.content, chars_per_source)) for s in sources]
    cite_sources = [s.to_citation_source(text) for s, text in trimmed]
    if compress and trimmed:
        try:
            from ..extraction.core import _json_to_toon

            rows = [
                {"id": s.n, "title": (s.title or s.url)[:200], "url": s.url, "content": " ".join(text.split())}
                for s, text in trimmed
            ]
            body = _json_to_toon(rows)
            return SourceBundle(body, "toon", cite_sources, before, len(body))
        except (RuntimeError, ValueError):
            pass
    tracker = CitationTracker(cite_sources)
    body = tracker.build_context()
    return SourceBundle(body, "markdown", cite_sources, before, len(body))


def build_synthesis_prompt(question: str, sub_questions: list[str], bundle: SourceBundle) -> str:
    subs = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, 1)) or "1. " + question
    if bundle.format == "toon":
        layout = (
            "Sources are given as a TOON table (a header row of field names, then one row per source). "
            "Cite a source by its `id`."
        )
    else:
        layout = "Each source starts with its number in square brackets."
    return (
        "You are a careful research analyst writing a sourced answer.\n\n"
        f"{CITATION_INSTRUCTION}\n"
        "Only cite the source numbers listed below. Never invent sources or cite anything else.\n\n"
        f"Question: {question}\n\n"
        f"Sub-questions investigated:\n{subs}\n\n"
        "Write Markdown with exactly these three sections:\n"
        "## Answer\n"
        "A direct, well-organized answer covering the sub-questions, every factual sentence cited.\n"
        "## Conflicting claims\n"
        "One bullet per point where sources disagree, citing each side (e.g. '[2] says X while [5] says Y'), "
        "or 'None found.'\n"
        "## Coverage gaps\n"
        "One bullet per sub-question or aspect the sources do not settle, or 'None.'\n\n"
        f"{layout}\n\nSources:\n{bundle.text}\n"
    )


def _bullets(block: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^(?:[-*•]|\d+[.)])\s+(.*)$", stripped)
        if m:
            if current:
                items.append(" ".join(current))
            current = [m.group(1).strip()]
        elif current:
            current.append(stripped)
        else:
            current = [stripped]
    if current:
        items.append(" ".join(current))
    return [i for i in items if i and not _NONE_RE.match(i.strip())]


def parse_synthesis(text: str) -> tuple[str, list[str], list[str]]:
    """Split the model output into ``(answer, conflicts, gaps)``.

    Missing headings degrade gracefully: the whole text becomes the answer.
    """
    text = (text or "").strip()
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return text, [], []
    sections: dict[str, str] = {}
    preamble = text[: matches[0].start()].strip()
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = m.group(1).lower()
        key = "answer" if name == "answer" else "conflicts" if "conflict" in name else "gaps"
        body = text[m.end() : end].strip()
        sections[key] = f"{sections[key]}\n{body}".strip() if key in sections else body
    answer = sections.get("answer") or preamble
    if preamble and sections.get("answer"):
        answer = f"{preamble}\n\n{answer}"
    return answer.strip(), _bullets(sections.get("conflicts", "")), _bullets(sections.get("gaps", ""))


def restrict_citations(text: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Remove citation ids not in *allowed*; drop markers left empty.

    Returns ``(clean_text, dropped_ids)``.
    """
    dropped: list[str] = []

    def _fix(m: re.Match[str]) -> str:
        ids = [s.strip() for s in m.group(1).split(",") if s.strip()]
        keep = [i for i in ids if i in allowed]
        dropped.extend(i for i in ids if i not in allowed)
        return f"[{', '.join(keep)}]" if keep else ""

    out = _MARKER_RE.sub(_fix, text or "")
    out = re.sub(r"[ \t]+([.,;:!?])", r"\1", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out, dropped


def parse_citations(answer: str, sources: list[Source]) -> CitedAnswer:
    """Parse ``[n]`` markers in *answer* against *sources* with the citations module."""
    return CitationTracker(sources).parse_response(answer)


def extractive_answer(sources: list[OpenedSource], reason: str, *, excerpt_chars: int = 400) -> str:
    """A no-LLM answer: one cited excerpt per opened source."""
    if not sources:
        return f"No sources could be opened ({reason}), so there is no cited answer."
    lines = [f"_Synthesis was skipped ({reason}). Key excerpts from the opened sources:_", ""]
    for s in sources:
        excerpt = " ".join((s.content or "").split())
        if len(excerpt) > excerpt_chars:
            cut = excerpt[:excerpt_chars]
            excerpt = (cut.rsplit(" ", 1)[0] if " " in cut else cut) + " ..."
        excerpt = _MARKER_RE.sub("", excerpt)
        lines.append(f"- **{s.title or s.url}**: {excerpt} [{s.n}]")
    return "\n".join(lines)


def fit_chars_per_source(
    *,
    default_chars: int,
    n_sources: int,
    tokens_remaining: int | None,
    answer_tokens: int,
    overhead_tokens: int = 900,
) -> int:
    """Shrink per-source characters so the synthesis prompt fits the token budget (~4 chars/token)."""
    if tokens_remaining is None or n_sources <= 0:
        return default_chars
    available = tokens_remaining - answer_tokens - overhead_tokens
    if available <= 0:
        return 0
    return max(0, min(default_chars, (available * 4) // n_sources))


__all__ = [
    "OpenedSource",
    "SourceBundle",
    "build_source_bundle",
    "build_synthesis_prompt",
    "extractive_answer",
    "fit_chars_per_source",
    "parse_citations",
    "parse_synthesis",
    "restrict_citations",
    "trim_content",
]
