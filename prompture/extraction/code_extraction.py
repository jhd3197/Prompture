"""Salvage code from LLM output when tool calls fail.

When an LLM is configured with file-writing tools (e.g. ``write_file``) but
fails to call them — common with weaker models, providers that don't
support tool-calling at all, or runs that hit token limits mid-response —
the code we wanted often still lives inside the assistant's text response,
just unstructured.

This module provides two generic extractors used to recover that code:

* :func:`extract_fenced_blocks` — pulls markdown-fenced code blocks
  (`````html``, `````css``, `````js``, …) out of arbitrary text, optionally
  filtered by language.
* :func:`extract_html_document` — finds a full ``<!DOCTYPE html>…</html>``
  block in raw text and splits its inline ``<style>`` / ``<script>``
  contents out into separate buckets.

Both return plain data structures.  The caller decides what filenames to
write to (or whether to write at all) — these helpers are framework-free
and have no I/O side effects.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Fenced code blocks
# ---------------------------------------------------------------------------


_FENCE_PATTERN = re.compile(r"```(\w+)?\s*\n([\s\S]*?)```", re.MULTILINE)


@dataclasses.dataclass(frozen=True)
class FencedBlock:
    """A single markdown-fenced code block found in text.

    Args:
        language: Lower-cased language tag from the opening fence
            (``"html"``, ``"css"``, …).  Empty string when the fence had
            no language tag.
        content: The block body, stripped of leading/trailing whitespace.
    """

    language: str
    content: str


def extract_fenced_blocks(
    text: str,
    *,
    languages: Iterable[str] | None = None,
) -> list[FencedBlock]:
    """Find every fenced code block in ``text``.

    Args:
        text: Raw text to scan (typically an LLM's final response).
        languages: If given, only return blocks whose language tag (lower
            case) is in this iterable.  Pass e.g. ``["html", "css",
            "js", "javascript"]`` to ignore noise blocks.

    Returns:
        A list of :class:`FencedBlock` in the order they appear in the
        text.  Empty when nothing matches.
    """
    allowed = {lang.lower() for lang in languages} if languages is not None else None
    out: list[FencedBlock] = []
    for match in _FENCE_PATTERN.finditer(text):
        lang = (match.group(1) or "").lower()
        body = match.group(2).strip()
        if not body:
            continue
        if allowed is not None and lang not in allowed:
            continue
        out.append(FencedBlock(language=lang, content=body))
    return out


# ---------------------------------------------------------------------------
# Raw HTML extraction
# ---------------------------------------------------------------------------


_OUTER_FENCE_OPEN = re.compile(r"^```\w*\s*\n")
_OUTER_FENCE_CLOSE = re.compile(r"\n```\s*$")
_HTML_DOC_PATTERN = re.compile(
    r"(<!DOCTYPE html[\s\S]*?</html>|<html[\s\S]*?</html>)",
    re.IGNORECASE,
)
_STYLE_PATTERN = re.compile(r"<style[^>]*>([\s\S]*?)</style>", re.IGNORECASE)
_SCRIPT_PATTERN = re.compile(r"<script[^>]*>([\s\S]*?)</script>", re.IGNORECASE)


@dataclasses.dataclass(frozen=True)
class ExtractedHtml:
    """A salvaged HTML document with its inline CSS/JS pulled out.

    Args:
        html: The full ``<!DOCTYPE html>…</html>`` (or ``<html>…</html>``)
            block found in the input.  Empty when nothing matched.
        styles: Tuple of inline ``<style>`` block contents (in document
            order, whitespace-stripped, empty blocks dropped).
        scripts: Tuple of inline ``<script>`` block contents (same
            rules).  External-only ``<script src=…>`` tags drop out
            because their body is empty.
    """

    html: str
    styles: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        """True when a document was actually salvaged."""
        return bool(self.html)


def extract_html_document(text: str) -> ExtractedHtml:
    """Salvage a full HTML document from arbitrary text.

    Strips an outer markdown fence (if the whole response is wrapped in
    one), then looks for a ``<!DOCTYPE html>…</html>`` (or bare
    ``<html>…</html>``) block.  When found, also extracts each inline
    ``<style>`` and ``<script>`` block so callers can write them out
    alongside the HTML if they want.

    Args:
        text: Raw text to scan.

    Returns:
        An :class:`ExtractedHtml` whose ``found`` is True when a
        document was salvaged.  Always returns an instance — never None
        — so call sites can check ``.found`` without branching on type.
    """
    if not text:
        return ExtractedHtml(html="")

    stripped = _OUTER_FENCE_OPEN.sub("", text.strip(), count=1)
    stripped = _OUTER_FENCE_CLOSE.sub("", stripped, count=1)

    match = _HTML_DOC_PATTERN.search(stripped)
    if not match:
        return ExtractedHtml(html="")

    html = match.group(1)
    styles = tuple(s.strip() for s in _STYLE_PATTERN.findall(html) if s.strip())
    scripts = tuple(s.strip() for s in _SCRIPT_PATTERN.findall(html) if s.strip())
    return ExtractedHtml(html=html, styles=styles, scripts=scripts)


__all__ = [
    "ExtractedHtml",
    "FencedBlock",
    "extract_fenced_blocks",
    "extract_html_document",
]
