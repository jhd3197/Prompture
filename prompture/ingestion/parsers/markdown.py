"""Markdown parser with frontmatter extraction and formatting strip."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..document import DocumentContent
from .base import BaseParser

# Regex patterns for stripping Markdown formatting
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BOLD_ITALIC_RE = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


class MarkdownParser(BaseParser):
    """Parse Markdown files with optional frontmatter extraction.

    Uses Tukuy's ``ExtractFrontmatterTransformer`` when available,
    otherwise falls back to a simple regex-based frontmatter parser.
    No external dependencies required (stdlib only).
    """

    extensions = [".md", ".markdown", ".mdx", ".txt", ".rst"]
    mime_types = ["text/markdown", "text/plain"]

    def parse(
        self,
        source: str | Path,
        *,
        pages: list[int] | None = None,
        max_file_size: int = 50 * 1024 * 1024,
        strip_formatting: bool = True,
        **kwargs: Any,
    ) -> DocumentContent:
        p = self._check_file(source, max_file_size)
        path_str = str(p)

        raw_text = p.read_text(encoding="utf-8", errors="replace")

        # --- Extract frontmatter ---
        metadata: dict[str, Any] = {}
        content = raw_text

        try:
            from tukuy.plugins.markdown import ExtractFrontmatterTransformer

            transformer = ExtractFrontmatterTransformer("_ingest_md_fm")
            result = transformer._transform(raw_text)
            if isinstance(result, dict):
                metadata = result.get("frontmatter", {})
                content = result.get("content", raw_text)
        except ImportError:
            # Fallback: simple regex-based frontmatter extraction
            content, metadata = self._extract_frontmatter_fallback(raw_text)

        # --- Optionally strip Markdown formatting ---
        if strip_formatting:
            text = self._strip_markdown(content)
        else:
            text = content

        text = text.strip()

        return DocumentContent(
            text=text,
            file_type="markdown",
            source_path=path_str,
            metadata=metadata,
            page_count=None,
            page_texts=None,
            char_count=len(text),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_frontmatter_fallback(raw: str) -> tuple[str, dict[str, Any]]:
        """Simple regex-based YAML frontmatter extraction."""
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", raw, re.DOTALL)
        if not match:
            return raw, {}

        raw_fm = match.group(1)
        content = match.group(2)
        fm: dict[str, Any] = {}

        for line in raw_fm.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                fm[key] = val

        return content, fm

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove Markdown formatting, leaving plain text."""
        # Remove images (before links to avoid partial match)
        text = _IMAGE_RE.sub(r"\1", text)
        # Replace links with their text
        text = _LINK_RE.sub(r"\1", text)
        # Remove code blocks (keep content)
        text = _CODE_BLOCK_RE.sub(r"\1", text)
        # Remove inline code backticks
        text = _INLINE_CODE_RE.sub(r"\1", text)
        # Remove heading markers
        text = _HEADING_RE.sub("", text)
        # Remove bold/italic markers
        text = _BOLD_ITALIC_RE.sub(r"\1", text)
        # Remove stray HTML tags
        text = _HTML_TAG_RE.sub("", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
