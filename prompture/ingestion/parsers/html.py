"""HTML parser wrapping Tukuy HTML transformers + boilerplate removal."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..document import DocumentContent
from .base import BaseParser

logger = logging.getLogger("prompture.ingestion.parsers.html")

# Tags whose content is considered boilerplate
_BOILERPLATE_TAGS = re.compile(
    r"<(script|style|nav|footer|header|aside|noscript)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)


class HtmlParser(BaseParser):
    """Parse HTML files using Tukuy's ``StripHtmlTagsTransformer``.

    Removes boilerplate (``<script>``, ``<style>``, ``<nav>``, etc.)
    before stripping tags.

    Requires ``beautifulsoup4``.
    Install with ``pip install prompture[ingest]``.
    """

    extensions = [".html", ".htm", ".xhtml"]
    mime_types = ["text/html", "application/xhtml+xml"]

    def parse(
        self,
        source: str | Path,
        *,
        pages: list[int] | None = None,
        max_file_size: int = 50 * 1024 * 1024,
        **kwargs: Any,
    ) -> DocumentContent:
        p = self._check_file(source, max_file_size)
        path_str = str(p)

        # Read raw HTML
        raw_html = p.read_text(encoding="utf-8", errors="replace")

        # Remove boilerplate tags before stripping
        cleaned_html = _BOILERPLATE_TAGS.sub("", raw_html)

        # --- Strip tags via Tukuy ---
        try:
            from tukuy.plugins.html import StripHtmlTagsTransformer

            transformer = StripHtmlTagsTransformer("_ingest_html_strip")
            text = transformer._transform(cleaned_html)
        except ImportError:
            # Fallback: try beautifulsoup4 directly
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(cleaned_html, "html.parser")
                text = soup.get_text(separator="\n")
            except ImportError:
                raise ImportError(
                    "beautifulsoup4 is required for HTML ingestion. Install with: pip install prompture[ingest]"
                ) from None

        # Collapse excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # Extract title from <title> tag
        metadata: dict[str, Any] = {}
        title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.DOTALL | re.IGNORECASE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()

        return DocumentContent(
            text=text,
            file_type="html",
            source_path=path_str,
            metadata=metadata,
            page_count=None,
            page_texts=None,
            char_count=len(text),
        )
