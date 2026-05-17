"""HTML document loader (BeautifulSoup-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..documents import Document, DocumentLoader, LoaderSource

_HTML_HINT = "HTMLLoader requires 'beautifulsoup4' and 'markdownify'. Install with: pip install 'prompture[rag-html]'"


def _require_bs4():
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_HTML_HINT) from e
    return BeautifulSoup


def _require_markdownify():
    try:
        import markdownify  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_HTML_HINT) from e
    return markdownify


_STRIPPED_TAGS = ("script", "style", "nav", "footer")


class HTMLLoader(DocumentLoader):
    """Load HTML into clean text or Markdown."""

    supported_extensions = (".html", ".htm")

    def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        BeautifulSoup = _require_bs4()
        mode = options.get("mode", "text")  # "text" | "markdown"

        if isinstance(source, bytes):
            html_str = source.decode(options.get("encoding", "utf-8"), errors="replace")
            source_label = "<bytes>"
        else:
            path = Path(source)
            html_str = path.read_text(encoding=options.get("encoding", "utf-8"))
            source_label = str(path)

        # Prefer lxml when available; fall back to html.parser.
        try:
            soup = BeautifulSoup(html_str, "lxml")
        except Exception:
            soup = BeautifulSoup(html_str, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        # Strip noisy tags before extraction.
        for tag_name in _STRIPPED_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        if mode == "markdown":
            markdownify = _require_markdownify()
            content = markdownify.markdownify(str(soup))
        else:
            content = soup.get_text(separator="\n", strip=True)

        return [
            Document(
                content=content,
                metadata={
                    "source": source_label,
                    "title": title,
                    "type": "html",
                },
            )
        ]
