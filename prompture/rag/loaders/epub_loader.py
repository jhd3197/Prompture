"""EPUB document loader (ebooklib + BeautifulSoup)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..documents import Document, DocumentLoader, LoaderSource

_EPUB_HINT = "EPUBLoader requires 'ebooklib' and 'beautifulsoup4'. Install with: pip install 'prompture[rag-epub]'"


def _require_ebooklib():
    try:
        import ebooklib  # type: ignore
        from ebooklib import epub  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_EPUB_HINT) from e
    return ebooklib, epub


def _require_bs4():
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(_EPUB_HINT) from e
    return BeautifulSoup


class EPUBLoader(DocumentLoader):
    """Load an EPUB file as one ``Document`` per chapter (HTML item)."""

    supported_extensions = (".epub",)

    def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        ebooklib, epub = _require_ebooklib()
        BeautifulSoup = _require_bs4()

        if isinstance(source, bytes):
            # ebooklib's read_epub wants a file path; for bytes input, write
            # to a temp file so the same code path applies.
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
                tmp.write(source)
                tmp_path = tmp.name
            try:
                book = epub.read_epub(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            source_label = "<bytes>"
        else:
            path = Path(source)
            book = epub.read_epub(str(path))
            source_label = str(path)

        docs: list[Document] = []
        chapter_num = 0
        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            chapter_num += 1
            try:
                soup = BeautifulSoup(item.get_content(), "lxml")
            except Exception:
                soup = BeautifulSoup(item.get_content(), "html.parser")
            title_tag = soup.find(["h1", "h2", "title"])
            chapter_title = title_tag.get_text(strip=True) if title_tag else item.get_name()
            content = soup.get_text(separator="\n", strip=True)
            docs.append(
                Document(
                    content=content,
                    metadata={
                        "source": source_label,
                        "chapter_title": chapter_title,
                        "chapter_number": chapter_num,
                        "type": "epub",
                    },
                )
            )
        return docs
