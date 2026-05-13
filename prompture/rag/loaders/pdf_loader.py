"""PDF document loader (pypdf-based)."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from ..documents import Document, DocumentLoader, LoaderSource

_PYPDF_HINT = "PDFLoader requires 'pypdf'. Install with: pip install 'prompture[rag-pdf]'"


def _require_pypdf():
    try:
        import pypdf  # type: ignore
    except ImportError as e:  # pragma: no cover - exercised via test mocks
        raise ImportError(_PYPDF_HINT) from e
    return pypdf


class PDFLoader(DocumentLoader):
    """Load PDFs page-by-page (or as a single concatenated document)."""

    supported_extensions = (".pdf",)

    def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        pypdf = _require_pypdf()
        mode = options.get("mode", "pages")  # "pages" | "single"

        if isinstance(source, bytes):
            stream: Any = BytesIO(source)
            source_label = "<bytes>"
        else:
            path = Path(source)
            stream = str(path)
            source_label = str(path)

        reader = pypdf.PdfReader(stream)
        total_pages = len(reader.pages)
        pages_text: list[str] = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")

        if mode == "single":
            content = "\n\n".join(pages_text)
            return [
                Document(
                    content=content,
                    metadata={
                        "source": source_label,
                        "total_pages": total_pages,
                        "type": "pdf",
                    },
                )
            ]

        docs: list[Document] = []
        for idx, text in enumerate(pages_text, start=1):
            docs.append(
                Document(
                    content=text,
                    metadata={
                        "source": source_label,
                        "page": idx,
                        "total_pages": total_pages,
                        "type": "pdf",
                    },
                )
            )
        return docs
