"""PDF parser wrapping Tukuy PDF transformers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..document import DocumentContent
from .base import BaseParser

logger = logging.getLogger("prompture.ingestion.parsers.pdf")


class PdfParser(BaseParser):
    """Parse PDF files using Tukuy's ``PdfExtractTextTransformer``.

    Falls back through: pdfplumber -> pypdf (via Tukuy) -> pymupdf.
    Requires at least one of ``pdfplumber``, ``pypdf``, or ``pymupdf``.
    Install all with ``pip install prompture[ingest]``.
    """

    extensions = [".pdf"]
    mime_types = ["application/pdf"]

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

        # --- Attempt 1: Tukuy PdfExtractTextTransformer (uses pypdf) ---
        text: str | None = None
        page_texts: list[str] | None = None
        page_count: int | None = None
        metadata: dict[str, Any] = {}

        text, page_texts, page_count, metadata = self._try_tukuy(path_str, pages)

        if text is None:
            # --- Attempt 2: pdfplumber directly ---
            text, page_texts, page_count, metadata = self._try_pdfplumber(path_str, pages)

        if text is None:
            # --- Attempt 3: pymupdf ---
            text, page_texts, page_count, metadata = self._try_pymupdf(path_str, pages)

        if text is None:
            raise ImportError(
                "No PDF backend available. Install one of: pypdf, pdfplumber, pymupdf. "
                "Or install all ingestion dependencies with: pip install prompture[ingest]"
            )

        # Try to extract table text via Tukuy (best-effort)
        table_text = self._try_tables(path_str, pages)
        if table_text:
            text = text + "\n\n" + table_text

        # Try to extract metadata via Tukuy (best-effort)
        if not metadata:
            metadata = self._try_metadata(path_str)

        return DocumentContent(
            text=text,
            file_type="pdf",
            source_path=path_str,
            metadata=metadata,
            page_count=page_count,
            page_texts=page_texts,
            char_count=len(text),
        )

    # ------------------------------------------------------------------
    # Backend attempts
    # ------------------------------------------------------------------

    @staticmethod
    def _try_tukuy(
        path: str, pages: list[int] | None
    ) -> tuple[str | None, list[str] | None, int | None, dict[str, Any]]:
        """Try Tukuy PdfExtractTextTransformer (wraps pypdf)."""
        try:
            from tukuy.plugins.pdf import PdfExtractTextTransformer  # noqa: F401

            # For per-page extraction we need pypdf directly
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader

            reader = PdfReader(path)
            total_pages = len(reader.pages)
            page_indices = pages if pages is not None else list(range(total_pages))

            page_text_list: list[str] = []
            for i in page_indices:
                if 0 <= i < total_pages:
                    t = reader.pages[i].extract_text() or ""
                    page_text_list.append(t)

            full_text = "\n\n".join(page_text_list)
            return full_text, page_text_list, total_pages, {}
        except ImportError:
            return None, None, None, {}
        except Exception as exc:
            logger.debug("Tukuy PDF backend failed: %s", exc)
            return None, None, None, {}

    @staticmethod
    def _try_pdfplumber(
        path: str, pages: list[int] | None
    ) -> tuple[str | None, list[str] | None, int | None, dict[str, Any]]:
        """Try pdfplumber directly."""
        try:
            import pdfplumber
        except ImportError:
            return None, None, None, {}

        try:
            with pdfplumber.open(path) as pdf:
                total_pages = len(pdf.pages)
                page_indices = pages if pages is not None else list(range(total_pages))
                page_text_list: list[str] = []
                for i in page_indices:
                    if 0 <= i < total_pages:
                        t = pdf.pages[i].extract_text() or ""
                        page_text_list.append(t)
                full_text = "\n\n".join(page_text_list)
                return full_text, page_text_list, total_pages, {}
        except Exception as exc:
            logger.debug("pdfplumber backend failed: %s", exc)
            return None, None, None, {}

    @staticmethod
    def _try_pymupdf(
        path: str, pages: list[int] | None
    ) -> tuple[str | None, list[str] | None, int | None, dict[str, Any]]:
        """Try pymupdf (fitz)."""
        try:
            import fitz  # pymupdf
        except ImportError:
            return None, None, None, {}

        try:
            doc = fitz.open(path)
            total_pages = len(doc)
            page_indices = pages if pages is not None else list(range(total_pages))
            page_text_list: list[str] = []
            for i in page_indices:
                if 0 <= i < total_pages:
                    t = doc[i].get_text() or ""
                    page_text_list.append(t)
            doc.close()
            full_text = "\n\n".join(page_text_list)
            return full_text, page_text_list, total_pages, {}
        except Exception as exc:
            logger.debug("pymupdf backend failed: %s", exc)
            return None, None, None, {}

    @staticmethod
    def _try_tables(path: str, pages: list[int] | None) -> str:
        """Best-effort table extraction via Tukuy."""
        try:
            from tukuy.plugins.pdf import PdfExtractTablesTransformer

            transformer = PdfExtractTablesTransformer("_ingest_tables", pages=pages)
            tables = transformer._transform(path)
            if not tables or (
                isinstance(tables, list) and tables and isinstance(tables[0], dict) and "error" in tables[0]
            ):
                return ""
            parts: list[str] = []
            for table in tables:
                if isinstance(table, dict) and "headers" in table and "rows" in table:
                    header_line = " | ".join(table["headers"])
                    parts.append(header_line)
                    for row in table["rows"]:
                        row_line = " | ".join(str(row.get(h, "")) for h in table["headers"])
                        parts.append(row_line)
                    parts.append("")
            return "\n".join(parts).strip()
        except Exception:
            return ""

    @staticmethod
    def _try_metadata(path: str) -> dict[str, Any]:
        """Best-effort metadata extraction via Tukuy."""
        try:
            from tukuy.plugins.pdf import PdfExtractMetadataTransformer

            transformer = PdfExtractMetadataTransformer("_ingest_meta")
            meta = transformer._transform(path)
            if isinstance(meta, dict) and "error" not in meta:
                return {k: v for k, v in meta.items() if k != "path" and v}
            return {}
        except Exception:
            return {}
