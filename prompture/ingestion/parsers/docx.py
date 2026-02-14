"""DOCX parser wrapping Tukuy DOCX transformers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..document import DocumentContent
from .base import BaseParser

logger = logging.getLogger("prompture.ingestion.parsers.docx")


class DocxParser(BaseParser):
    """Parse DOCX files using Tukuy's ``DocxToTextTransformer``.

    Requires ``python-docx``.
    Install with ``pip install prompture[ingest]``.
    """

    extensions = [".docx", ".doc"]
    mime_types = [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ]

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

        # --- Extract text via Tukuy ---
        try:
            from tukuy.plugins.docx import DocxToTextTransformer

            transformer = DocxToTextTransformer("_ingest_docx_text")
            text = transformer._transform(path_str)
            if text.startswith("[error]"):
                raise ImportError(text)
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX ingestion. Install with: pip install prompture[ingest]"
            ) from None

        # --- Per-paragraph "page" approximation ---
        # DOCX doesn't have real pages, but we split by paragraphs
        # for chunking compatibility.
        page_texts: list[str] | None = None
        try:
            from docx import Document

            doc = Document(path_str)
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            if paragraphs:
                page_texts = paragraphs
        except Exception:
            pass

        # --- Metadata via Tukuy ---
        metadata: dict[str, Any] = {}
        try:
            from tukuy.plugins.docx import DocxExtractMetadataTransformer

            meta_transformer = DocxExtractMetadataTransformer("_ingest_docx_meta")
            meta = meta_transformer._transform(path_str)
            if isinstance(meta, dict) and "error" not in meta:
                metadata = {k: v for k, v in meta.items() if k != "path" and v}
        except Exception:
            pass

        return DocumentContent(
            text=text,
            file_type="docx",
            source_path=path_str,
            metadata=metadata,
            page_count=len(page_texts) if page_texts else None,
            page_texts=page_texts,
            char_count=len(text),
        )
