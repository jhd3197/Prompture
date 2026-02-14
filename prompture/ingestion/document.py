"""Document content dataclass for ingested documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


@dataclass(frozen=True)
class DocumentContent:
    """Normalized document representation for extraction pipelines.

    Attributes:
        text: Extracted plain-text content from the document.
        file_type: Canonical file type identifier (e.g. ``"pdf"``, ``"docx"``).
        source_path: Original file path when loaded from disk.
        metadata: Parser-specific metadata (title, author, etc.).
        page_count: Total number of pages, if applicable.
        page_texts: Per-page text list for page-aware documents (PDF, DOCX).
        char_count: Total character count of ``text``.
    """

    text: str
    file_type: str
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    page_count: int | None = None
    page_texts: list[str] | None = None
    char_count: int = 0


# Public type alias accepted by all ingestion-aware APIs.
DocumentInput = Union[str, Path, "DocumentContent"]
