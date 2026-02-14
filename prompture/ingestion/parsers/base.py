"""Base parser ABC for document ingestion."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..document import DocumentContent


class BaseParser(ABC):
    """Abstract base class for document parsers.

    Subclasses must implement :meth:`parse`.  The default
    :meth:`async_parse` wraps the sync method via
    ``asyncio.to_thread()``.
    """

    # Subclasses should override with their supported extensions and MIME types.
    extensions: list[str] = []
    mime_types: list[str] = []

    @abstractmethod
    def parse(
        self,
        source: str | Path,
        *,
        pages: list[int] | None = None,
        max_file_size: int = 50 * 1024 * 1024,
        **kwargs: Any,
    ) -> DocumentContent:
        """Parse a file into a :class:`DocumentContent`.

        Args:
            source: Path to the file on disk.
            pages: Optional page indices to extract (0-based). Only
                   meaningful for page-oriented formats (PDF, DOCX).
            max_file_size: Maximum allowed file size in bytes.
            **kwargs: Parser-specific options.

        Returns:
            A frozen :class:`DocumentContent` with extracted text.

        Raises:
            FileNotFoundError: If *source* does not exist.
            ValueError: If the file exceeds *max_file_size*.
            ImportError: If required optional dependencies are missing.
        """

    async def async_parse(
        self,
        source: str | Path,
        *,
        pages: list[int] | None = None,
        max_file_size: int = 50 * 1024 * 1024,
        **kwargs: Any,
    ) -> DocumentContent:
        """Async version of :meth:`parse`.

        Default implementation delegates to ``asyncio.to_thread()``.
        """
        return await asyncio.to_thread(
            self.parse,
            source,
            pages=pages,
            max_file_size=max_file_size,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_file(source: str | Path, max_file_size: int) -> Path:
        """Validate that *source* exists and does not exceed *max_file_size*.

        Returns the resolved :class:`Path`.
        """
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        size = p.stat().st_size
        if size > max_file_size:
            mb = max_file_size / (1024 * 1024)
            raise ValueError(
                f"File size ({size:,} bytes) exceeds the maximum allowed ({max_file_size:,} bytes / {mb:.0f} MB): {p}"
            )
        return p
