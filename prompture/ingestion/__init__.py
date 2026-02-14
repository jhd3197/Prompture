"""Document ingestion: parse files into text for LLM extraction.

Public API
----------
- :func:`ingest` / :func:`async_ingest` — Smart constructors that
  auto-detect file type and return a :class:`DocumentContent`.
- :class:`DocumentContent` — Frozen dataclass holding parsed text.
- :class:`ChunkingConfig` / :class:`DocumentChunk` / :func:`chunk_document`
  — Split large documents for chunked extraction.
- :func:`register_parser` / :func:`get_parser` / :func:`list_parsers`
  — Parser registry (extensible).

Quick start::

    from prompture.ingestion import ingest

    doc = ingest("report.pdf")
    print(doc.text[:200])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking import ChunkingConfig, DocumentChunk, chunk_document
from .document import DocumentContent, DocumentInput
from .parsers import get_parser, is_parser_registered, list_parsers, register_parser, unregister_parser


def ingest(
    source: DocumentInput,
    *,
    file_type: str | None = None,
    pages: list[int] | None = None,
    max_file_size: int = 50 * 1024 * 1024,
    **kwargs: Any,
) -> DocumentContent:
    """Parse a document into a :class:`DocumentContent`.

    Accepts:
    - ``DocumentContent`` — returned as-is (passthrough).
    - ``str`` / ``pathlib.Path`` — auto-detects file type from extension
      and delegates to the appropriate parser.

    Args:
        source: File path or an already-parsed ``DocumentContent``.
        file_type: Override auto-detection (e.g. ``"pdf"``, ``"csv"``).
        pages: Page indices to extract (0-based). Only meaningful for
               page-oriented formats like PDF.
        max_file_size: Maximum file size in bytes (default 50 MB).
        **kwargs: Forwarded to the parser's ``parse()`` method.

    Returns:
        A frozen :class:`DocumentContent` instance.

    Raises:
        TypeError: If *source* is ``bytes`` or an unsupported type.
        ValueError: If the file extension is not recognized (and
                    *file_type* is not set).
        FileNotFoundError: If the file does not exist.
        ImportError: If required parser dependencies are missing.
    """
    # Passthrough
    if isinstance(source, DocumentContent):
        return source

    # Bytes are not supported (ambiguous without file_type)
    if isinstance(source, bytes):
        raise TypeError("Cannot ingest raw bytes without a file path. Write to a temp file or pass a file_type= hint.")

    # str / Path -> detect + parse
    if isinstance(source, (str, Path)):
        if file_type is None:
            from .detect import detect_file_type

            file_type = detect_file_type(source)

        parser = get_parser(file_type)
        return parser.parse(
            source,
            pages=pages,
            max_file_size=max_file_size,
            **kwargs,
        )

    raise TypeError(f"Unsupported source type: {type(source).__name__}")


async def async_ingest(
    source: DocumentInput,
    *,
    file_type: str | None = None,
    pages: list[int] | None = None,
    max_file_size: int = 50 * 1024 * 1024,
    **kwargs: Any,
) -> DocumentContent:
    """Async version of :func:`ingest`.

    Default parser implementations delegate to ``asyncio.to_thread()``.
    """
    # Passthrough
    if isinstance(source, DocumentContent):
        return source

    if isinstance(source, bytes):
        raise TypeError("Cannot ingest raw bytes without a file path. Write to a temp file or pass a file_type= hint.")

    if isinstance(source, (str, Path)):
        if file_type is None:
            from .detect import detect_file_type

            file_type = detect_file_type(source)

        parser = get_parser(file_type)
        return await parser.async_parse(
            source,
            pages=pages,
            max_file_size=max_file_size,
            **kwargs,
        )

    raise TypeError(f"Unsupported source type: {type(source).__name__}")


__all__ = [
    "ChunkingConfig",
    "DocumentChunk",
    "DocumentContent",
    "DocumentInput",
    "async_ingest",
    "chunk_document",
    "get_parser",
    "ingest",
    "is_parser_registered",
    "list_parsers",
    "register_parser",
    "unregister_parser",
]
