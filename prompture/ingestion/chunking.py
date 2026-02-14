"""Document chunking for large-document extraction."""

from __future__ import annotations

from dataclasses import dataclass

from .document import DocumentContent


@dataclass
class ChunkingConfig:
    """Configuration for document chunking.

    Attributes:
        max_chars_per_chunk: Maximum characters per chunk.
        overlap_chars: Character overlap between adjacent chunks
            (character-based strategy only).
        strategy: Chunking strategy — ``"page"`` groups consecutive
            pages; ``"chars"`` splits at paragraph boundaries.
    """

    max_chars_per_chunk: int = 50_000
    overlap_chars: int = 500
    strategy: str = "page"  # "page" | "chars"


@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk of a larger document.

    Attributes:
        text: Chunk text content.
        chunk_index: Zero-based index of this chunk.
        total_chunks: Total number of chunks the document was split into.
        page_range: Inclusive ``(start, end)`` page range for page-based
            chunks (1-indexed).  ``None`` for character-based chunks.
    """

    text: str
    chunk_index: int
    total_chunks: int
    page_range: tuple[int, int] | None = None


def chunk_document(
    doc: DocumentContent,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Split a :class:`DocumentContent` into chunks.

    When ``config.strategy`` is ``"page"`` and the document has
    ``page_texts``, pages are grouped into chunks until
    ``max_chars_per_chunk`` is reached.  Otherwise, falls through
    to character-based splitting at paragraph boundaries.

    Args:
        doc: Parsed document content.
        config: Chunking configuration. Uses defaults if ``None``.

    Returns:
        A list of :class:`DocumentChunk` instances. If the document
        fits within a single chunk the list has one element.
    """
    if config is None:
        config = ChunkingConfig()

    # Short-circuit: document fits in one chunk
    if doc.char_count <= config.max_chars_per_chunk:
        return [
            DocumentChunk(
                text=doc.text,
                chunk_index=0,
                total_chunks=1,
                page_range=(1, doc.page_count) if doc.page_count else None,
            )
        ]

    # --- Page-based strategy ---
    if config.strategy == "page" and doc.page_texts:
        return _chunk_by_pages(doc, config)

    # --- Character-based strategy (fallback or explicit) ---
    return _chunk_by_chars(doc, config)


# ------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------


def _chunk_by_pages(doc: DocumentContent, config: ChunkingConfig) -> list[DocumentChunk]:
    """Group consecutive pages until max_chars_per_chunk is reached."""
    assert doc.page_texts is not None  # caller guarantees this

    chunks: list[_RawChunk] = []
    current_texts: list[str] = []
    current_chars = 0
    start_page = 0

    for i, page_text in enumerate(doc.page_texts):
        page_len = len(page_text)

        # If adding this page would exceed the limit, flush current chunk
        if current_texts and (current_chars + page_len) > config.max_chars_per_chunk:
            chunks.append(
                _RawChunk(
                    text="\n\n".join(current_texts),
                    page_range=(start_page + 1, i),  # 1-indexed, inclusive of last included page
                )
            )
            current_texts = []
            current_chars = 0
            start_page = i

        current_texts.append(page_text)
        current_chars += page_len

    # Flush remainder
    if current_texts:
        chunks.append(
            _RawChunk(
                text="\n\n".join(current_texts),
                page_range=(start_page + 1, len(doc.page_texts)),
            )
        )

    total = len(chunks)
    return [
        DocumentChunk(
            text=rc.text,
            chunk_index=idx,
            total_chunks=total,
            page_range=rc.page_range,
        )
        for idx, rc in enumerate(chunks)
    ]


def _chunk_by_chars(doc: DocumentContent, config: ChunkingConfig) -> list[DocumentChunk]:
    """Split text at paragraph boundaries with overlap."""
    text = doc.text
    paragraphs = text.split("\n\n")

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_parts and (current_len + para_len + 2) > config.max_chars_per_chunk:
            chunks.append("\n\n".join(current_parts))

            # Calculate overlap: take trailing paragraphs that fit
            # within overlap_chars
            overlap_parts: list[str] = []
            overlap_len = 0
            for p in reversed(current_parts):
                if overlap_len + len(p) + 2 > config.overlap_chars:
                    break
                overlap_parts.insert(0, p)
                overlap_len += len(p) + 2

            current_parts = overlap_parts
            current_len = overlap_len

        current_parts.append(para)
        current_len += para_len + 2

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    total = len(chunks)
    return [
        DocumentChunk(
            text=chunk_text,
            chunk_index=idx,
            total_chunks=total,
            page_range=None,
        )
        for idx, chunk_text in enumerate(chunks)
    ]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


@dataclass
class _RawChunk:
    """Temporary container during page-based chunking."""

    text: str
    page_range: tuple[int, int] | None = None
