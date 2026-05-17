"""Core RAG chunker abstractions.

Defines :class:`TextChunker` and :class:`AsyncTextChunker` abstract base
classes together with a thin :class:`_SyncToAsyncChunker` adapter that
wraps a sync chunker in :func:`asyncio.to_thread` to provide trivial
async support.

Chunkers consume :class:`~prompture.rag.documents.Document` objects and
return smaller ``Document`` chunks suitable for embedding.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from ..documents import Document


class TextChunker(ABC):
    """Split text or :class:`Document` objects into smaller chunks.

    Subclasses implement :meth:`split_text`.  The base class provides
    :meth:`split_documents` which preserves and extends parent metadata
    with ``chunk_index`` and ``chunk_count`` for each emitted chunk.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **_kwargs: Any):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> list[str]:
        """Split a raw string into smaller chunks."""
        raise NotImplementedError

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split each Document's content and propagate metadata.

        Each emitted chunk inherits the parent document's metadata, with
        ``chunk_index`` and ``chunk_count`` added.  ``parent_source`` is
        set when the parent had a ``source`` key, so callers can trace a
        chunk back to its origin even if downstream code rewrites
        ``source``.
        """
        result: list[Document] = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            count = len(chunks)
            parent_source = doc.metadata.get("source")
            for i, chunk in enumerate(chunks):
                new_meta = dict(doc.metadata)
                new_meta["chunk_index"] = i
                new_meta["chunk_count"] = count
                if parent_source is not None and "parent_source" not in new_meta:
                    new_meta["parent_source"] = parent_source
                result.append(Document(content=chunk, metadata=new_meta))
        return result


class AsyncTextChunker(ABC):
    """Async counterpart to :class:`TextChunker`."""

    chunk_size: int = 1000
    chunk_overlap: int = 200

    @abstractmethod
    async def split_text(self, text: str) -> list[str]:
        raise NotImplementedError

    async def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            chunks = await self.split_text(doc.content)
            count = len(chunks)
            parent_source = doc.metadata.get("source")
            for i, chunk in enumerate(chunks):
                new_meta = dict(doc.metadata)
                new_meta["chunk_index"] = i
                new_meta["chunk_count"] = count
                if parent_source is not None and "parent_source" not in new_meta:
                    new_meta["parent_source"] = parent_source
                result.append(Document(content=chunk, metadata=new_meta))
        return result


class _SyncToAsyncChunker(AsyncTextChunker):
    """Wraps a sync :class:`TextChunker` in :func:`asyncio.to_thread`.

    Mirrors the loader-side ``_SyncToAsyncLoader`` adapter so chunkers
    backed by purely synchronous algorithms gain a no-fuss async sibling
    without duplicating implementation logic.
    """

    def __init__(self, sync_chunker: TextChunker):
        self._sync = sync_chunker
        self.chunk_size = getattr(sync_chunker, "chunk_size", 1000)
        self.chunk_overlap = getattr(sync_chunker, "chunk_overlap", 200)

    async def split_text(self, text: str) -> list[str]:
        return await asyncio.to_thread(self._sync.split_text, text)

    async def split_documents(self, documents: list[Document]) -> list[Document]:
        return await asyncio.to_thread(self._sync.split_documents, documents)
