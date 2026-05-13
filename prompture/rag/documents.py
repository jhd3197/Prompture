"""Core RAG document abstractions.

Defines the ``Document`` data container, the ``DocumentLoader`` /
``AsyncDocumentLoader`` abstract base classes, and a thin
``_SyncToAsyncLoader`` adapter that wraps a sync loader in
``asyncio.to_thread`` to provide trivial async support for I/O-bound file
readers.

These primitives are the foundation for the rest of the RAG layer
(chunkers, vector stores, retrievers — added in later phases).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

# A "source" can be a local file path, raw bytes, or (eventually) a URL.
# URL support is deferred to a later phase; loaders should treat str inputs
# that look like URLs as a documented future enhancement.
LoaderSource = Union[str, Path, bytes]


@dataclass
class Document:
    """A unit of content produced by a ``DocumentLoader``.

    Attributes:
        content: The full text content of the document chunk.
        metadata: Free-form metadata (source path, page number, sheet name,
            chapter title, etc.).  Loaders set conventional keys like
            ``source``, ``type``, and ``page``.
    """

    content: str
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)


class DocumentLoader(ABC):
    """Synchronous loader: returns or streams ``Document`` objects."""

    #: File extensions handled by this loader (e.g. ``(".pdf",)``).  Used by
    #: the loader registry for auto-detection via ``get_loader_for_path``.
    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        """Load all documents from ``source`` eagerly."""
        raise NotImplementedError

    def lazy_load(self, source: LoaderSource, **options: Any) -> Iterator[Document]:
        """Iterate over documents.

        Default implementation calls :meth:`load` and yields its items.
        Subclasses may override this for true streaming on large files.
        """
        yield from self.load(source, **options)


class AsyncDocumentLoader(ABC):
    """Asynchronous loader: returns or streams ``Document`` objects."""

    supported_extensions: tuple[str, ...] = ()

    @abstractmethod
    async def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        """Load all documents from ``source`` eagerly."""
        raise NotImplementedError

    async def alazy_load(self, source: LoaderSource, **options: Any) -> AsyncIterator[Document]:
        """Async iterate over documents.

        Default implementation awaits :meth:`load` and yields its items.
        """
        docs = await self.load(source, **options)
        for doc in docs:
            yield doc


class _SyncToAsyncLoader(AsyncDocumentLoader):
    """Wraps a sync ``DocumentLoader`` in :func:`asyncio.to_thread`.

    Provides a no-fuss async sibling for file-based loaders whose underlying
    parser libraries are synchronous.  Avoids duplicating loader logic.
    """

    def __init__(self, sync_loader: DocumentLoader):
        self._sync = sync_loader
        self.supported_extensions = sync_loader.supported_extensions

    async def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        return await asyncio.to_thread(self._sync.load, source, **options)
