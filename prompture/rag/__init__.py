"""Prompture RAG (Retrieval-Augmented Generation) layer.

Phase 10 introduces the document loader subset of the RAG stack:

* :class:`Document` — content + metadata container.
* :class:`DocumentLoader` / :class:`AsyncDocumentLoader` — abstract base
  classes for synchronous and asynchronous loaders.
* Built-in loaders for plain text, PDF, DOCX, HTML, Markdown, JSON / JSONL,
  CSV, EPUB, and XLSX.
* A name + extension-indexed registry exposing :func:`get_loader`,
  :func:`get_loader_for_path`, and async variants.

Subsequent phases will add chunkers (Phase 11), vector stores (Phase 12),
and retrievers / end-to-end pipelines (Phase 13).
"""

from .documents import (
    AsyncDocumentLoader,
    Document,
    DocumentLoader,
    _SyncToAsyncLoader,
)
from .loader_registry import (
    ASYNC_LOADER_REGISTRY,
    LOADER_REGISTRY,
    get_async_loader,
    get_async_loader_for_path,
    get_loader,
    get_loader_for_path,
    register_async_loader,
    register_loader,
)
from .loaders import (
    CSVLoader,
    DOCXLoader,
    EPUBLoader,
    HTMLLoader,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    XLSXLoader,
)

# ── Register built-in loaders ────────────────────────────────────────────────

_BUILTIN_LOADERS = {
    "text": TextLoader,
    "pdf": PDFLoader,
    "docx": DOCXLoader,
    "html": HTMLLoader,
    "markdown": MarkdownLoader,
    "json": JSONLoader,
    "csv": CSVLoader,
    "epub": EPUBLoader,
    "xlsx": XLSXLoader,
}

for _name, _cls in _BUILTIN_LOADERS.items():
    register_loader(_name, _cls, overwrite=True)
    register_async_loader(
        _name,
        (lambda c=_cls: _SyncToAsyncLoader(c())),
        overwrite=True,
    )

del _name, _cls

__all__ = [
    "ASYNC_LOADER_REGISTRY",
    "LOADER_REGISTRY",
    "AsyncDocumentLoader",
    "CSVLoader",
    "DOCXLoader",
    "Document",
    "DocumentLoader",
    "EPUBLoader",
    "HTMLLoader",
    "JSONLoader",
    "MarkdownLoader",
    "PDFLoader",
    "TextLoader",
    "XLSXLoader",
    "get_async_loader",
    "get_async_loader_for_path",
    "get_loader",
    "get_loader_for_path",
    "register_async_loader",
    "register_loader",
]
