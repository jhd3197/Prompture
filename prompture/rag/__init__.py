"""Prompture RAG (Retrieval-Augmented Generation) layer.

Phase 10 introduced the document loader subset of the RAG stack;
Phase 11 added chunkers; Phase 12 added vector stores; Phase 13 adds
retrievers and an end-to-end :class:`RAGPipeline`:

* :class:`Document` — content + metadata container.
* :class:`DocumentLoader` / :class:`AsyncDocumentLoader` — abstract base
  classes for synchronous and asynchronous loaders.
* Built-in loaders for plain text, PDF, DOCX, HTML, Markdown, JSON / JSONL,
  CSV, EPUB, and XLSX.
* :class:`TextChunker` / :class:`AsyncTextChunker` plus concrete chunkers
  (:class:`CharacterChunker`, :class:`RecursiveCharacterChunker`,
  :class:`TokenChunker`, :class:`SemanticChunker`,
  :class:`MarkdownChunker`).
* :class:`VectorStore` / :class:`AsyncVectorStore` plus concrete adapters
  (:class:`ChromaVectorStore`, :class:`PineconeVectorStore`,
  :class:`QdrantVectorStore`, :class:`PgVectorStore`,
  :class:`FAISSVectorStore`, :class:`WeaviateVectorStore`).
* Name + extension-indexed registries exposing :func:`get_loader`,
  :func:`get_loader_for_path`, :func:`get_chunker`,
  :func:`get_vectorstore`, and async variants.

"""

from .chunkers import (
    ASYNC_CHUNKER_REGISTRY,
    CHUNKER_REGISTRY,
    AsyncTextChunker,
    CharacterChunker,
    MarkdownChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    TextChunker,
    TokenChunker,
    get_async_chunker,
    get_chunker,
    register_async_chunker,
    register_chunker,
)
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
from .pipeline import (
    DEFAULT_RAG_PROMPT,
    AsyncRAGPipeline,
    RAGAnswer,
    RAGPipeline,
)
from .retrievers import (
    ASYNC_RETRIEVER_REGISTRY,
    RETRIEVER_REGISTRY,
    AsyncRetriever,
    HybridRetriever,
    MMRRetriever,
    Retriever,
    VectorStoreRetriever,
    get_async_retriever,
    get_retriever,
    register_async_retriever,
    register_retriever,
)
from .vectorstores import (
    ASYNC_VECTORSTORE_REGISTRY,
    VECTORSTORE_REGISTRY,
    AsyncVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    PgVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    VectorSearchResult,
    VectorStore,
    WeaviateVectorStore,
    get_async_vectorstore,
    get_vectorstore,
    register_async_vectorstore,
    register_vectorstore,
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
    "ASYNC_CHUNKER_REGISTRY",
    "ASYNC_LOADER_REGISTRY",
    "ASYNC_RETRIEVER_REGISTRY",
    "ASYNC_VECTORSTORE_REGISTRY",
    "CHUNKER_REGISTRY",
    "DEFAULT_RAG_PROMPT",
    "LOADER_REGISTRY",
    "RETRIEVER_REGISTRY",
    "VECTORSTORE_REGISTRY",
    "AsyncDocumentLoader",
    "AsyncRAGPipeline",
    "AsyncRetriever",
    "AsyncTextChunker",
    "AsyncVectorStore",
    "CSVLoader",
    "CharacterChunker",
    "ChromaVectorStore",
    "DOCXLoader",
    "Document",
    "DocumentLoader",
    "EPUBLoader",
    "FAISSVectorStore",
    "HTMLLoader",
    "HybridRetriever",
    "JSONLoader",
    "MMRRetriever",
    "MarkdownChunker",
    "MarkdownLoader",
    "PDFLoader",
    "PgVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "RAGAnswer",
    "RAGPipeline",
    "RecursiveCharacterChunker",
    "Retriever",
    "SemanticChunker",
    "TextChunker",
    "TextLoader",
    "TokenChunker",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreRetriever",
    "WeaviateVectorStore",
    "XLSXLoader",
    "get_async_chunker",
    "get_async_loader",
    "get_async_loader_for_path",
    "get_async_retriever",
    "get_async_vectorstore",
    "get_chunker",
    "get_loader",
    "get_loader_for_path",
    "get_retriever",
    "get_vectorstore",
    "register_async_chunker",
    "register_async_loader",
    "register_async_retriever",
    "register_async_vectorstore",
    "register_chunker",
    "register_loader",
    "register_retriever",
    "register_vectorstore",
]
