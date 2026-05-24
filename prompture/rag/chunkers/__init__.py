"""Prompture RAG text chunkers.

Phase 11 adds chunker primitives that slice :class:`Document` objects
produced by the Phase 10 loaders into smaller pieces ready for
embedding.

Built-in chunkers:

* :class:`CharacterChunker` — fixed-size character windows with a single
  separator.
* :class:`RecursiveCharacterChunker` — LangChain-style recursive split
  over a hierarchy of separators.
* :class:`TokenChunker` — token-windowed splitting via ``tiktoken``
  (install ``prompture[rag-token]``).
* :class:`SemanticChunker` — sentence grouping driven by embedding
  cosine distance; needs an ``embedding_driver``.
* :class:`MarkdownChunker` — Markdown-aware splitter that preserves
  header context in metadata.

Async siblings for the deterministic chunkers are registered via a
``_SyncToAsyncChunker`` adapter that wraps the sync implementation in
:func:`asyncio.to_thread`.
"""

from __future__ import annotations

from .base import AsyncTextChunker, TextChunker, _SyncToAsyncChunker
from .character import CharacterChunker
from .chunker_registry import (
    ASYNC_CHUNKER_REGISTRY,
    CHUNKER_REGISTRY,
    get_async_chunker,
    get_chunker,
    register_async_chunker,
    register_chunker,
)
from .contextual import DEFAULT_CONTEXTUAL_PROMPT, ContextualChunker
from .markdown import MarkdownChunker
from .recursive import RecursiveCharacterChunker
from .semantic import SemanticChunker
from .token import TokenChunker

# ── Register built-in sync chunkers ──────────────────────────────────────────

_BUILTIN_CHUNKERS: dict[str, type[TextChunker]] = {
    "character": CharacterChunker,
    "recursive": RecursiveCharacterChunker,
    "token": TokenChunker,
    "semantic": SemanticChunker,
    "markdown": MarkdownChunker,
}

for _name, _cls in _BUILTIN_CHUNKERS.items():
    register_chunker(_name, _cls, overwrite=True)


# ContextualChunker is a wrapper, not a primary splitter — it requires a
# `base` chunker and an LLM driver at instantiation time. Register it under
# its own factory so users can still discover it via `get_chunker`.
def _contextual_factory(**kwargs):
    return ContextualChunker(**kwargs)


register_chunker("contextual", _contextual_factory, overwrite=True)


# Async siblings: wrap the sync class in _SyncToAsyncChunker.  SemanticChunker
# and TokenChunker require runtime arguments (embedding_driver / tiktoken
# installed) — their factories simply forward kwargs so the caller controls
# instantiation.
def _make_async_factory(sync_cls: type[TextChunker]):
    def _factory(**kwargs):
        return _SyncToAsyncChunker(sync_cls(**kwargs))

    return _factory


for _name, _cls in _BUILTIN_CHUNKERS.items():
    register_async_chunker(_name, _make_async_factory(_cls), overwrite=True)

# Async sibling for ContextualChunker — same wrapping pattern. The LLM call
# inside is sync; the adapter handles to_thread.
register_async_chunker(
    "contextual",
    lambda **kw: _SyncToAsyncChunker(ContextualChunker(**kw)),
    overwrite=True,
)

del _name, _cls

__all__ = [
    "ASYNC_CHUNKER_REGISTRY",
    "CHUNKER_REGISTRY",
    "DEFAULT_CONTEXTUAL_PROMPT",
    "AsyncTextChunker",
    "CharacterChunker",
    "ContextualChunker",
    "MarkdownChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "TextChunker",
    "TokenChunker",
    "get_async_chunker",
    "get_chunker",
    "register_async_chunker",
    "register_chunker",
]
