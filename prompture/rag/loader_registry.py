"""Document loader registry and factory functions.

Mirrors the rerank / image-gen / video-gen registry pattern but for RAG
document loaders.  Loaders are registered by name and indexed by their
``supported_extensions`` so callers can resolve a loader either by an
explicit name (``get_loader("pdf")``) or directly from a file path
(``get_loader_for_path("foo.pdf")``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

from .documents import AsyncDocumentLoader, DocumentLoader

# ── Registries (public, mutable) ─────────────────────────────────────────────

#: name → zero-arg factory returning a sync DocumentLoader
LOADER_REGISTRY: dict[str, Callable[[], DocumentLoader]] = {}

#: name → zero-arg factory returning an AsyncDocumentLoader
ASYNC_LOADER_REGISTRY: dict[str, Callable[[], AsyncDocumentLoader]] = {}

#: extension (lower-case, with leading dot) → loader name
_EXTENSION_INDEX: dict[str, str] = {}
_ASYNC_EXTENSION_INDEX: dict[str, str] = {}


# ── Registration helpers ─────────────────────────────────────────────────────


def register_loader(
    name: str,
    factory: Callable[[], DocumentLoader],
    overwrite: bool = False,
) -> None:
    """Register a sync loader factory under ``name``.

    Also indexes each ``supported_extensions`` entry of the loader produced
    by ``factory()`` so :func:`get_loader_for_path` can resolve files by
    extension.
    """
    key = name.lower()
    if not overwrite and key in LOADER_REGISTRY:
        raise ValueError(f"Loader '{name}' already registered. Pass overwrite=True to replace.")
    LOADER_REGISTRY[key] = factory
    # Probe the loader for its supported extensions.  We instantiate eagerly
    # since loader constructors are cheap (lazy-import the heavy parser deps).
    try:
        probe = factory()
        for ext in getattr(probe, "supported_extensions", ()) or ():
            _EXTENSION_INDEX[ext.lower()] = key
    except Exception:
        # Don't let an instantiation failure during registration break the
        # registry — most loaders won't fail here, but if their parser dep
        # is missing we shouldn't crash on import.
        pass


def register_async_loader(
    name: str,
    factory: Callable[[], AsyncDocumentLoader],
    overwrite: bool = False,
) -> None:
    """Register an async loader factory under ``name``."""
    key = name.lower()
    if not overwrite and key in ASYNC_LOADER_REGISTRY:
        raise ValueError(f"Async loader '{name}' already registered. Pass overwrite=True to replace.")
    ASYNC_LOADER_REGISTRY[key] = factory
    try:
        probe = factory()
        for ext in getattr(probe, "supported_extensions", ()) or ():
            _ASYNC_EXTENSION_INDEX[ext.lower()] = key
    except Exception:
        pass


# ── Factory functions ────────────────────────────────────────────────────────


def get_loader(name: str) -> DocumentLoader:
    """Instantiate a sync loader by registered name."""
    key = name.lower()
    if key not in LOADER_REGISTRY:
        available = ", ".join(sorted(LOADER_REGISTRY)) or "<none>"
        raise KeyError(f"No sync loader registered under '{name}'. Available: {available}")
    return LOADER_REGISTRY[key]()


def get_async_loader(name: str) -> AsyncDocumentLoader:
    """Instantiate an async loader by registered name."""
    key = name.lower()
    if key not in ASYNC_LOADER_REGISTRY:
        available = ", ".join(sorted(ASYNC_LOADER_REGISTRY)) or "<none>"
        raise KeyError(f"No async loader registered under '{name}'. Available: {available}")
    return ASYNC_LOADER_REGISTRY[key]()


def _extension_of(path: Union[str, Path]) -> str:
    suffix = Path(str(path)).suffix.lower()
    if not suffix:
        raise KeyError(f"Path '{path}' has no file extension; cannot auto-detect a loader.")
    return suffix


def get_loader_for_path(path: Union[str, Path]) -> DocumentLoader:
    """Return a sync loader appropriate for ``path``'s file extension."""
    ext = _extension_of(path)
    if ext not in _EXTENSION_INDEX:
        known = ", ".join(sorted(_EXTENSION_INDEX)) or "<none>"
        raise KeyError(f"No loader registered for extension '{ext}'. Known extensions: {known}")
    return get_loader(_EXTENSION_INDEX[ext])


def get_async_loader_for_path(path: Union[str, Path]) -> AsyncDocumentLoader:
    """Return an async loader appropriate for ``path``'s file extension."""
    ext = _extension_of(path)
    if ext not in _ASYNC_EXTENSION_INDEX:
        known = ", ".join(sorted(_ASYNC_EXTENSION_INDEX)) or "<none>"
        raise KeyError(f"No async loader registered for extension '{ext}'. Known extensions: {known}")
    return get_async_loader(_ASYNC_EXTENSION_INDEX[ext])
