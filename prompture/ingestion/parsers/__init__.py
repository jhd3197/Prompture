"""Parser registry for document ingestion.

Follows the same lazy-factory pattern used by
``prompture.drivers.registry``.
"""

from __future__ import annotations

import logging
from typing import Callable

from .base import BaseParser

logger = logging.getLogger("prompture.ingestion.parsers")

# Type alias for parser factory functions.
ParserFactory = Callable[[], BaseParser]

# Internal registries
_PARSER_REGISTRY: dict[str, ParserFactory] = {}
_EXT_MAPPING: dict[str, str] = {}
_MIME_MAPPING: dict[str, str] = {}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def register_parser(
    name: str,
    factory: ParserFactory,
    *,
    extensions: list[str] | None = None,
    mime_types: list[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Register a parser factory.

    Args:
        name: Canonical parser name (e.g. ``"pdf"``).  Lowercased.
        factory: Zero-arg callable that returns a :class:`BaseParser`.
        extensions: File extensions this parser handles (e.g. ``[".pdf"]``).
        mime_types: MIME types this parser handles.
        overwrite: Allow replacing an existing registration.
    """
    name = name.lower()
    if name in _PARSER_REGISTRY and not overwrite:
        raise ValueError(f"Parser '{name}' is already registered. Use overwrite=True to replace it.")
    _PARSER_REGISTRY[name] = factory
    for ext in extensions or []:
        _EXT_MAPPING[ext.lower()] = name
    for mime in mime_types or []:
        _MIME_MAPPING[mime.lower()] = name
    logger.debug("Registered parser: %s", name)


def get_parser(name: str) -> BaseParser:
    """Instantiate and return a parser by name.

    Args:
        name: Canonical parser name.

    Raises:
        ValueError: If no parser is registered under *name*.
    """
    name = name.lower()
    if name not in _PARSER_REGISTRY:
        available = ", ".join(sorted(_PARSER_REGISTRY.keys())) or "(none)"
        raise ValueError(
            f"No parser registered for '{name}'. Available parsers: {available}. "
            f"Install dependencies with: pip install prompture[ingest]"
        )
    return _PARSER_REGISTRY[name]()


def list_parsers() -> list[str]:
    """Return a sorted list of registered parser names."""
    return sorted(_PARSER_REGISTRY.keys())


def is_parser_registered(name: str) -> bool:
    """Check whether a parser is registered."""
    return name.lower() in _PARSER_REGISTRY


def unregister_parser(name: str) -> bool:
    """Remove a parser registration.  Returns True if it existed."""
    name = name.lower()
    if name in _PARSER_REGISTRY:
        del _PARSER_REGISTRY[name]
        # Clean up extension / MIME mappings
        for k, v in list(_EXT_MAPPING.items()):
            if v == name:
                del _EXT_MAPPING[k]
        for k, v in list(_MIME_MAPPING.items()):
            if v == name:
                del _MIME_MAPPING[k]
        return True
    return False


# ------------------------------------------------------------------
# Built-in parser registrations (lazy — only instantiated on first use)
# ------------------------------------------------------------------

register_parser(
    "pdf",
    lambda: _lazy_import("prompture.ingestion.parsers.pdf", "PdfParser"),
    extensions=[".pdf"],
    mime_types=["application/pdf"],
)
register_parser(
    "docx",
    lambda: _lazy_import("prompture.ingestion.parsers.docx", "DocxParser"),
    extensions=[".docx", ".doc"],
    mime_types=[
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ],
)
register_parser(
    "html",
    lambda: _lazy_import("prompture.ingestion.parsers.html", "HtmlParser"),
    extensions=[".html", ".htm", ".xhtml"],
    mime_types=["text/html", "application/xhtml+xml"],
)
register_parser(
    "markdown",
    lambda: _lazy_import("prompture.ingestion.parsers.markdown", "MarkdownParser"),
    extensions=[".md", ".markdown", ".mdx", ".txt", ".rst"],
    mime_types=["text/markdown", "text/plain"],
)
register_parser(
    "csv",
    lambda: _lazy_import("prompture.ingestion.parsers.csv_parser", "CsvParser"),
    extensions=[".csv", ".tsv"],
    mime_types=["text/csv", "text/tab-separated-values"],
)
register_parser(
    "xlsx",
    lambda: _lazy_import("prompture.ingestion.parsers.xlsx", "XlsxParser"),
    extensions=[".xlsx", ".xls"],
    mime_types=[
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ],
)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _lazy_import(module_path: str, class_name: str) -> BaseParser:
    """Import a parser class lazily and return an instance."""
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls()
