"""File type detection based on extension."""

from __future__ import annotations

from pathlib import Path

# Extension -> canonical parser name
_EXT_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "docx",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".csv": "csv",
    ".tsv": "csv",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".txt": "markdown",  # plain text treated as markdown (no-op strip)
    ".rst": "markdown",
}

# MIME type -> canonical parser name
_MIME_MAP: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/markdown": "markdown",
    "text/csv": "csv",
    "text/tab-separated-values": "csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "text/plain": "markdown",
}


def detect_file_type(source: str | Path) -> str:
    """Detect canonical parser name from a file path extension.

    Args:
        source: File path (string or Path).

    Returns:
        Canonical parser name (e.g. ``"pdf"``, ``"docx"``).

    Raises:
        ValueError: If the extension is not recognized.
    """
    ext = Path(source).suffix.lower()
    if ext in _EXT_MAP:
        return _EXT_MAP[ext]
    raise ValueError(
        f"Unsupported file extension '{ext}'. "
        f"Supported extensions: {', '.join(sorted(_EXT_MAP.keys()))}. "
        f"Use file_type= to override detection."
    )


def detect_file_type_from_mime(mime_type: str) -> str:
    """Detect canonical parser name from a MIME type string.

    Args:
        mime_type: MIME type (e.g. ``"application/pdf"``).

    Returns:
        Canonical parser name.

    Raises:
        ValueError: If the MIME type is not recognized.
    """
    normalized = mime_type.strip().lower().split(";")[0]
    if normalized in _MIME_MAP:
        return _MIME_MAP[normalized]
    raise ValueError(f"Unsupported MIME type '{mime_type}'. Use file_type= to override detection.")
