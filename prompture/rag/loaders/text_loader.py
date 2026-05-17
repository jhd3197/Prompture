"""Plain text document loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..documents import Document, DocumentLoader, LoaderSource


class TextLoader(DocumentLoader):
    """Read a plain-text file or bytes blob into a single ``Document``."""

    supported_extensions = (".txt", ".log", ".text")

    def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        encoding = options.get("encoding", "utf-8")
        if isinstance(source, bytes):
            content = source.decode(encoding, errors="replace")
            metadata = {"source": "<bytes>", "type": "text"}
        else:
            path = Path(source)
            content = path.read_text(encoding=encoding)
            metadata = {"source": str(path), "type": "text"}
        return [Document(content=content, metadata=metadata)]
