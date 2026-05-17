"""Markdown document loader (stdlib, no external deps)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..documents import Document, DocumentLoader, LoaderSource

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class MarkdownLoader(DocumentLoader):
    """Load Markdown either as one blob or split by heading."""

    supported_extensions = (".md", ".markdown")

    def load(self, source: LoaderSource, **options: Any) -> list[Document]:
        mode = options.get("mode", "raw")  # "raw" | "by_heading"
        encoding = options.get("encoding", "utf-8")

        if isinstance(source, bytes):
            text = source.decode(encoding, errors="replace")
            source_label = "<bytes>"
        else:
            path = Path(source)
            text = path.read_text(encoding=encoding)
            source_label = str(path)

        if mode == "raw":
            return [
                Document(
                    content=text,
                    metadata={"source": source_label, "type": "markdown"},
                )
            ]

        # mode == "by_heading": split on heading boundaries.
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [
                Document(
                    content=text,
                    metadata={"source": source_label, "type": "markdown"},
                )
            ]

        docs: list[Document] = []
        # Capture any preamble text before the first heading.
        first_start = matches[0].start()
        if first_start > 0:
            preamble = text[:first_start].strip()
            if preamble:
                docs.append(
                    Document(
                        content=preamble,
                        metadata={
                            "source": source_label,
                            "type": "markdown",
                            "heading": None,
                            "level": 0,
                        },
                    )
                )

        for i, match in enumerate(matches):
            level = len(match.group(1))
            heading = match.group(2)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            docs.append(
                Document(
                    content=section,
                    metadata={
                        "source": source_label,
                        "type": "markdown",
                        "heading": heading,
                        "level": level,
                    },
                )
            )
        return docs
