"""Markdown-aware chunker.

Splits Markdown by header levels and tracks the active header hierarchy
in each emitted chunk's metadata.  Each chunk's ``metadata["headers"]``
contains the current ``# Header 1`` / ``## Header 2`` / ... context so
downstream retrievers can present results with their structural breadcrumb.
"""

from __future__ import annotations

import re
from typing import Any

from ..documents import Document
from .base import TextChunker

_DEFAULT_HEADERS: list[tuple[str, str]] = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


class MarkdownChunker(TextChunker):
    """Split Markdown text on header boundaries.

    Args:
        headers_to_split_on: List of ``(prefix, name)`` tuples — e.g.
            ``[("#", "Header 1"), ("##", "Header 2")]``.  Defaults to
            all six standard ATX header levels.
        return_each_line: When ``True``, each non-header line becomes a
            separate chunk that still carries the active header context
            in its metadata.  When ``False`` (default), all lines under
            a given header path are merged into one chunk.
        chunk_size: Accepted for interface symmetry but not used by
            this chunker.
        chunk_overlap: Accepted for interface symmetry but not used by
            this chunker.
    """

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        return_each_line: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs: Any,
    ):
        # Allow chunk_overlap >= chunk_size only if both are zero; otherwise
        # honor the base check so callers don't silently get broken behaviour.
        if chunk_size > 0 and chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.headers_to_split_on = (
            list(headers_to_split_on) if headers_to_split_on is not None else list(_DEFAULT_HEADERS)
        )
        # Sort by length descending so "##" matches before "#" when scanning.
        self.headers_to_split_on.sort(key=lambda pair: len(pair[0]), reverse=True)
        self.return_each_line = return_each_line
        self._extra: dict[str, Any] = dict(kwargs)

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        """Return content chunks (without header metadata).

        Use :meth:`split_documents` when you need the per-chunk header
        breadcrumb in metadata.
        """
        return [seg["content"] for seg in self._segments(text)]

    # ------------------------------------------------------------------
    def split_documents(self, documents: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in documents:
            segments = self._segments(doc.content)
            count = len(segments)
            parent_source = doc.metadata.get("source")
            for i, seg in enumerate(segments):
                new_meta = dict(doc.metadata)
                new_meta["chunk_index"] = i
                new_meta["chunk_count"] = count
                if seg["headers"]:
                    new_meta["headers"] = dict(seg["headers"])
                if parent_source is not None and "parent_source" not in new_meta:
                    new_meta["parent_source"] = parent_source
                result.append(Document(content=seg["content"], metadata=new_meta))
        return result

    # ------------------------------------------------------------------
    def _segments(self, text: str) -> list[dict[str, Any]]:
        if not text:
            return []

        lines = text.splitlines()
        active: dict[str, str] = {}
        segments: list[dict[str, Any]] = []
        current_lines: list[str] = []
        current_headers: dict[str, str] = {}

        def flush(buffered: list[str], headers: dict[str, str]) -> None:
            content = "\n".join(buffered).strip()
            if content:
                segments.append({"content": content, "headers": dict(headers)})

        for raw_line in lines:
            header_match = self._match_header(raw_line)
            if header_match is not None:
                prefix, _name, value = header_match
                # Boundary: emit whatever has accumulated under the prior path.
                if current_lines:
                    flush(current_lines, current_headers)
                    current_lines = []
                # Drop any deeper-level header entries — they no longer apply.
                level = len(prefix)
                for known_prefix in list(active.keys()):
                    if len(known_prefix) >= level:
                        active.pop(known_prefix, None)
                active[prefix] = value
                current_headers = self._headers_dict(active)
                continue

            if self.return_each_line:
                if raw_line.strip():
                    flush([raw_line], current_headers)
            else:
                current_lines.append(raw_line)

        if current_lines:
            flush(current_lines, current_headers)
        return segments

    def _match_header(self, line: str) -> tuple[str, str, str] | None:
        stripped = line.lstrip()
        for prefix, name in self.headers_to_split_on:
            pattern = rf"^{re.escape(prefix)}\s+(.*)$"
            m = re.match(pattern, stripped)
            if m:
                # Make sure the prefix isn't a substring of a longer header
                # marker — e.g. don't let "#" match the start of "##".
                if stripped.startswith(prefix + "#"):
                    continue
                return prefix, name, m.group(1).strip()
        return None

    def _headers_dict(self, active: dict[str, str]) -> dict[str, str]:
        # Map active prefixes to their configured names, in level order.
        ordered: dict[str, str] = {}
        configured = {prefix: name for prefix, name in self.headers_to_split_on}
        for prefix in sorted(active.keys(), key=len):
            name = configured.get(prefix)
            if name:
                ordered[name] = active[prefix]
        return ordered
