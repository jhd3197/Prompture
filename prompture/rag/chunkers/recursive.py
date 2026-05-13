"""LangChain-style recursive character chunker.

Tries a list of separators from largest to smallest, recursively
splitting any piece that exceeds ``chunk_size`` with the next separator
in the list.  Adjacent small pieces are merged to fill ``chunk_size``
with ``chunk_overlap`` characters of overlap.
"""

from __future__ import annotations

import re
from typing import Any

from .base import TextChunker

_DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


class RecursiveCharacterChunker(TextChunker):
    """Recursively split text using a hierarchy of separators.

    Args:
        chunk_size: Target maximum chunk size (characters).
        chunk_overlap: Overlap between adjacent chunks (characters).
        separators: Ordered list of separators tried from largest to
            smallest.  Defaults to ``["\\n\\n", "\\n", ". ", " ", ""]``.
        keep_separator: If ``True``, the separator is retained at the
            start of each piece (except the first).
        is_separator_regex: If ``True``, separators are treated as
            regular expressions; otherwise they're literal substrings.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separators = list(separators) if separators is not None else list(_DEFAULT_SEPARATORS)
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    # ------------------------------------------------------------------
    def _split(self, text: str, separators: list[str]) -> list[str]:
        # Pick the first separator present in text, falling back to the last.
        separator = separators[-1]
        remaining: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining = separators[i + 1 :]
                break
            pattern = sep if self.is_separator_regex else re.escape(sep)
            if re.search(pattern, text):
                separator = sep
                remaining = separators[i + 1 :]
                break

        if separator == "":
            # Character-level hard cut for anything that survived this far.
            return self._hard_cut(text)

        pattern = separator if self.is_separator_regex else re.escape(separator)
        splits = self._split_with_separator(text, pattern, separator)

        # Recursively split any chunk that's still too large, and merge small ones.
        good_splits: list[str] = []
        final_chunks: list[str] = []
        for piece in splits:
            if len(piece) < self.chunk_size:
                good_splits.append(piece)
            else:
                if good_splits:
                    final_chunks.extend(self._merge(good_splits, separator))
                    good_splits = []
                if not remaining:
                    final_chunks.extend(self._hard_cut(piece))
                else:
                    final_chunks.extend(self._split(piece, remaining))
        if good_splits:
            final_chunks.extend(self._merge(good_splits, separator))
        return final_chunks

    def _split_with_separator(self, text: str, pattern: str, literal: str) -> list[str]:
        if self.keep_separator and literal:
            # Use a capture group so the separator is retained, then re-attach
            # it to the following segment.
            tokens = re.split(f"({pattern})", text)
            result: list[str] = []
            buf = tokens[0]
            i = 1
            while i < len(tokens):
                sep_match = tokens[i]
                segment = tokens[i + 1] if i + 1 < len(tokens) else ""
                if buf:
                    result.append(buf)
                buf = sep_match + segment
                i += 2
            if buf:
                result.append(buf)
            return [p for p in result if p != ""]
        return [p for p in re.split(pattern, text) if p != ""]

    def _merge(self, pieces: list[str], separator: str) -> list[str]:
        joiner = "" if self.keep_separator else separator
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        joiner_len = len(joiner)
        for piece in pieces:
            additional = len(piece) + (joiner_len if current else 0)
            if current and current_len + additional > self.chunk_size:
                merged = joiner.join(current)
                if merged:
                    chunks.append(merged)
                # Re-seed with overlap tail from the merged chunk.
                if self.chunk_overlap > 0 and merged:
                    tail = merged[-self.chunk_overlap :]
                    current = [tail]
                    current_len = len(tail)
                else:
                    current = []
                    current_len = 0
                additional = len(piece) + (joiner_len if current else 0)
            current.append(piece)
            current_len += additional
        if current:
            merged = joiner.join(current)
            if merged:
                chunks.append(merged)
        return chunks

    def _hard_cut(self, text: str) -> list[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size]
            if piece:
                chunks.append(piece)
            if start + self.chunk_size >= len(text):
                break
        return chunks
