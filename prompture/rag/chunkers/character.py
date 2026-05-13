"""Simple character-based chunker.

Splits text on a single configurable separator (default: blank lines) and
merges the resulting pieces into chunks of at most ``chunk_size``
characters with ``chunk_overlap`` characters of overlap.  Falls back to a
hard character cut when the separator is absent.
"""

from __future__ import annotations

from typing import Any

from .base import TextChunker


class CharacterChunker(TextChunker):
    """Fixed-size character window chunker with a single separator.

    Args:
        chunk_size: Target maximum size of each chunk (characters).
        chunk_overlap: Number of overlapping characters between adjacent
            chunks.  Must be strictly less than ``chunk_size``.
        separator: Substring used to split the input before merging.
            Defaults to ``"\\n\\n"``.
        keep_separator: If ``True``, the separator is retained at the
            start of each piece (except the very first).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        keep_separator: bool = False,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separator = separator
        self.keep_separator = keep_separator

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        if not text:
            return []

        if self.separator and self.separator in text:
            parts = text.split(self.separator)
            if self.keep_separator:
                parts = [parts[0]] + [self.separator + p for p in parts[1:]]
            parts = [p for p in parts if p != ""]
        else:
            # Hard-cut fallback: chop into chunk_size windows with overlap.
            return self._hard_cut(text)

        return self._merge_pieces(parts)

    # ------------------------------------------------------------------
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

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        joiner = "" if self.keep_separator else self.separator
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        joiner_len = len(joiner)

        for piece in pieces:
            piece_len = len(piece)
            additional = piece_len + (joiner_len if current else 0)
            if current and current_len + additional > self.chunk_size:
                chunks.append(joiner.join(current))
                # Build overlap tail from the just-emitted chunk.
                if self.chunk_overlap > 0 and chunks[-1]:
                    tail = chunks[-1][-self.chunk_overlap :]
                    current = [tail]
                    current_len = len(tail)
                else:
                    current = []
                    current_len = 0
                additional = piece_len + (joiner_len if current else 0)
            current.append(piece)
            current_len += additional

        if current:
            chunks.append(joiner.join(current))

        # If any individual piece exceeded chunk_size, hard-cut it.
        flattened: list[str] = []
        for c in chunks:
            if len(c) > self.chunk_size:
                flattened.extend(self._hard_cut(c))
            else:
                flattened.append(c)
        return flattened
