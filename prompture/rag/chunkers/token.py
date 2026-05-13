"""Token-based chunker using ``tiktoken``.

Splits text into fixed-size windows of *tokens* (not characters) with
configurable overlap.  ``tiktoken`` is imported lazily so the base
install stays small; install ``prompture[rag-token]`` to use this
chunker.
"""

from __future__ import annotations

from typing import Any

from .base import TextChunker


class TokenChunker(TextChunker):
    """Chunk text by token count using ``tiktoken``.

    Args:
        chunk_size: Target maximum chunk size measured in *tokens*.
        chunk_overlap: Overlap between adjacent chunks measured in
            *tokens*.  Must be less than ``chunk_size``.
        encoding_name: ``tiktoken`` encoding name (default
            ``"cl100k_base"`` — works for GPT-3.5/4 and the
            ``text-embedding-3-*`` family).
        model_name: Optional model name; when supplied, the encoding is
            resolved via ``tiktoken.encoding_for_model(model_name)`` and
            overrides ``encoding_name``.

    Raises:
        ImportError: if ``tiktoken`` is not installed.  Install with
            ``pip install prompture[rag-token]``.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base",
        model_name: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        try:
            import tiktoken  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via importorskip in tests
            raise ImportError(
                "TokenChunker requires the 'tiktoken' package. Install with: pip install 'prompture[rag-token]'"
            ) from exc

        self._tiktoken = tiktoken
        if model_name:
            self._encoder = tiktoken.encoding_for_model(model_name)
        else:
            self._encoder = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name
        self.model_name = model_name

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        tokens = self._encoder.encode(text)
        if not tokens:
            return []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(self._encoder.decode(window))
            if start + self.chunk_size >= len(tokens):
                break
        return chunks
