"""Compose loaders + chunkers + ``extract_with_model`` into a dataset
generator.

The function :func:`generate_qa_dataset` walks a source (file path,
glob, list of paths, or a list of pre-loaded ``Document`` objects),
chunks each document, and asks an LLM to emit several question/answer
pairs grounded in each chunk.  Results are aggregated, de-duplicated by
question text, and optionally written to disk in JSONL, ShareGPT, or
Alpaca format.
"""

from __future__ import annotations

import asyncio
import glob as _glob
import logging
from pathlib import Path
from typing import Any, Literal, Union

from ..extraction.async_core import extract_with_model as _async_extract_with_model
from ..extraction.core import extract_with_model
from ..rag.documents import Document
from .formats import to_alpaca, to_jsonl, to_sharegpt, write_dataset
from .schemas import QAPair, QAPairBatch

logger = logging.getLogger("prompture.dataset")

OutputFormat = Literal["jsonl", "sharegpt", "alpaca"]

DEFAULT_PROMPT_TEMPLATE = (
    "You are a dataset author. Read the SOURCE TEXT below and write {n} "
    "high-quality question/answer pairs that someone studying this "
    "material would benefit from. Each question must be self-contained "
    "and each answer must be drawn strictly from the SOURCE TEXT — no "
    "outside knowledge. Cover different facts; do not repeat questions."
    "\n\nSOURCE TEXT:\n{chunk}"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_source(
    source: Union[str, Path, list[Union[str, Path]], list[Document]],
) -> list[Document]:
    """Turn any accepted *source* shape into a flat list of ``Document``."""
    from ..rag.loader_registry import get_loader_for_path

    # Case 1: already a list of Document objects
    if isinstance(source, list) and source and isinstance(source[0], Document):
        return list(source)  # type: ignore[arg-type]

    # Case 2: a single path or glob
    if isinstance(source, (str, Path)):
        paths = _expand_paths([source])
    else:
        # Case 3: list of paths / globs
        paths = _expand_paths(source)  # type: ignore[arg-type]

    docs: list[Document] = []
    for p in paths:
        loader = get_loader_for_path(p)
        docs.extend(loader.load(p))
    return docs


def _expand_paths(items: list[Union[str, Path]]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        s = str(item)
        if any(ch in s for ch in "*?[]"):
            for match in _glob.glob(s, recursive=True):
                out.append(Path(match))
        else:
            out.append(Path(s))
    return out


def _chunk_documents(docs: list[Document], chunker: Any) -> list[Document]:
    """Apply *chunker* to *docs* — default to a recursive chunker."""
    if chunker is None:
        from ..rag.chunkers import RecursiveCharacterChunker

        chunker = RecursiveCharacterChunker(chunk_size=1200, chunk_overlap=120)
    return chunker.split_documents(docs)


def _dedupe(pairs: list[QAPair]) -> list[QAPair]:
    """Drop pairs with the same (normalised) question text."""
    seen: set[str] = set()
    out: list[QAPair] = []
    for p in pairs:
        key = " ".join(p.question.lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _format_records(
    pairs: list[QAPair],
    output_format: OutputFormat,
) -> list[dict[str, Any]]:
    if output_format == "jsonl":
        return to_jsonl(pairs)
    if output_format == "sharegpt":
        return to_sharegpt(pairs)
    if output_format == "alpaca":
        return to_alpaca(pairs)
    raise ValueError(f"Unknown output_format: {output_format!r}")


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


def generate_qa_dataset(
    source: Union[str, Path, list[Union[str, Path]], list[Document]],
    *,
    model: str,
    n_per_chunk: int = 3,
    chunker: Any = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    output_path: Union[str, Path, None] = None,
    output_format: OutputFormat = "jsonl",
    options: dict[str, Any] | None = None,
    on_chunk: Any = None,
    max_chunks: int | None = None,
    dedupe: bool = True,
) -> list[QAPair]:
    """Generate a synthetic Q&A dataset from *source*.

    Args:
        source: One of — a file path, a glob (``"docs/**/*.pdf"``), a
            list of paths, or a pre-loaded list of :class:`Document`.
        model: Model string passed to ``extract_with_model``
            (e.g. ``"openai/gpt-4o-mini"``).
        n_per_chunk: How many Q&A pairs to request per chunk.
        chunker: A pre-built chunker instance.  Defaults to a
            ``RecursiveCharacterChunker(chunk_size=1200, chunk_overlap=120)``.
        prompt_template: Override the per-chunk prompt.  Must include
            the placeholders ``{n}`` and ``{chunk}``.
        output_path: When set, write the result to this path as JSONL
            (the file format is always one JSON record per line; the
            *record shape* is determined by ``output_format``).
        output_format: ``"jsonl"`` (default), ``"sharegpt"``, or
            ``"alpaca"``.
        options: Extra options forwarded to ``extract_with_model``.
        on_chunk: Optional callback ``fn(chunk_index, total, chunk)``
            invoked before each LLM call.  Use for progress bars.
        max_chunks: Stop after this many chunks (useful for previewing
            a large corpus).
        dedupe: Drop pairs with duplicate questions (case-insensitive,
            whitespace-normalised).  Default ``True``.

    Returns:
        The list of :class:`QAPair` instances actually emitted.  When
        ``output_path`` is set, the file is also written.

    Example::

        pairs = generate_qa_dataset(
            "policy.pdf",
            model="openai/gpt-4o-mini",
            n_per_chunk=4,
            output_path="training.jsonl",
            output_format="sharegpt",
        )
    """
    docs = _resolve_source(source)
    chunks = _chunk_documents(docs, chunker)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    if not chunks:
        logger.warning("No chunks produced from source — nothing to generate.")
        return []

    all_pairs: list[QAPair] = []
    for i, chunk in enumerate(chunks):
        if on_chunk is not None:
            on_chunk(i, len(chunks), chunk)
        prompt = prompt_template.format(n=n_per_chunk, chunk=chunk.content)
        try:
            result = extract_with_model(
                QAPairBatch,
                prompt,
                model_name=model,
                options=options,
            )
            batch: QAPairBatch = result["model"]
            all_pairs.extend(batch.pairs)
        except Exception as exc:  # pragma: no cover — best-effort across many chunks
            logger.warning(
                "Chunk %d/%d (source=%s) failed: %s",
                i + 1,
                len(chunks),
                chunk.metadata.get("source", "?"),
                exc,
            )

    if dedupe:
        all_pairs = _dedupe(all_pairs)

    if output_path is not None:
        records = _format_records(all_pairs, output_format)
        write_dataset(records, output_path)

    return all_pairs


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def agenerate_qa_dataset(
    source: Union[str, Path, list[Union[str, Path]], list[Document]],
    *,
    model: str,
    n_per_chunk: int = 3,
    chunker: Any = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    output_path: Union[str, Path, None] = None,
    output_format: OutputFormat = "jsonl",
    options: dict[str, Any] | None = None,
    on_chunk: Any = None,
    max_chunks: int | None = None,
    dedupe: bool = True,
    concurrency: int = 4,
) -> list[QAPair]:
    """Async variant of :func:`generate_qa_dataset` with bounded concurrency.

    Same arguments plus:
        concurrency: Maximum number of chunks processed concurrently.
            Defaults to 4.  Use 1 for serial async execution.
    """
    docs = _resolve_source(source)
    chunks = _chunk_documents(docs, chunker)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    if not chunks:
        logger.warning("No chunks produced from source — nothing to generate.")
        return []

    semaphore = asyncio.Semaphore(concurrency)

    async def _process(idx: int, chunk: Document) -> list[QAPair]:
        if on_chunk is not None:
            on_chunk(idx, len(chunks), chunk)
        async with semaphore:
            prompt = prompt_template.format(n=n_per_chunk, chunk=chunk.content)
            try:
                result = await _async_extract_with_model(
                    QAPairBatch,
                    prompt,
                    model_name=model,
                    options=options,
                )
                return list(result["model"].pairs)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Chunk %d/%d (source=%s) failed: %s",
                    idx + 1,
                    len(chunks),
                    chunk.metadata.get("source", "?"),
                    exc,
                )
                return []

    batches = await asyncio.gather(*[_process(i, c) for i, c in enumerate(chunks)])
    all_pairs: list[QAPair] = [p for batch in batches for p in batch]

    if dedupe:
        all_pairs = _dedupe(all_pairs)

    if output_path is not None:
        records = _format_records(all_pairs, output_format)
        write_dataset(records, output_path)

    return all_pairs
