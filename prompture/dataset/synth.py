"""Compose loaders + chunkers + ``extract_with_model`` into a dataset
generator.

The function :func:`generate_qa_dataset` walks a source (file path,
glob, list of paths, or a list of pre-loaded ``Document`` objects),
chunks each document, and asks an LLM to emit several question/answer
pairs grounded in each chunk.  Results are aggregated, optionally
quality-filtered, de-duplicated, and written to disk in JSONL,
ShareGPT, or Alpaca format.

Phase 8C extensions (all opt-in, default-off):

* ``personas`` — run each chunk through multiple :class:`Persona`
  voices for stylistic diversity in the generated set.
* ``quality_filter`` — drop short / runaway / refusal-flavoured
  pairs via :class:`prompture.dataset.QualityFilter`.
* ``dedup`` — choose between exact / shingle / semantic dedup via
  :class:`prompture.dataset.DedupConfig`. The legacy ``dedupe=True``
  argument is preserved and maps to exact dedup.
"""

from __future__ import annotations

import asyncio
import glob as _glob
import logging
from pathlib import Path
from typing import Any, Literal

from ..extraction.async_core import extract_with_model as _async_extract_with_model
from ..extraction.core import extract_with_model
from ..rag.documents import Document
from .dedup import DedupConfig, apply_dedup
from .filters import QualityFilter
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
    source: str | Path | list[str | Path] | list[Document],
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


def _expand_paths(items: list[str | Path]) -> list[Path]:
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
    """Drop pairs with the same (normalised) question text.

    .. deprecated:: 1.2.0
        Use :func:`prompture.dataset.dedup_exact` (or the unified
        :class:`DedupConfig`) directly. Retained for backward compatibility
        only.
    """
    from .dedup import dedup_exact

    kept, _removed = dedup_exact(pairs)
    return kept


def _resolve_personas(personas: list[Any] | None) -> list[tuple[str, str]]:
    """Normalise an optional list of personas into ``(name, prefix)`` tuples.

    Returns ``[("(none)", "")]`` when ``personas`` is missing — the
    chunk-processing loop iterates the list, so this preserves the
    "one call per chunk" default behaviour cleanly.
    """
    if not personas:
        return [("(none)", "")]
    voices: list[tuple[str, str]] = []
    for persona in personas:
        # Lazy import to keep this module dep-free when personas aren't used.
        try:
            from ..agents.persona import Persona
        except ImportError:
            Persona = None  # type: ignore[assignment, misc]

        if Persona is not None and isinstance(persona, Persona):
            name = persona.name or "persona"
            prefix = persona.render()
        elif isinstance(persona, str):
            name = "string-persona"
            prefix = persona
        elif hasattr(persona, "render"):
            name = getattr(persona, "name", None) or "persona"
            prefix = persona.render()
        else:
            name = str(persona)
            prefix = str(persona)
        voices.append((name, prefix))
    return voices


def _render_prompt(
    prompt_template: str,
    n_per_chunk: int,
    chunk: Document,
    persona_prefix: str,
) -> str:
    """Format ``prompt_template`` and prepend ``persona_prefix`` when given."""
    base = prompt_template.format(n=n_per_chunk, chunk=chunk.content)
    if not persona_prefix:
        return base
    return f"{persona_prefix.strip()}\n\n{base}"


def _resolve_dedup(dedup: DedupConfig | None, legacy_dedupe: bool) -> DedupConfig:
    """Reconcile the new ``dedup`` config with the legacy ``dedupe`` flag.

    Explicit ``dedup`` always wins. Otherwise: ``dedupe=True`` maps to
    ``DedupConfig(strategy="exact")``; ``False`` maps to ``"none"``.
    """
    if dedup is not None:
        return dedup
    return DedupConfig(strategy="exact" if legacy_dedupe else "none")


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
    source: str | Path | list[str | Path] | list[Document],
    *,
    model: str,
    n_per_chunk: int = 3,
    chunker: Any = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    output_path: str | Path | None = None,
    output_format: OutputFormat = "jsonl",
    options: dict[str, Any] | None = None,
    on_chunk: Any = None,
    max_chunks: int | None = None,
    dedupe: bool = True,
    personas: list[Any] | None = None,
    quality_filter: QualityFilter | None = None,
    dedup: DedupConfig | None = None,
    return_stats: bool = False,
) -> list[QAPair] | tuple[list[QAPair], dict[str, Any]]:
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
            whitespace-normalised). Equivalent to ``dedup=DedupConfig(strategy="exact")``.
            Defaults to ``True``. Ignored when ``dedup`` is supplied
            explicitly.
        personas: Optional list of :class:`prompture.agents.Persona`
            (or anything exposing ``.render()`` and ``.name``). When
            set, every chunk is processed once per persona — the
            persona's rendered text is prepended to the prompt so the
            generated pairs reflect each persona's voice. Multiplies
            the LLM call budget by ``len(personas)``.
        quality_filter: Optional :class:`QualityFilter`. Applied
            *after* generation but *before* dedup. When ``None``, no
            filtering happens. Pass ``QualityFilter()`` to enable the
            shape + length + refusal filter trio.
        dedup: Optional :class:`DedupConfig` overriding the legacy
            ``dedupe`` flag. Choose ``"exact"`` (cheap, the default
            via ``dedupe=True``), ``"shingle"`` (paraphrase-aware), or
            ``"semantic"`` (embedding-based).
        return_stats: When True, returns ``(pairs, stats_dict)``
            instead of just ``pairs``. ``stats_dict`` carries
            ``"raw_count"``, ``"filter"``, and ``"dedup_removed"``
            keys for observability.

    Returns:
        The list of :class:`QAPair` instances actually emitted. When
        ``output_path`` is set, the file is also written. When
        ``return_stats=True``, returns ``(pairs, stats)``.

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

    persona_voices = _resolve_personas(personas)

    all_pairs: list[QAPair] = []
    for i, chunk in enumerate(chunks):
        if on_chunk is not None:
            on_chunk(i, len(chunks), chunk)
        for persona_name, persona_prefix in persona_voices:
            prompt = _render_prompt(prompt_template, n_per_chunk, chunk, persona_prefix)
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
                    "Chunk %d/%d persona=%r (source=%s) failed: %s",
                    i + 1,
                    len(chunks),
                    persona_name,
                    chunk.metadata.get("source", "?"),
                    exc,
                )

    raw_count = len(all_pairs)
    stats: dict[str, Any] = {"raw_count": raw_count}

    if quality_filter is not None:
        all_pairs, filter_stats = quality_filter.filter(all_pairs)
        stats["filter"] = filter_stats.to_dict()

    dedup_cfg = _resolve_dedup(dedup, dedupe)
    if dedup_cfg.strategy != "none":
        all_pairs, removed = apply_dedup(all_pairs, dedup_cfg)
        stats["dedup_removed"] = removed
        stats["dedup_strategy"] = dedup_cfg.strategy

    if output_path is not None:
        records = _format_records(all_pairs, output_format)
        write_dataset(records, output_path)

    if return_stats:
        return all_pairs, stats
    return all_pairs


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------


async def agenerate_qa_dataset(
    source: str | Path | list[str | Path] | list[Document],
    *,
    model: str,
    n_per_chunk: int = 3,
    chunker: Any = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    output_path: str | Path | None = None,
    output_format: OutputFormat = "jsonl",
    options: dict[str, Any] | None = None,
    on_chunk: Any = None,
    max_chunks: int | None = None,
    dedupe: bool = True,
    concurrency: int = 4,
    personas: list[Any] | None = None,
    quality_filter: QualityFilter | None = None,
    dedup: DedupConfig | None = None,
    return_stats: bool = False,
) -> list[QAPair] | tuple[list[QAPair], dict[str, Any]]:
    """Async variant of :func:`generate_qa_dataset` with bounded concurrency.

    Same arguments as :func:`generate_qa_dataset` plus:

        concurrency: Maximum number of chunks processed concurrently.
            Defaults to 4. Use 1 for serial async execution.

    ``personas``, ``quality_filter``, ``dedup``, ``return_stats``
    behave identically to the sync version. With multiple personas
    the semaphore still caps total parallel calls — the personas are
    fanned out within each chunk's task slot.
    """
    docs = _resolve_source(source)
    chunks = _chunk_documents(docs, chunker)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    if not chunks:
        logger.warning("No chunks produced from source — nothing to generate.")
        return ([], {"raw_count": 0}) if return_stats else []

    persona_voices = _resolve_personas(personas)
    semaphore = asyncio.Semaphore(concurrency)

    async def _process(idx: int, chunk: Document) -> list[QAPair]:
        if on_chunk is not None:
            on_chunk(idx, len(chunks), chunk)
        out: list[QAPair] = []
        for persona_name, persona_prefix in persona_voices:
            async with semaphore:
                prompt = _render_prompt(prompt_template, n_per_chunk, chunk, persona_prefix)
                try:
                    result = await _async_extract_with_model(
                        QAPairBatch,
                        prompt,
                        model_name=model,
                        options=options,
                    )
                    out.extend(result["model"].pairs)
                except Exception as exc:  # pragma: no cover
                    logger.warning(
                        "Chunk %d/%d persona=%r (source=%s) failed: %s",
                        idx + 1,
                        len(chunks),
                        persona_name,
                        chunk.metadata.get("source", "?"),
                        exc,
                    )
        return out

    batches = await asyncio.gather(*[_process(i, c) for i, c in enumerate(chunks)])
    all_pairs: list[QAPair] = [p for batch in batches for p in batch]

    raw_count = len(all_pairs)
    stats: dict[str, Any] = {"raw_count": raw_count}

    if quality_filter is not None:
        all_pairs, filter_stats = quality_filter.filter(all_pairs)
        stats["filter"] = filter_stats.to_dict()

    dedup_cfg = _resolve_dedup(dedup, dedupe)
    if dedup_cfg.strategy != "none":
        all_pairs, removed = apply_dedup(all_pairs, dedup_cfg)
        stats["dedup_removed"] = removed
        stats["dedup_strategy"] = dedup_cfg.strategy

    if output_path is not None:
        records = _format_records(all_pairs, output_format)
        write_dataset(records, output_path)

    if return_stats:
        return all_pairs, stats
    return all_pairs
