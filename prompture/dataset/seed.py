"""Self-Instruct seed-based dataset expansion.

Implements the `Self-Instruct <https://arxiv.org/abs/2212.10560>`_
pattern: given a small set of high-quality "seed" examples, ask the
LLM to generate more in the same style, then mix the new examples
back into the seed pool so subsequent rounds see broader inspiration.

Unlike :mod:`evol` (which transforms existing pairs along quality
axes), this module **generates from scratch** — useful when you have
a few hand-curated gold examples and need to scale to 1k-10k+ without
a corpus of source documents.

Quick start::

    from prompture.dataset.seed import SelfInstruct
    from prompture.dataset.schemas import QAPair

    seeds = [
        QAPair(question="What is mitosis?", answer="..."),
        QAPair(question="Define osmosis.", answer="..."),
    ]
    expander = SelfInstruct(model="openai/gpt-4o-mini")
    result = expander.expand(seeds, target_count=50, batch_size=5)
    print(len(result.pairs))  # ~50, modulo dedup
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from ..extraction.core import extract_with_model
from .dedup import dedup_exact
from .schemas import QAPair, QAPairBatch

logger = logging.getLogger(__name__)


#: Default expansion prompt. ``{topic}``, ``{n}``, ``{seed_block}`` are
#: substituted with the dataset topic (optional), per-call batch size,
#: and a formatted sample of K seed examples respectively.
DEFAULT_EXPANSION_PROMPT = (
    "You are creating training data for an instruction-following model.\n\n"
    "The dataset focus is: {topic}\n\n"
    "Below are example Q&A pairs to anchor the style, depth, and topic "
    "scope. Generate {n} NEW pairs in the same style but covering "
    "DIFFERENT subject matter. Do not paraphrase the examples — produce "
    "genuinely new questions that a learner studying the same topic "
    "would benefit from. Each answer must be complete, accurate, and "
    "self-contained.\n\n"
    "EXAMPLE PAIRS:\n{seed_block}\n\n"
    "Respond with a single JSON object: {{\"pairs\": [{{\"question\": ..., \"answer\": ...}}, ...]}}"
)


@dataclass
class SelfInstructResult:
    """Outcome of a :meth:`SelfInstruct.expand` call.

    Attributes:
        pairs: Generated pairs (does NOT include the original seeds).
        rounds_completed: How many expansion rounds finished before
            ``target_count`` was reached or ``max_rounds`` was hit.
        meta: Aggregated token / cost meta across all LLM calls.
    """

    pairs: list[QAPair] = field(default_factory=list)
    rounds_completed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def _format_seed_block(seeds: list[QAPair]) -> str:
    """Render a list of QAPairs as a numbered Q&A block."""
    parts: list[str] = []
    for i, seed in enumerate(seeds, 1):
        parts.append(f"{i}. Q: {seed.question}\n   A: {seed.answer}")
    return "\n\n".join(parts)


class SelfInstruct:
    """Expand a seed set into a larger Q&A dataset via iterative LLM generation.

    Args:
        model: Model string passed to :func:`extract_with_model`.
        topic: Short freeform tag describing the dataset focus (e.g.
            ``"intro biology study questions"``). Embedded in the
            prompt so generated content stays on-topic.
        prompt: Override the expansion-prompt template. Must contain
            ``{topic}``, ``{n}``, ``{seed_block}``.
        options: Forwarded to :func:`extract_with_model`. Defaults to
            ``temperature=0.7`` so each round produces diverse outputs.
        seed_sample_size: How many seed examples to include in each
            prompt. Defaults to 8 — too many crowds the context; too
            few collapses diversity.
        seed_random: Optional ``random.Random`` for deterministic seed
            sampling (useful in tests). Defaults to a fresh ``Random()``.
    """

    def __init__(
        self,
        *,
        model: str,
        topic: str = "general knowledge",
        prompt: str = DEFAULT_EXPANSION_PROMPT,
        options: dict[str, Any] | None = None,
        seed_sample_size: int = 8,
        seed_random: random.Random | None = None,
    ) -> None:
        for placeholder in ("{topic}", "{n}", "{seed_block}"):
            if placeholder not in prompt:
                raise ValueError(f"prompt is missing required placeholder {placeholder!r}.")
        if seed_sample_size < 1:
            raise ValueError(f"seed_sample_size must be >= 1; got {seed_sample_size}.")
        self.model = model
        self.topic = topic
        self.prompt = prompt
        self.options = dict(options) if options is not None else {"temperature": 0.7, "max_tokens": 1024}
        self.seed_sample_size = seed_sample_size
        self._rng = seed_random or random.Random()  # nosec B311 - non-cryptographic dataset sampling

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(
        self,
        seeds: list[QAPair],
        *,
        target_count: int,
        batch_size: int = 5,
        max_rounds: int | None = None,
        feed_back: bool = True,
        dedup_against_seeds: bool = True,
    ) -> SelfInstructResult:
        """Generate up to ``target_count`` new pairs from the seed pool.

        Args:
            seeds: Hand-curated example pairs. At least one is required.
            target_count: Stop after this many *new* pairs (post-dedup).
            batch_size: How many pairs to request per LLM call. Smaller
                batches give the model more attention per pair; larger
                batches amortise prompt overhead. Defaults to 5.
            max_rounds: Hard cap on the number of generation rounds.
                ``None`` (default) lets the loop run until
                ``target_count`` is reached.
            feed_back: When True (default), newly-generated pairs are
                added to the candidate seed pool so subsequent rounds
                see broader inspiration. When False, only the original
                seeds are ever sampled.
            dedup_against_seeds: When True (default), generated pairs
                that exactly-match a seed's question (normalised) are
                dropped. Disable when you want exact paraphrases.

        Returns:
            :class:`SelfInstructResult` with the new pairs, rounds
            completed, and aggregated meta.

        Raises:
            ValueError: when ``seeds`` is empty or ``target_count`` < 1.
        """
        if not seeds:
            raise ValueError("Self-Instruct requires at least one seed.")
        if target_count < 1:
            raise ValueError(f"target_count must be >= 1; got {target_count}.")

        result = SelfInstructResult()
        meta = {"calls": 0}
        # Pool starts as the seeds; new pairs join when feed_back=True.
        pool: list[QAPair] = list(seeds)

        # Maximum rounds to attempt — bounded so a buggy LLM can't loop forever.
        cap = max_rounds if max_rounds is not None else max(20, (target_count // max(batch_size, 1)) * 3)

        while len(result.pairs) < target_count and result.rounds_completed < cap:
            sample = self._sample_seeds(pool)
            new_pairs, call_meta = self._one_round(sample, n=batch_size)
            meta["calls"] += 1
            for k, v in call_meta.items():
                if isinstance(v, (int, float)):
                    meta[k] = (meta.get(k) or 0) + v
            result.rounds_completed += 1

            if not new_pairs:
                logger.debug("Self-Instruct round %d produced no pairs.", result.rounds_completed)
                continue

            # Drop anything that collides with the existing dataset (seeds + result).
            candidates: list[QAPair] = list(result.pairs)
            if dedup_against_seeds:
                candidates.extend(seeds)
            dedup_input = candidates + new_pairs
            kept, _removed = dedup_exact(dedup_input)
            # ``kept`` includes the prior items in order — slice off the front.
            fresh = kept[len(candidates) :]

            # Respect target_count exactly.
            room = target_count - len(result.pairs)
            fresh = fresh[:room]

            result.pairs.extend(fresh)
            if feed_back:
                pool.extend(fresh)

            if not fresh:
                logger.debug(
                    "Self-Instruct round %d: %d candidate(s) but all duplicates after dedup.",
                    result.rounds_completed,
                    len(new_pairs),
                )

        result.meta = meta
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sample_seeds(self, pool: list[QAPair]) -> list[QAPair]:
        """Return up to ``seed_sample_size`` shuffled samples from ``pool``."""
        if len(pool) <= self.seed_sample_size:
            return list(pool)
        return self._rng.sample(pool, self.seed_sample_size)

    def _one_round(
        self,
        seed_sample: list[QAPair],
        *,
        n: int,
    ) -> tuple[list[QAPair], dict[str, Any]]:
        """Run one expansion LLM call. Returns ``(new_pairs, call_meta)``."""
        rendered = self.prompt.format(
            topic=self.topic,
            n=n,
            seed_block=_format_seed_block(seed_sample),
        )
        try:
            result = extract_with_model(
                QAPairBatch,
                rendered,
                model_name=self.model,
                options=self.options,
            )
        except Exception as exc:
            logger.warning("Self-Instruct expansion call failed: %s", exc)
            return [], {}

        usage = result.get("usage") if isinstance(result, dict) else None
        call_meta: dict[str, Any] = {}
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if isinstance(usage.get(key), (int, float)):
                    call_meta[key] = int(usage[key])
            if isinstance(usage.get("cost"), (int, float)):
                call_meta["cost"] = float(usage["cost"])

        batch = result.get("model") if isinstance(result, dict) else None
        if isinstance(batch, QAPairBatch):
            return list(batch.pairs), call_meta

        # Fallback: parse raw JSON string.
        text = result.get("json_string", "") if isinstance(result, dict) else ""
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return [], call_meta
        raw_pairs = obj.get("pairs") if isinstance(obj, dict) else None
        if not isinstance(raw_pairs, list):
            return [], call_meta
        out: list[QAPair] = []
        for item in raw_pairs:
            if isinstance(item, dict):
                q = str(item.get("question") or "").strip()
                a = str(item.get("answer") or "").strip()
                if q and a:
                    out.append(QAPair(question=q, answer=a))
        return out, call_meta


__all__ = [
    "DEFAULT_EXPANSION_PROMPT",
    "SelfInstruct",
    "SelfInstructResult",
]
