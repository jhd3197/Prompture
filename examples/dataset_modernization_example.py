"""Synthetic dataset modernisation example.

Walks through every Phase 8C addition end-to-end against a tiny
in-memory corpus:

1. ``generate_qa_dataset`` with **personas** for stylistic diversity.
2. ``QualityFilter`` to drop junk before dedup.
3. ``DedupConfig(strategy="shingle")`` for paraphrase-aware dedup.
4. ``EvolInstruct`` to evolve the surviving pairs along depth /
   breadth / reasoning axes.
5. ``SelfInstruct`` to expand a small seed set into a larger one.

Falls back to a scripted LLM if Ollama isn't reachable so the demo
runs in any environment. With Ollama available, the same flow scales
to real corpora — swap the inline ``Document`` for a file path or
glob.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.agents.persona import Persona
from prompture.dataset import (
    DedupConfig,
    EvolInstruct,
    QAPair,
    QualityFilter,
    SelfInstruct,
    generate_qa_dataset,
)
from prompture.dataset.schemas import QAPairBatch
from prompture.rag.documents import Document

# =============================================================================
# Sample corpus — two short docs so the demo runs quickly
# =============================================================================

CORPUS = [
    Document(
        content=(
            "Mitochondria are membrane-bound organelles inside most "
            "eukaryotic cells. They generate adenosine triphosphate (ATP) "
            "via oxidative phosphorylation and have their own circular DNA "
            "inherited maternally."
        ),
        metadata={"source": "biology-textbook-1"},
    ),
    Document(
        content=(
            "Photosynthesis converts light energy into chemical energy in "
            "chloroplasts. The Calvin cycle fixes CO2 using NADPH and ATP "
            "produced by the light-dependent reactions in the thylakoid "
            "membrane."
        ),
        metadata={"source": "biology-textbook-2"},
    ),
]


# =============================================================================
# Scripted LLM fallback
# =============================================================================


class _ScriptedDataDriver:
    """Returns plausible synthetic responses keyed by prompt content."""

    model = "scripted/datagen"
    model_name = "scripted/datagen"

    _SCRIPT: dict[str, list[QAPair]] = {
        # Mitochondria chunk responses (one for each persona)
        "mitochondria-tutor": [
            QAPair(
                question="Where in the cell is ATP generated?",
                answer="ATP is generated in the mitochondria via oxidative phosphorylation.",
            ),
            QAPair(
                question="How is mitochondrial DNA inherited?",
                answer="Mitochondrial DNA is inherited maternally as a circular genome.",
            ),
        ],
        "mitochondria-quiz": [
            QAPair(
                question="What organelle hosts oxidative phosphorylation?",
                answer="The mitochondrion hosts oxidative phosphorylation.",
            ),
            QAPair(
                question="Is mitochondrial DNA circular or linear?",
                answer="Mitochondrial DNA is circular.",
            ),
        ],
        # Photosynthesis chunk responses
        "photosynthesis-tutor": [
            QAPair(
                question="In which organelle does photosynthesis occur?",
                answer="Photosynthesis occurs in chloroplasts.",
            ),
            QAPair(
                question="What molecules drive the Calvin cycle?",
                answer="NADPH and ATP from the light-dependent reactions drive the Calvin cycle.",
            ),
        ],
        "photosynthesis-quiz": [
            QAPair(
                question="What does the Calvin cycle fix?",
                answer="The Calvin cycle fixes CO2.",
            ),
            QAPair(
                question="Where do the light-dependent reactions take place?",
                answer="They take place in the thylakoid membrane.",
            ),
        ],
        # Evol-Instruct evolutions
        "evol-depth": [QAPair(
            question="In a cell deprived of oxygen, what happens to ATP output?",
            answer="Without oxygen, oxidative phosphorylation halts and ATP yield drops sharply.",
        )],
        "evol-breadth": [QAPair(
            question="What other organelle has its own DNA besides mitochondria?",
            answer="Chloroplasts also have their own circular DNA, inherited maternally in most plants.",
        )],
        "evol-reasoning": [QAPair(
            question="Why does a marathon runner's muscle pH drop when ATP demand exceeds aerobic supply?",
            answer="When demand outpaces oxidative phosphorylation, lactic-acid fermentation takes over to produce ATP, lowering pH.",
        )],
        # Self-Instruct expansion
        "self-instruct-batch": [
            QAPair(
                question="What is the role of NADPH in photosynthesis?",
                answer="NADPH provides the reducing power needed by the Calvin cycle to fix CO2.",
            ),
            QAPair(
                question="What is a chloroplast?",
                answer="A chloroplast is a double-membraned organelle in plant cells where photosynthesis takes place.",
            ),
            QAPair(
                question="What pigment captures light in plants?",
                answer="Chlorophyll captures light, primarily in the red and blue wavelengths.",
            ),
        ],
    }

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "", "meta": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}}

    # When the example uses extract_with_model, prompture invokes our patched
    # _fake_extract directly — we don't need a real generate path.


def _pick_driver_or_mock() -> tuple[str, bool]:
    """Return (model_string, using_real_llm)."""
    try:
        from prompture.drivers.ollama_driver import OllamaDriver

        driver = OllamaDriver(
            endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        )
        driver.generate("ping", {"temperature": 0.0, "max_tokens": 4})
        print(f"[LLM: ollama/{driver.model}]\n")
        return f"ollama/{driver.model}", True
    except Exception as e:
        print(f"[LLM: scripted fallback — Ollama not reachable: {type(e).__name__}]\n")
        return "scripted/datagen", False


# When Ollama isn't around we monkey-patch ``extract_with_model`` to return
# pre-baked pairs keyed by the persona / operator the prompt mentions.
def _content_keyed_extract(*args: Any, **kwargs: Any) -> dict[str, Any]:
    text = args[1] if len(args) > 1 else kwargs.get("text", "")
    text_lower = (text or "").lower()

    # Determine which canned response to return based on prompt content.
    key: str
    if "tutor" in text_lower and "mitochond" in text_lower:
        key = "mitochondria-tutor"
    elif "quiz" in text_lower and "mitochond" in text_lower:
        key = "mitochondria-quiz"
    elif "tutor" in text_lower and "photosynth" in text_lower:
        key = "photosynthesis-tutor"
    elif "quiz" in text_lower and "photosynth" in text_lower:
        key = "photosynthesis-quiz"
    elif "more specific" in text_lower:
        key = "evol-depth"
    elif "broaden" in text_lower or "sibling" in text_lower:
        key = "evol-breadth"
    elif "multi-step" in text_lower or "reasoning" in text_lower:
        key = "evol-reasoning"
    elif "example pairs" in text_lower or "training data" in text_lower:
        key = "self-instruct-batch"
    elif "mitochond" in text_lower:
        key = "mitochondria-tutor"
    else:
        key = "photosynthesis-tutor"

    pairs = _ScriptedDataDriver._SCRIPT.get(key, [])
    if "{" in str(kwargs.get("model_cls", "")) or "QAPairBatch" in str(args[0] if args else ""):
        return {
            "model": QAPairBatch(pairs=pairs),
            "json_string": "",
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80, "cost": 0.0},
        }
    # Single QAPair extraction (EvolInstruct)
    return {
        "model": pairs[0] if pairs else QAPair(question="?", answer="."),
        "json_string": "",
        "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80, "cost": 0.0},
    }


# =============================================================================
# Examples
# =============================================================================


def example_personas_and_filters(model: str) -> list[QAPair]:
    print("=" * 70)
    print("STEP 1: generate_qa_dataset with personas + quality_filter + shingle dedup")
    print("=" * 70)

    tutor = Persona(
        name="tutor",
        system_prompt="You are a patient biology tutor. Ask clear, foundational questions and explain answers in plain English.",
    )
    quiz = Persona(
        name="quiz",
        system_prompt="You write multiple-choice-style quiz questions. Be terse, direct, factual.",
    )

    pairs, stats = generate_qa_dataset(
        CORPUS,
        model=model,
        personas=[tutor, quiz],
        quality_filter=QualityFilter(),
        dedup=DedupConfig(strategy="shingle", threshold=0.7, shingle_size=4),
        return_stats=True,
    )

    print(f"  raw pairs:        {stats['raw_count']}")
    if "filter" in stats:
        print(f"  filter dropped:   {stats['filter']['dropped']}")
    if "dedup_removed" in stats:
        print(f"  dedup removed:    {stats['dedup_removed']} ({stats['dedup_strategy']})")
    print(f"  surviving pairs:  {len(pairs)}")
    for i, p in enumerate(pairs[:5], 1):
        print(f"    [{i}] Q: {p.question}")
        print(f"        A: {p.answer}")
    print()
    return pairs


def example_evol_instruct(model: str, base_pairs: list[QAPair]) -> list[QAPair]:
    print("=" * 70)
    print("STEP 2: EvolInstruct — depth / breadth / reasoning evolutions")
    print("=" * 70)
    if not base_pairs:
        print("  (no input pairs — skipping)\n")
        return []

    evol = EvolInstruct(model=model, seed=42)
    seeds = base_pairs[:1]  # one source pair keeps the demo short
    result = evol.evolve_batch(seeds, operators=["depth", "breadth", "reasoning"], rounds=1)
    print(f"  source pairs: {len(seeds)}")
    print(f"  evolved:      {len(result.evolved)}")
    print(f"  failures:     {result.failures}")
    print(f"  LLM calls:    {result.meta.get('calls', 0)}")
    for i, p in enumerate(result.evolved, 1):
        print(f"    [{i}] Q: {p.question}")
        print(f"        A: {p.answer}")
    print()
    return result.evolved


def example_self_instruct(model: str, seeds: list[QAPair]) -> list[QAPair]:
    print("=" * 70)
    print("STEP 3: SelfInstruct — expand seeds into more in-style pairs")
    print("=" * 70)
    if not seeds:
        print("  (no seeds — skipping)\n")
        return []

    si = SelfInstruct(
        model=model,
        topic="introductory cell biology study questions",
        seed_random=random.Random(0),
        seed_sample_size=3,
    )
    result = si.expand(
        seeds=seeds[:3] if len(seeds) >= 3 else seeds,
        target_count=5,
        batch_size=3,
        max_rounds=3,
    )
    print(f"  seeds:            {len(seeds[:3] if len(seeds) >= 3 else seeds)}")
    print(f"  generated:        {len(result.pairs)}")
    print(f"  rounds completed: {result.rounds_completed}")
    print(f"  LLM calls:        {result.meta.get('calls', 0)}")
    for i, p in enumerate(result.pairs, 1):
        print(f"    [{i}] Q: {p.question}")
        print(f"        A: {p.answer}")
    print()
    return result.pairs


def main() -> None:
    model, real = _pick_driver_or_mock()
    if real:
        all_base = example_personas_and_filters(model)
        evolved = example_evol_instruct(model, all_base)
        _ = example_self_instruct(model, all_base + evolved)
    else:
        from unittest.mock import patch

        with (
            patch("prompture.dataset.synth.extract_with_model", _content_keyed_extract),
            patch("prompture.dataset.evol.extract_with_model", _content_keyed_extract),
            patch("prompture.dataset.seed.extract_with_model", _content_keyed_extract),
        ):
            all_base = example_personas_and_filters(model)
            evolved = example_evol_instruct(model, all_base)
            _ = example_self_instruct(model, all_base + evolved)


if __name__ == "__main__":
    main()
