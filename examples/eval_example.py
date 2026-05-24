"""LLM-as-judge evaluation examples.

Runs all four evaluators in :mod:`prompture.eval`:

1. :class:`LLMJudge` — rubric scoring on a 1..5 scale
2. :class:`PairwiseJudge` — A/B preference with position-bias correction
3. :class:`SelfConsistencyEvaluator` — N-sample majority vote
4. :class:`FaithfulnessEvaluator` — claim-by-claim RAG grounding

By default the example uses Ollama (so it runs without any cloud API
key). Set ``OLLAMA_MODEL`` to control which model evaluates. When
Ollama isn't reachable the example falls back to a scripted driver
that simulates plausible judge responses — the demo still shows the
API end-to-end.

Run it
~~~~~~

::

    ollama pull llama3.1:8b
    OLLAMA_MODEL=llama3.1:8b python examples/eval_example.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.eval import (
    FaithfulnessEvaluator,
    LLMJudge,
    PairwiseJudge,
    SelfConsistencyEvaluator,
)

# =============================================================================
# Driver selection
# =============================================================================


class _ScriptedJudgeDriver:
    """Fallback driver that returns plausible judge responses keyed by content.

    The match logic is intentionally crude — the goal is to make the
    example illustrative without an LLM, not to be a real eval.
    """

    model = "scripted/judge"
    model_name = "scripted/judge"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        text = self._respond(prompt)
        return {
            "text": text,
            "meta": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70, "cost": 0.0},
        }

    @staticmethod
    def _respond(prompt: str) -> str:
        p = prompt.lower()

        # FaithfulnessEvaluator decomposition step ----------------------
        if "atomic claims" in p:
            if "eiffel" in p:
                return '["Paris is the capital of France.", "The Eiffel Tower is in Paris."]'
            if "wright" in p:
                return (
                    '["The Wright brothers built the first powered airplane.", '
                    '"They worked as bicycle mechanics in Dayton, Ohio.", '
                    '"Their first flight was in 1903."]'
                )
            return '["claim 1", "claim 2"]'

        # FaithfulnessEvaluator verification step -----------------------
        if "supported   (the evidence" in p or "contradicted   (the evidence" in p:
            # crude content match against evidence section
            if "eiffel" in p:
                return "SUPPORTED"
            if "wright" in p:
                # Mix verdicts so the example shows partial grounding.
                if "1903" in p[: p.find("claim:")]:
                    return "SUPPORTED"
                if "bicycle" in p:
                    return "SUPPORTED"
                return "SILENT"
            return "SUPPORTED"

        # PairwiseJudge -------------------------------------------------
        if "pick exactly one" in p and "response a:" in p and "response b:" in p:
            # Crude: prefer the longer response.
            a_start = p.find("response a:")
            b_start = p.find("response b:")
            tail_start = p.find("pick exactly one")
            a_text = p[a_start:b_start]
            b_text = p[b_start:tail_start]
            choice = "A" if len(a_text) > len(b_text) else "B"
            return f"CHOICE: {choice}\nREASON: Picked the more detailed answer."

        # LLMJudge ------------------------------------------------------
        if "impartial evaluator" in p:
            # Always a 4 with a positive reasoning, per_criterion mixed.
            return json.dumps(
                {
                    "score": 4,
                    "per_criterion": {"factuality": 5, "clarity": 4, "brevity": 3},
                    "reasoning": "Mostly factual and clear; slightly verbose.",
                }
            )

        # SelfConsistencyEvaluator (raw QA prompt) ----------------------
        if "17 * 23" in p or "17*23" in p:
            return "391"

        return "Acknowledged."


def _pick_driver():
    """Return Ollama if reachable, otherwise the scripted fallback."""
    try:
        from prompture.drivers.ollama_driver import OllamaDriver

        driver = OllamaDriver(
            endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        )
        # Reachability probe.
        driver.generate("ping", {"temperature": 0.0, "max_tokens": 4})
        print(f"[LLM: ollama/{driver.model}]\n")
        return driver
    except Exception as e:
        print(f"[LLM: scripted fallback — Ollama not reachable: {type(e).__name__}]\n")
        return _ScriptedJudgeDriver()


# =============================================================================
# Examples
# =============================================================================


def example_llm_judge(driver) -> None:
    print("=" * 70)
    print("EXAMPLE 1: LLMJudge — rubric scoring")
    print("=" * 70)
    judge = LLMJudge(
        driver=driver,
        criteria=[
            "factuality: claims must match the reference",
            "clarity: clean prose, no jargon",
            "brevity: short and to the point",
        ],
        scale=5,
        pass_threshold=4,
    )
    result = judge.evaluate(
        input_text="What is the capital of France?",
        output="Paris is the capital of France. It is also France's largest city.",
        reference="Paris",
    )
    print(f"  score:        {result.score}/5")
    print(f"  passed (>=4): {result.passed}")
    print(f"  per criterion: {result.per_criterion}")
    print(f"  reasoning:    {result.reasoning}")
    print()


def example_pairwise(driver) -> None:
    print("=" * 70)
    print("EXAMPLE 2: PairwiseJudge — A/B with position-bias correction")
    print("=" * 70)
    judge = PairwiseJudge(driver=driver)
    result = judge.compare(
        input_text="Summarise quantum entanglement in one sentence.",
        output_a="Two particles share a state.",
        output_b=(
            "Quantum entanglement is a physical phenomenon in which two or more particles "
            "share a quantum state, so measuring one instantly determines properties of the "
            "other regardless of distance."
        ),
    )
    print(f"  winner:     {result.winner}")
    print(f"  confidence: {result.confidence:.0%}")
    print(f"  votes:      {result.votes}")
    print()


def example_self_consistency(driver) -> None:
    print("=" * 70)
    print("EXAMPLE 3: SelfConsistencyEvaluator — sample N, vote")
    print("=" * 70)
    evaluator = SelfConsistencyEvaluator(driver=driver, n_samples=5)
    result = evaluator.evaluate("What is 17 * 23? Answer with just the number.")
    print(f"  consensus:  {result.consensus}")
    print(f"  agreement:  {result.agreement:.0%}")
    print(f"  counts:     {result.counts}")
    print()


def example_faithfulness(driver) -> None:
    print("=" * 70)
    print("EXAMPLE 4: FaithfulnessEvaluator — claim-by-claim RAG grounding")
    print("=" * 70)
    evaluator = FaithfulnessEvaluator(driver=driver)
    result = evaluator.evaluate(
        answer=(
            "The Wright brothers built the first powered airplane. They worked as "
            "bicycle mechanics in Dayton, Ohio. Their first flight was in 1903."
        ),
        evidence=[
            "Orville and Wilbur Wright, owners of a bicycle shop in Dayton, Ohio, "
            "designed and flew the first heavier-than-air powered aircraft in 1903.",
            "Their initial flight at Kitty Hawk lasted 12 seconds.",
        ],
    )
    print(f"  score:        {result.score:.0%}")
    print(f"  supported:    {len(result.supported)}")
    print(f"  unsupported:  {len(result.unsupported)}")
    print(f"  contradicted: {len(result.contradicted)}")
    if result.unsupported:
        print("  unsupported claims:")
        for c in result.unsupported:
            print(f"    - {c}")
    print()


def main() -> None:
    driver = _pick_driver()
    example_llm_judge(driver)
    example_pairwise(driver)
    example_self_consistency(driver)
    example_faithfulness(driver)


if __name__ == "__main__":
    main()
