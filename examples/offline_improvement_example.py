"""Offline improvement: propose, evaluate on dev, report on held-out, promote.

Demonstrates :mod:`prompture.execution.improve` end to end, with no model calls:

1. Failing outcome records motivate two prompt **candidates** — one genuinely
   better, one deliberately worse.
2. A **candidate search optimizer** ranks them on the *development* split only.
   Touching the held-out split raises.
3. A **scorecard** records dev and held-out comparisons separately, together
   with the evaluation's own cost and the fixture provenance.
4. The **promotion ledger** refuses the weak candidate, accepts the good one,
   keeps the baseline, and rolls back on request.

The "model" is a deterministic scripted driver whose obedience depends on the
prompt it is given, so the whole pipeline is reproducible and free. Real
improvement work substitutes a live driver here and nothing else changes.

Run it
~~~~~~

::

    python examples/offline_improvement_example.py
"""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from prompture.drivers.base import Driver
from prompture.execution import (
    DEFAULT_GATES,
    JsonlOutcomeStore,
    ResourceLimits,
    Split,
    load_bundled_fixture_set,
    run_fixture_set,
)
from prompture.execution.improve import (
    CandidateSearchOptimizer,
    CandidateStore,
    GEPAOptimizerAdapter,
    PromotionError,
    PromotionLedger,
    Scorecard,
    candidate_from_failures,
    compare_optimizers,
)
from prompture.execution.strategies import DirectStrategy

FIXTURES = load_bundled_fixture_set("extraction_contacts")
GATE = DEFAULT_GATES["extraction_contacts"]

TARGET = "contact_extraction_instruction"
BASELINE_PROMPT = "Extract the contact details from the following text:"
GOOD_PROMPT = (
    "Extract the contact details from the following text. Use null for any field "
    "the text does not state. Never infer a company from a project name."
)
WEAK_PROMPT = (
    "Extract the contact details from the following text. Always fill in every "
    "field; make your best guess when the text is unclear."
)


class Contact(BaseModel):
    name: str | None = Field(None, description="Full name of the primary contact.")
    email: str | None = Field(None, description="Email address, or null when absent.")
    company: str | None = Field(None, description="Employer, or null when absent.")
    role: str | None = Field(None, description="Job title, or null when absent.")
    phone: str | None = Field(None, description="Phone number, or null when absent.")


# =============================================================================
# A scripted driver whose behaviour depends on the instruction it is given
# =============================================================================


class _ObedientDriver(Driver):
    """Answers from the fixture, but only abstains when told to.

    It exists so the example has a *reason* for one prompt to beat another: the
    prompt that forbids invention scores better on the cases with absent fields,
    exactly as a real model would. It is not a model, and the numbers below are
    a demonstration of the pipeline, not a measurement of anything.
    """

    model = "scripted/obedient"
    model_name = "scripted/obedient"
    supports_json_mode = False
    supports_json_schema = False

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        allowed_to_abstain = "Use null" in prompt
        answer = self._answer_for(prompt, allowed_to_abstain)
        return {
            "text": json.dumps(answer),
            "meta": {
                "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": 32,
                "total_tokens": max(1, len(prompt) // 4) + 32,
                "cost": 0.0002,
                "raw_response": {},
            },
        }

    @staticmethod
    def _answer_for(prompt: str, allowed_to_abstain: bool) -> dict[str, Any]:
        case = next((c for c in FIXTURES if str(c.inputs.get("text", "")) in prompt), None)
        if case is None:
            return {"name": None, "email": None, "company": None, "role": None, "phone": None}
        expected = dict(case.expected["fields"])
        if allowed_to_abstain:
            return expected
        # Without permission to say "unknown", invent something for every gap —
        # the failure mode the better prompt is meant to eliminate.
        return {k: (v if v is not None else f"guessed-{k}") for k, v in expected.items()}


# =============================================================================
# Evaluation plumbing
# =============================================================================


def evaluate_prompt(instruction: str, *, split: Split, label: str):
    """Run the whole split with *instruction* and return the workload summary."""
    driver = _ObedientDriver()
    report = run_fixture_set(
        FIXTURES,
        DirectStrategy(driver=driver, model="scripted/obedient", max_repairs=0, instruction=instruction),
        split=split,
        repeats=3,
        label=label,
        model="scripted/obedient",
        output_model=Contact,
        limits=ResourceLimits(max_llm_calls=2),
        request_overrides={"instruction": instruction},
    )
    return report.summaries[0]


def make_evaluator():
    """The ``(candidate, cases) -> score`` callable the optimizer needs."""

    def evaluate(candidate, cases):
        summary = evaluate_prompt(candidate.content, split=Split.DEV, label=candidate.id)
        return summary.mean_score or 0.0

    return evaluate


def main() -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(errors="replace")

    print("Prompture - offline improvement")
    workdir = Path(tempfile.mkdtemp(prefix="prompture-improve-"))
    print(f"  working directory: {workdir}")

    # --- 1. Baseline, and the failures that motivate a change ---------------
    print("\n" + "=" * 78)
    print("1. Baseline and its failures")
    print("=" * 78)

    outcomes = JsonlOutcomeStore(workdir / "outcomes.jsonl")
    baseline_dev = evaluate_prompt(BASELINE_PROMPT, split=Split.DEV, label="baseline")
    print(baseline_dev.format())

    run_fixture_set(
        FIXTURES,
        DirectStrategy(driver=_ObedientDriver(), model="scripted/obedient", max_repairs=0, instruction=BASELINE_PROMPT),
        split=Split.DEV,
        repeats=1,
        label="baseline",
        model="scripted/obedient",
        output_model=Contact,
        outcome_store=outcomes,
        request_overrides={"instruction": BASELINE_PROMPT},
    )
    failures = [r for r in outcomes.read() if r.correct is False]
    print(f"\n  {len(failures)} failing outcome record(s); the first few task ids:")
    for record in failures[:4]:
        wrong = record.metadata.get("score_details", {}).get("wrong_fields", [])
        print(f"    - {record.task_id}: wrong fields {wrong}")

    # --- 2. Candidates ------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. Two candidates, both inert until promoted")
    print("=" * 78)

    candidates = CandidateStore(workdir / "candidates.json")
    good = candidates.add(
        candidate_from_failures(
            TARGET,
            base_content=BASELINE_PROMPT,
            proposed_content=GOOD_PROMPT,
            failures=failures,
            rationale="the model invents values for fields the text never states",
        )
    )
    weak = candidates.add(
        candidate_from_failures(
            TARGET,
            base_content=BASELINE_PROMPT,
            proposed_content=WEAK_PROMPT,
            failures=failures,
            rationale="deliberately worse: it encourages guessing",
        )
    )
    for candidate in (good, weak):
        print(f"  {candidate.summary()}")
    print("\n  diff of the promising candidate:")
    for line in good.diff().splitlines()[2:]:
        print(f"    {line}")

    ledger = PromotionLedger(workdir / "promotions.json", min_gain=0.02)
    ledger.set_baseline(TARGET, BASELINE_PROMPT)
    print(f"\n  active version: {ledger.active(TARGET).version[:12]} (baseline retained)")

    # --- 3. Optimize on the development split only --------------------------
    print("\n" + "=" * 78)
    print("3. Candidate search on the development split")
    print("=" * 78)

    evaluate = make_evaluator()
    results = compare_optimizers(
        {
            "candidate_search": CandidateSearchOptimizer(),
            # Not installed in most environments; the comparison records that
            # honestly instead of silently substituting the simple search.
            "gepa": GEPAOptimizerAdapter(require=False),
        },
        [weak, good],
        FIXTURES.dev(),
        evaluate,
    )
    for label, result in results.items():
        print(f"  {label}:")
        for candidate, score in result.ranked:
            print(f"    {score:.3f}  {candidate.id}  {candidate.content[:56]}...")
        for note in result.notes:
            print(f"    note: {note[:110]}")

    winner = results["candidate_search"].best
    print(f"\n  winner on dev: {winner.id}")

    try:
        CandidateSearchOptimizer().optimize([good], FIXTURES.cases, evaluate)
    except ValueError as exc:
        print(f"  held-out guard: {exc}")

    # --- 4. Scorecard, dev and held-out kept apart --------------------------
    print("\n" + "=" * 78)
    print("4. Scorecard")
    print("=" * 78)

    card = Scorecard(
        candidate_id=winner.id,
        target=winner.target,
        candidate_version=winner.version,
        base_version=winner.base_version,
    )
    card.record_dev(
        baseline_dev,
        evaluate_prompt(winner.content, split=Split.DEV, label="candidate"),
        gate=GATE,
        provenance=FIXTURES.provenance(),
    )
    card.record_heldout(
        evaluate_prompt(BASELINE_PROMPT, split=Split.HELDOUT, label="baseline"),
        evaluate_prompt(winner.content, split=Split.HELDOUT, label="candidate"),
        gate=GATE,
        provenance=FIXTURES.provenance(),
    )
    print(card.format())

    # --- 5. Promotion and rollback ------------------------------------------
    print("\n" + "=" * 78)
    print("5. Promotion is explicit, and reversible")
    print("=" * 78)

    weak_card = Scorecard(candidate_id=weak.id, target=weak.target, candidate_version=weak.version)
    weak_card.record_dev(baseline_dev, evaluate_prompt(weak.content, split=Split.DEV, label="weak"), gate=GATE)
    weak_card.record_heldout(
        evaluate_prompt(BASELINE_PROMPT, split=Split.HELDOUT, label="baseline"),
        evaluate_prompt(weak.content, split=Split.HELDOUT, label="weak"),
        gate=GATE,
    )
    try:
        ledger.promote(weak, weak_card)
    except PromotionError as exc:
        print(f"  weak candidate refused: {exc}")
    ledger.reject(weak, weak_card)
    candidates.update(weak)

    ok, why = card.promotable(min_gain=0.02)
    print(f"  good candidate promotable: {ok} - {why}")
    if ok:
        ledger.promote(winner, card)
        print(f"  active version now: {ledger.active(TARGET).version[:12]} (from {winner.origin})")

    restored = ledger.rollback(TARGET, reason="demonstrating the rollback path")
    print(f"  rolled back to    : {restored.version[:12]}")
    print("\n  ledger history:")
    for record in ledger.records(TARGET):
        print(f"    {record.action:<9} -> {record.to_version[:12]}  {record.reason[:70]}")

    print(f"\n  artefacts written to {workdir}")
    print(
        "\nThe numbers above come from a scripted driver, so they demonstrate the\n"
        "pipeline rather than measure a model. Swap in a real driver to get\n"
        "measurements; the promotion rules do not change."
    )


if __name__ == "__main__":
    main()
