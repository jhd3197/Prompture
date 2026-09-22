"""Decision Model Example — typed decisions instead of generated text

Demonstrates the ``decision`` modality: a *state* plus a map of typed
*questions* in, one typed *answer* per question out. No prose, nothing to
parse, nothing to hallucinate.

Three question primitives:

* ``Noul``   — yes/no, returns a calibrated probability of yes
* ``Choice`` — pick one option, returns the winner + full distribution
* ``Score``  — rate against an ordinal rubric, returns a weighted value

Every provider below speaks the same contract, so the only thing that changes
between them is the ``"provider/model"`` string:

* ``typesafe/jev-latest`` — hosted, ``TYPESAFE_API_KEY`` (key from
  https://console.typesafe.ai/). Billed on input tokens; output is free.
* ``kev/kev-4b``          — self-hosted, serves the same ``/v1/systemone``
  endpoint. Point ``KEV_BASE_URL`` at it; no key needed.
* ``laya/router``         — in-process open weights.
  Install with ``pip install prompture[laya]``; no server, no key.

Sections below:

1. Ask several questions in one call
2. Confidence-gated routing — the pattern that pays for itself
3. Swapping providers without changing the questions
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.drivers.decision_base import Choice, Noul, Score
from prompture.drivers.decision_registry import get_decision_driver_for_model

# =============================================================================
# The state and the questions — shared by every section
# =============================================================================

TICKET = {
    "from": "ops@acme.example",
    "subject": "Duplicate charge on invoice #4411",
    "body": (
        "Hi, we were billed twice for March. Please refund the duplicate today or we will have to cancel our plan."
    ),
}

QUESTIONS = {
    "department": Choice(
        "Which team should handle this request?",
        criteria={
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, upgrades, new contracts",
            "other": "everything else",
        },
    ),
    "urgency": Score(
        "How urgent is this request?",
        criteria=["not urgent", "this week", "blocking or deadline-driven"],
    ),
    "churn_risk": Noul("Does the sender threaten to cancel or leave?"),
    "refund_requested": Noul("Does the sender explicitly ask for a refund?"),
}


def _print_answers(result) -> None:
    """Print each answer with the metadata that makes it actionable."""
    for qid, answer in result.answers.items():
        if answer.type == "choice":
            runner_up = answer.ranked()[1] if len(answer.ranked()) > 1 else ("—", 0.0)
            print(f"  {qid:18} {answer.choice:<12} confidence {answer.confidence:.2f}")
            print(f"  {'':18} runner-up: {runner_up[0]} ({runner_up[1]:.2f})")
        elif answer.type == "score":
            print(f"  {qid:18} {answer.score:.2f}  ({answer.nearest_level})  confidence {answer.confidence:.2f}")
        else:
            print(f"  {qid:18} {answer.noul:.2f} probability")


# =============================================================================
# 1. Several questions, one call
# =============================================================================


def example_batched_questions(model: str) -> None:
    print("\n" + "=" * 70)
    print("1. Four questions, one call")
    print("=" * 70)

    driver = get_decision_driver_for_model(model)
    result = driver.decide(TICKET, QUESTIONS)

    print(f"\nModel that answered: {result.model}")
    _print_answers(result)

    usage = driver.last_usage
    print(f"\n  questions: {usage['questions']}   input tokens: {usage['input_tokens']}")
    print(f"  cost: ${usage['cost']:.6f}" + ("  (pricing unknown)" if usage["pricing_unknown"] else ""))
    print("\n  All four questions shared one reading of the ticket — batching")
    print("  beats looping, and the questions cannot see each other's answers.")


# =============================================================================
# 2. Confidence-gated routing
# =============================================================================


def example_confidence_gate(model: str) -> None:
    print("\n" + "=" * 70)
    print("2. Confidence-gated routing")
    print("=" * 70)

    driver = get_decision_driver_for_model(model)
    result = driver.decide(TICKET, {"department": QUESTIONS["department"]})
    answer = result["department"]

    threshold = 0.85
    print(f"\n  routed to : {answer.choice}")
    print(f"  confidence: {answer.confidence:.2f}  (threshold {threshold})")

    if answer.confidence >= threshold:
        print("  -> act automatically; no LLM call needed")
    else:
        print("  -> escalate: hand the ticket to a full LLM (or a human)")

    print("\n  This is the pattern that pays for itself: a decision costs a")
    print("  fraction of a cent, so it decides whether the expensive model")
    print("  runs at all. The probabilities are calibrated, so the threshold")
    print("  means something.")


# =============================================================================
# 3. Same questions, different provider
# =============================================================================


def example_provider_swap() -> None:
    print("\n" + "=" * 70)
    print("3. Same questions, different provider")
    print("=" * 70)

    print("\n  The question map above is provider-agnostic. To move a workload")
    print("  off the hosted API, change one string:\n")
    for model, note in (
        ("typesafe/jev-latest", "hosted, $0.042/Mtok input, output free"),
        ("kev/kev-4b", "self-hosted server, same /v1/systemone contract"),
        ("laya/router", "in-process weights, no server at all"),
    ):
        call = f"get_decision_driver_for_model({model!r})"
        print(f"    {call:<52} # {note}")

    print("\n  Worth knowing before you swap: the open checkpoints trade")
    print("  accuracy for cost, and both degrade on high-cardinality choices")
    print("  (50+ options). Benchmark on your own labels before switching.")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("\n" + "=" * 70)
    print("DECISION MODEL EXAMPLE")
    print("=" * 70)

    # Pick whichever provider is actually reachable.
    if os.getenv("TYPESAFE_API_KEY"):
        model = "typesafe/jev-latest"
    elif os.getenv("KEV_BASE_URL"):
        model = "kev/kev-latest"
    else:
        print("\nNo decision provider configured. Set one of:")
        print("  TYPESAFE_API_KEY  — hosted Jev (https://console.typesafe.ai/)")
        print("  KEV_BASE_URL      — a local Kev server")
        print("\nShowing the provider-swap section only.\n")
        example_provider_swap()
        return

    print(f"\nUsing: {model}")
    example_batched_questions(model)
    example_confidence_gate(model)
    example_provider_swap()

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
