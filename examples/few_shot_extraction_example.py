"""Few-Shot Extraction Example

Demonstrates :class:`prompture.extraction.few_shot.FewShotExampleStore`
plumbed into :func:`prompture.extract_with_model` via the new
``examples_store`` parameter.

The example builds a small bank of curated ``(input, expected_output)``
pairs that encode subtle extraction conventions the model would
otherwise have to guess at — date formats, US state abbreviations,
phone normalisation. We then run extraction with and without the
example bank against a contrived test set and tally how many records
the model gets right under each setting.

Run it
~~~~~~

Set ``OLLAMA_ENDPOINT`` (default ``http://localhost:11434``) and pull a
small instruction-tuned model::

    ollama pull llama3.1:8b
    OLLAMA_MODEL=llama3.1:8b python examples/few_shot_extraction_example.py

For OpenAI/Claude instead, set the corresponding env var and edit
``MODEL`` below. The example skips gracefully when no LLM is reachable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from prompture import extract_with_model
from prompture.exceptions import ConfigurationError
from prompture.extraction.few_shot import FewShotExampleStore

# =============================================================================
# Schema — narrow enough that the conventions in our examples actually matter
# =============================================================================


class ContactRecord(BaseModel):
    """Address-book row with strict normalisation rules."""

    full_name: str = Field(..., description="Full legal name in 'First Last' order")
    state: str = Field(..., description="US state as 2-letter uppercase abbreviation")
    phone_e164: str = Field(..., description="Phone number in E.164 format with + and country code")
    join_date: str = Field(..., description="Date the contact joined, as YYYY-MM-DD")


# =============================================================================
# Few-shot bank — encodes the conventions a free-style LLM tends to violate
# =============================================================================

CURATED_EXAMPLES: list[tuple[str, dict[str, str]]] = [
    (
        "Robert Schwartz, lives in Texas, phone (512) 555-0177, joined March 4th 2021.",
        {
            "full_name": "Robert Schwartz",
            "state": "TX",
            "phone_e164": "+15125550177",
            "join_date": "2021-03-04",
        },
    ),
    (
        "Aisha Khan from California, number 415.555.0123, started Jan 9, 2020.",
        {
            "full_name": "Aisha Khan",
            "state": "CA",
            "phone_e164": "+14155550123",
            "join_date": "2020-01-09",
        },
    ),
    (
        "Yuki Tanaka — New York — cell 1-212-555-0144 — onboarded 12/15/2022.",
        {
            "full_name": "Yuki Tanaka",
            "state": "NY",
            "phone_e164": "+12125550144",
            "join_date": "2022-12-15",
        },
    ),
    (
        "Diego Hernández (Florida) at +1 305 555 0190; came aboard 7-22-2019.",
        {
            "full_name": "Diego Hernández",
            "state": "FL",
            "phone_e164": "+13055550190",
            "join_date": "2019-07-22",
        },
    ),
]

# Test set — same format / convention diversity, different people.
TEST_CASES: list[tuple[str, dict[str, str]]] = [
    (
        "Maria Lopez out of Illinois, phone (773) 555-0111, joined April 5 2023.",
        {
            "full_name": "Maria Lopez",
            "state": "IL",
            "phone_e164": "+17735550111",
            "join_date": "2023-04-05",
        },
    ),
    (
        "Liam O'Brien from Massachusetts, 617-555-0166, started 02/14/2020.",
        {
            "full_name": "Liam O'Brien",
            "state": "MA",
            "phone_e164": "+16175550166",
            "join_date": "2020-02-14",
        },
    ),
    (
        "Priya Patel — Washington — 1.206.555.0188 — onboarded November 3rd 2021.",
        {
            "full_name": "Priya Patel",
            "state": "WA",
            "phone_e164": "+12065550188",
            "join_date": "2021-11-03",
        },
    ),
]


# =============================================================================
# Model — Ollama by default; can be swapped via env
# =============================================================================

MODEL = os.getenv("PROMPTURE_FEWSHOT_MODEL", "ollama/" + os.getenv("OLLAMA_MODEL", "llama3.1:8b"))


def _try_extract(text: str, examples_store: FewShotExampleStore | None) -> ContactRecord | None:
    try:
        kwargs: dict = {"model_name": MODEL}
        if examples_store is not None:
            kwargs["examples_store"] = examples_store
        result = extract_with_model(ContactRecord, text, **kwargs)
        return result["model"]
    except (ConfigurationError, ConnectionError) as e:
        print(f"    [extraction failed: {type(e).__name__}: {str(e)[:120]}]")
        return None
    except Exception as e:
        print(f"    [extraction failed: {type(e).__name__}: {str(e)[:120]}]")
        return None


def _score(pred: ContactRecord | None, gold: dict[str, str]) -> int:
    if pred is None:
        return 0
    return sum(1 for k, v in gold.items() if getattr(pred, k, None) == v)


def main() -> None:
    print("=" * 70)
    print(f"Few-Shot extraction comparison — model = {MODEL}")
    print("=" * 70)

    store = FewShotExampleStore()
    store.add_many(CURATED_EXAMPLES)
    print(f"Loaded {len(store)} curated examples into the few-shot store.\n")

    field_count = len(TEST_CASES[0][1])
    total = field_count * len(TEST_CASES)

    without_correct = 0
    with_correct = 0

    for i, (text, gold) in enumerate(TEST_CASES, 1):
        print(f"[case {i}] {text}")
        print(f"  expected: {json.dumps(gold)}")

        print("  -> without few-shot:")
        pred_no = _try_extract(text, examples_store=None)
        s_no = _score(pred_no, gold)
        without_correct += s_no
        if pred_no is not None:
            print(f"    got: {pred_no.model_dump_json()}")
            print(f"    fields correct: {s_no}/{field_count}")

        print("  -> with few-shot:")
        pred_fs = _try_extract(text, examples_store=store)
        s_fs = _score(pred_fs, gold)
        with_correct += s_fs
        if pred_fs is not None:
            print(f"    got: {pred_fs.model_dump_json()}")
            print(f"    fields correct: {s_fs}/{field_count}")
        print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  without few-shot: {without_correct}/{total} fields correct "
          f"({without_correct / total:.0%})")
    print(f"  with few-shot:    {with_correct}/{total} fields correct "
          f"({with_correct / total:.0%})")
    if with_correct > without_correct:
        delta = with_correct - without_correct
        print(f"  -> +{delta} fields correct from few-shot examples "
              f"({delta / total:.0%} absolute improvement).")
    elif with_correct == without_correct:
        print("  -> no measurable improvement on this test set (model may already be solid).")
    else:
        print("  -> few-shot regressed accuracy - examples may be confusing this model.")


if __name__ == "__main__":
    main()
