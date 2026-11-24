"""Replay measured Laya proposals and ask a second driver to confirm each proposed recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from prompture.workflow.discovery import select_recipe
from scripts.benchmark_recipe_discovery import catalog, summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposals", required=True, type=Path, help="Measured Laya decisions on the confirmation holdout"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    from prompture.drivers import get_driver_for_model

    driver = get_driver_for_model(args.model)
    source = json.loads(args.proposals.read_text(encoding="utf-8"))
    rows = []
    with TemporaryDirectory() as tmp:
        store = catalog(Path(tmp) / "recipes.sqlite")
        candidates = {c.name: c for c in store.search("", limit=6)}
        for proposal in source["arms"]["examples"]["rows"]:
            predicted = proposal["predicted"]
            status = proposal["status"]
            reviewed = []
            verification_usage = {}
            if status == "selected":
                result = select_recipe(
                    store.scope,
                    proposal["request"],
                    [candidates[predicted]],
                    driver,
                    {"temperature": 0, "max_tokens": 16},
                    confirm_first=True,
                )
                status = result.status
                predicted = result.recipe.name if result.recipe else ("abstain" if status == "abstained" else "invalid")
                verification_usage = result.usage
                reviewed = list(result.candidates)
            rows.append(
                {
                    "request": proposal["request"],
                    "expected": proposal["expected"],
                    "proposed": proposal["predicted"],
                    "predicted": predicted,
                    "status": status,
                    "candidates": reviewed,
                    "review_input": verification_usage.get("prompt_tokens", 0),
                    "review_output": verification_usage.get("completion_tokens", 0),
                    "input_tokens": proposal["input_tokens"] + verification_usage.get("prompt_tokens", 0),
                    "output_tokens": proposal["output_tokens"] + verification_usage.get("completion_tokens", 0),
                }
            )
    summary = summarize(rows)
    report = {
        "proposer": source["model"],
        "reviewer": args.model,
        "fixture_sha256": source["fixture_sha256"],
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.model, summary)


if __name__ == "__main__":
    main()
