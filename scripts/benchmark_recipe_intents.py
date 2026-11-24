"""Measure a 13-intent decision menu against the same published warehouse requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from prompture.drivers import get_decision_driver_for_model, get_driver_for_model
from scripts.benchmark_recipe_discovery import catalog, summarize

# Six read intents map to existing recipes. All seven other intents leave the store idle.
DECLINES = {
    "0": "Other request outside these lookups.",
    "7": "Change or write warehouse data, inventory or a physical item.",
    "8": "Multiple distinct jobs or lookups are required.",
    "9": "Unclear which warehouse question the user wants answered.",
    "10": "Predict arrival time or future delivery performance.",
    "11": "Inspect facilities, environment, staffing or equipment.",
    "12": "Evaluate prices, contracts, finances or business performance.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", action="store_true", help="Use a text driver with the same labeled options")
    parser.add_argument(
        "--baseline", type=Path, nargs="+", required=True, help="Full-menu reports used for request and recipe order"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    driver = get_driver_for_model(args.model) if args.text else get_decision_driver_for_model(args.model)
    rows = []
    fixtures = []
    with TemporaryDirectory() as tmp:
        store = catalog(Path(tmp) / "warehouse.sqlite")
        for path in args.baseline:
            baseline = json.loads(path.read_text(encoding="utf-8"))
            fixtures.append(baseline["fixture_sha256"])
            for record in baseline["arms"]["6"]["rows"]:
                match = store.discover(
                    record["request"],
                    model=driver,
                    limit=6,
                    decline_reasons=tuple(DECLINES.values()),
                    options={"temperature": 0, "max_tokens": 16} if args.text else {},
                )
                assert list(match.candidates) == record["candidates"]
                if match.recipe:
                    predicted = match.recipe.name
                    choice = str(match.candidates.index(predicted) + 1)
                else:
                    predicted = "abstain" if match.status == "abstained" else "invalid"
                    choice = next((key for key, reason in DECLINES.items() if reason == match.decline_reason), None)
                model_name = args.model if args.text else driver.last_usage["raw_response"].get("model", args.model)
                rows.append(
                    {
                        "request": record["request"],
                        "expected": record["expected"],
                        "choice": choice,
                        "predicted": predicted,
                        "candidates": match.candidates,
                        "status": match.status,
                        "served_model": model_name,
                        "input_tokens": match.usage["prompt_tokens"],
                        "output_tokens": match.usage["completion_tokens"],
                    }
                )
    summary = summarize(rows)
    args.out.write_text(
        json.dumps({"model": args.model, "fixtures": fixtures, "summary": summary, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.model, summary)


if __name__ == "__main__":
    main()
