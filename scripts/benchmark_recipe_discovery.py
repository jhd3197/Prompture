"""Evaluate unseen requests against the real saved-recipe discovery API, one model per invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from prompture.agents.tools_schema import ToolRegistry
from prompture.workflow import Graph, Handle, HandleType, Node, RecipeExampleIndex, RecipeStore

CASES = Path(__file__).with_name("data") / "recipe_discovery_cases.json"
# Host-authored catalog descriptions and confirmed aliases. No evaluation prompt is an alias.
CATALOG = {
    "carton_contents": ("Compare packed items and their quantities to the shipment lines.", "verify pack list"),
    "inbound_count": (
        "Compare supplier delivery quantities received against the expected purchase order.",
        "check receipt",
    ),
    "dispatch_lane": (
        "Verify the outbound parcel sort lane against its assigned carrier dock.",
        "verify outbound lane",
    ),
    "free_stock": ("Look up available inventory after reservations at one facility.", "check unreserved stock"),
    "custody_history": ("Find the last accepted shipment handoff and its custody trail.", "look up tracking history"),
    "label_revision": (
        "Compare the printed shipping label version to the current issued revision.",
        "check label version",
    ),
}


def catalog(path: Path) -> RecipeStore:
    store = RecipeStore(path, scope="routing-eval", registry=ToolRegistry(), contract="snapshot-v1")
    for name, (description, alias) in CATALOG.items():
        # Each graph reads a different field of a supplied warehouse verification snapshot.
        graph = Graph(
            id=name,
            name=description,
            inputs=[Handle("snapshot", HandleType.JSON)],
            nodes=[Node("read", "transform", {"op": "get", "value": "{{inputs.snapshot}}", "path": [name]})],
            outputs={"result": "{{read.value}}"},
        )
        store.publish(graph, aliases=[alias])
    return store


def summarize(rows: list[dict]) -> dict:
    supported = [r for r in rows if r["expected"] != "abstain"]
    negatives = len(rows) - len(supported)
    return {
        "requests": len(rows),
        "supported": len(supported),
        "should_abstain": negatives,
        "correct": sum(r["predicted"] == r["expected"] for r in rows),
        "correct_recipe": sum(r["predicted"] == r["expected"] for r in supported),
        "correct_abstentions": sum(r["expected"] == r["predicted"] == "abstain" for r in rows),
        "false_accepts": sum(r["expected"] == "abstain" and r["status"] == "selected" for r in rows),
        "invalid": sum(r["status"] == "invalid" for r in rows),
        "retrieved_gold": sum(r["expected"] in r["candidates"] for r in supported),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Prompture provider/model string")
    parser.add_argument("--ollama-endpoint", help="Override the endpoint for a locally installed Ollama model")
    parser.add_argument("--out", required=True, type=Path)
    fixture = parser.add_mutually_exclusive_group()
    fixture.add_argument(
        "--shift-holdout", action="store_true", help="Use new requests and compare four candidate menus"
    )
    fixture.add_argument(
        "--confirmation-holdout", action="store_true", help="Evaluate confirmation on the second fresh set"
    )
    parser.add_argument(
        "--decision", action="store_true", help="Use a typed decision driver, such as laya/typed-decisions"
    )
    args = parser.parse_args()
    if args.decision:
        from prompture.drivers.decision_registry import get_decision_driver_for_model

        driver = get_decision_driver_for_model(args.model)
    elif args.ollama_endpoint:
        from prompture.drivers.ollama_driver import OllamaDriver

        if not args.model.startswith("ollama/"):
            parser.error("--ollama-endpoint requires an ollama/ model")
        driver = OllamaDriver(endpoint=args.ollama_endpoint, model=args.model.split("/", 1)[1])
    else:
        from prompture.drivers import get_driver_for_model

        driver = get_driver_for_model(args.model)
    data_file = CASES.with_name("warehouse_shift_holdout.json") if args.shift_holdout else CASES
    if args.confirmation_holdout:
        data_file = CASES.with_name("warehouse_confirmation_holdout.json")
    raw = data_file.read_text(encoding="utf-8")
    cases = [(request, label) for label, requests in json.loads(raw).items() for request in requests]
    assert set(label for _, label in cases) == {*CATALOG, "abstain"}
    assert not {q for q, _ in cases} & ({*CATALOG} | {alias for _, alias in CATALOG.values()})
    menus = [("6", 6, None), ("3", 3, None)]
    if args.shift_holdout or args.confirmation_holdout:
        old = json.loads(CASES.read_text(encoding="utf-8"))
        assert not {q.casefold() for q, _ in cases} & {q.casefold() for phrases in old.values() for q in phrases}
        examples = {name: old[name] for name in CATALOG}
        categories = {
            "inbound_count": "inventory",
            "free_stock": "inventory",
            "carton_contents": "packaging",
            "label_revision": "packaging",
            "dispatch_lane": "transport",
            "custody_history": "transport",
        }
        menus.extend(
            [
                ("examples", 3, RecipeExampleIndex(examples)),
                ("categories", 3, RecipeExampleIndex(examples, categories=categories)),
            ]
        )
        if args.confirmation_holdout:
            first_shift = json.loads(CASES.with_name("warehouse_shift_holdout.json").read_text(encoding="utf-8"))
            assert not {q.casefold() for q, _ in cases} & {
                q.casefold() for phrases in first_shift.values() for q in phrases
            }
            menus = [menus[0], menus[2], ("confirm", 1, RecipeExampleIndex(examples))]
    random.Random(131).shuffle(cases)
    report = {
        "model": args.model,
        "fixture_sha256": hashlib.sha256(raw.replace("\r\n", "\n").encode()).hexdigest(),
        "arms": {},
    }
    with TemporaryDirectory() as tmp:
        store = catalog(Path(tmp) / "recipes.sqlite")
        for arm, limit, index in menus:
            rows = []
            for request, expected in cases:
                start = perf_counter()
                options = {} if args.decision else {"temperature": 0, "max_tokens": 16}
                match = store.discover(
                    request, model=driver, limit=limit, index=index, options=options, confirm_first=arm == "confirm"
                )
                predicted = (
                    match.recipe.name if match.recipe else ("abstain" if match.status == "abstained" else "invalid")
                )
                rows.append(
                    {
                        "request": request,
                        "expected": expected,
                        "predicted": predicted,
                        "status": match.status,
                        "candidates": match.candidates,
                        "input_tokens": match.usage["prompt_tokens"],
                        "output_tokens": match.usage["completion_tokens"],
                        "seconds": round(perf_counter() - start, 3),
                    }
                )
                if args.decision:
                    metadata = driver.last_usage
                    rows[-1]["served_model"] = (metadata.get("raw_response") or {}).get("model", args.model)
                    if not metadata.get("pricing_unknown", True):
                        rows[-1]["estimated_cost_usd"] = metadata["cost"]
            summary = summarize(rows)
            report["arms"][arm] = {"summary": summary, "rows": rows}
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(args.model, arm, summary, flush=True)


if __name__ == "__main__":
    main()
