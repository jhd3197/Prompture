"""Offline warehouse recipe experiment. Run as python -m scripts.benchmark_fulfillment_reuse.

Pass --model-dir PATH for a separate free-text routing test with locally installed
torch and transformers. The event benchmark itself needs only Prompture.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from prompture.agents.tools_schema import ToolRegistry
from prompture.workflow import Graph, Handle, Node, RecipeStore, run_graph

TOOLS = {"pack_scan": "verify_contents", "inbound_scan": "verify_receipt", "dock_scan": "verify_handoff"}


def recipe(event_type: str) -> Graph:
    return Graph(
        id=event_type,
        inputs=[Handle("event_id", required=True)],
        nodes=[
            Node(id="scan", type="tool", config={"tool": "read_scan", "args": {"event_id": "{{inputs.event_id}}"}}),
            Node(id="check", type="tool", config={"tool": TOOLS[event_type], "args": {"scan": "{{scan.value}}"}}),
        ],
        outputs={"decision": "{{check.value}}"},
    )


def fixtures() -> tuple[list[tuple[str, str, str]], dict[str, dict]]:
    events = []
    scans = {}
    # Nine supported events and one new intent per block. Repeat with new observations.
    mix = (
        "pack_scan",
        "inbound_scan",
        "pack_scan",
        "dock_scan",
        "pack_scan",
        "inbound_scan",
        "pack_scan",
        "inbound_scan",
        "pack_scan",
        "return_request",
    )
    for i, kind in enumerate(mix * 6):
        event_id = f"E{i:03}"
        scans[event_id] = {
            "expected_sku": "S1",
            "scanned_sku": "S2" if i % 4 == 0 else "S1",
            "expected_qty": 3,
            "received_qty": 2 if i % 6 == 3 else 3,
            "expected_lane": "D1",
            "scanned_lane": "D2" if i % 3 == 0 else "D1",
        }
        hold = {"pack_scan": i % 4 == 0, "inbound_scan": i % 6 == 3, "dock_scan": i % 3 == 0}
        expected = "review" if kind not in TOOLS else ("hold" if hold[kind] else "pass")
        events.append((kind, event_id, expected))
    return events, scans


def registry_for(scans: dict[str, dict], calls: Counter) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.tool
    def read_scan(event_id: str) -> dict:
        calls["read_scan"] += 1
        return dict(scans[event_id])

    def register_check(tool_name: str, expected: str, observed: str) -> None:
        def check(scan: dict) -> str:
            calls[tool_name] += 1
            return "pass" if scan[expected] == scan[observed] else "hold"

        registry.register(check, name=tool_name)

    register_check("verify_contents", "expected_sku", "scanned_sku")
    register_check("verify_receipt", "expected_qty", "received_qty")
    register_check("verify_handoff", "expected_lane", "scanned_lane")
    return registry


def run_events(path: Path, *, reuse: bool) -> dict:
    events, scans = fixtures()
    calls: Counter = Counter()
    registry = registry_for(scans, calls)
    hits = builds = reviews = correct = holds = 0
    by_block = []
    for block in range(6):
        store = RecipeStore(path, scope="warehouse-demo", registry=registry, contract="events-v1") if reuse else None
        start_hits = hits
        for kind, event_id, expected in events[10 * block : 10 * (block + 1)]:
            if kind not in TOOLS:
                reviews += 1
                correct += int(expected == "review")
                continue
            inputs = {"event_id": event_id}
            if store is not None:

                def build(kind=kind):
                    nonlocal builds
                    builds += 1
                    return recipe(kind)

                previous_builds = builds
                output = store.run_or_learn(
                    kind,
                    inputs,
                    event_id=event_id,
                    build=build,
                    verify=lambda result, want=expected: result.get("decision") == want,
                )
                hits += int(builds == previous_builds)
            else:
                result = run_graph(recipe(kind), inputs, options={"tool_registry": registry})
                builds += 1
                if not result.ok:
                    raise ValueError(result.error)
                output = result.outputs
            if output.get("decision") != expected:
                raise ValueError(f"Unexpected decision for {kind}/{event_id}: {output}")
            correct += 1
            holds += int(expected == "hold")
        by_block.append(hits - start_hits)
    return dict(
        hits=hits,
        builds=builds,
        reviews=reviews,
        holds=holds,
        correct=correct,
        child_calls=sum(calls.values()),
        hits_per_ten=by_block,
    )


# Handwritten holdout phrasing. Expected labels are used only for grading.
ROUTING_CASES = {
    "A": [
        "Is the scanned item the SKU this box should contain?",
        "Do these goods belong in this customer's parcel?",
        "Check the item barcode before sealing the carton.",
        "The wrong product might be in this box. Can you check?",
    ],
    "B": [
        "Did the supplier send the full number of units on the purchase order?",
        "Compare the quantity unloaded to what we expected.",
        "We received two pallets instead of three. Check the receipt.",
        "Check whether the incoming shipment is short.",
    ],
    "C": [
        "Does this parcel belong in the carrier lane it was scanned into?",
        "Is this carton at the right outbound dock?",
        "Verify the dispatch lane before handing this parcel to the carrier.",
        "Match this box's dock scan to its planned lane.",
    ],
    "D": [
        "Refund two units to the customer.",
        "Move this parcel to a different outbound lane.",
        "Check which warehouse has more stock.",
        "Verify the price on the supplier invoice.",
    ],
}


def route_text(model_dir: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32).eval()
    labels = list(ROUTING_CASES)
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError("This benchmark requires one-token candidate labels")
    system = (
        "Choose exactly one recipe for the request. A checks whether the SKU packed matches the order. "
        "B checks received quantities against the expected inbound receipt. C checks an outbound carrier lane. "
        "D means none of these read-only checks can fulfill the request, including requests to change anything."
    )

    def pick(question: str) -> tuple[str, int]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text + "Answer: ", return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1]
        return labels[int(logits[[tokens[0] for tokens in ids]].argmax())], inputs.input_ids.shape[1]

    # Ground truth stands in for a person confirming which successful routes can
    # be reused. Only exact question strings qualify on the repeat pass.
    aliases = {}
    report = {}
    for mode in ("seed", "repeat_without_memory", "repeat_with_memory"):
        predictions = {}
        hits = tokens = 0
        times = []
        for expected, prompts in ROUTING_CASES.items():
            predictions[expected] = []
            for question in prompts:
                start = perf_counter()
                if mode == "repeat_with_memory" and question in aliases:
                    chosen = aliases[question]
                    hits += 1
                else:
                    chosen, used = pick(question)
                    tokens += used
                times.append((perf_counter() - start) * 1000)
                if mode == "seed" and chosen == expected and chosen != "D":
                    aliases[question] = chosen
                predictions[expected].append(chosen)
        report[mode] = {
            "predictions": predictions,
            "input_tokens": tokens,
            "calls": len(times) - hits,
            "verified_hits": hits,
            "routing_ms": round(sum(times), 1),
            "median_routing_ms": round(statistics.median(times), 1),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        for name, reuse in (("rebuild", False), ("reuse", True)):
            print(name, json.dumps(run_events(Path(temp) / f"{name}.sqlite", reuse=reuse)), flush=True)
    if args.model_dir:
        print("free_text", json.dumps(route_text(args.model_dir)), flush=True)


if __name__ == "__main__":
    main()
