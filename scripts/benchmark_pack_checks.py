"""Matched fulfillment benchmark. Default is offline. --live sends synthetic fixtures to Anthropic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import requests

from prompture.agents.tools_schema import ToolRegistry
from prompture.workflow import RecipeStore
from scripts.pack_checks import POLICY, audit, fixture, pack_graph, tools_for

ARMS = ("direct", "ptc", "recipe", "ptc-recipe")
EXECUTOR = "code_execution_20260120"
PROMPT = (
    "Check all cartons in batch B1 using the available tools. "
    + POLICY
    + " Use fresh observations. Respond with just a JSON object mapping shipment IDs to the two strings."
)


def grade(answer: object, expected: dict) -> dict:
    answer = answer if isinstance(answer, dict) else {}
    return {
        "exact": answer == expected,
        "correct": sum(answer.get(key) == value for key, value in expected.items()),
        "total": len(expected),
        "false_releases": sum(
            isinstance(answer.get(k), list) and answer[k][:1] == ["pass"]
            for k, value in expected.items()
            if value[0] != "pass"
        ),
    }


def parse_answer(text: str) -> dict:
    """Accept one unambiguous JSON object, including one surrounded by narration or a code fence."""
    decoder = json.JSONDecoder()
    objects = []
    pos = text.find("{")
    while pos >= 0:
        try:
            value, end = decoder.raw_decode(text, pos)
        except ValueError:
            pos = text.find("{", pos + 1)
            continue
        objects.append(value)
        pos = text.find("{", end)
    if len(objects) != 1:
        raise ValueError(f"Expected one JSON decision object, found {len(objects)}")
    return objects[0]


def learn_and_replay(path: Path, model: str) -> dict:
    """Measure the one-time model planning cost separately from subsequent typed events."""
    from prompture.workflow import architect

    manifests, scans, expected = fixture()
    reads = Counter()
    store = RecipeStore(path, scope="learn", registry=tools_for(manifests, scans, reads), contract="packing-v1")
    usage = {}
    planning_calls = 0

    def build():
        nonlocal planning_calls
        planning_calls += 1
        result = architect(
            "Build a reusable read-only graph with one string input batch. Tools: read_manifests(batch), "
            "read_scans(batch), audit(manifests, scans). Pass the actual dictionaries from the two reads to audit. "
            "Use the graph input, never a literal batch. Return the audit result under the graph output decisions. "
            "Each tool node has one result handle named result. Use only these three tool nodes.",
            model=f"claude/{model}",
            graph_id="audit_pack_batch",
            max_retries=0,
        )
        usage.update(
            {key: result.usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")}
        )
        if not result.ok:
            raise RuntimeError(result.error or "No valid graph")
        return result.graph

    runs = []
    for phase in ("cold", "replay_changed_scan"):
        if phase == "replay_changed_scan":
            scans["S11"]["carton"] = "changed"
            expected["S11"] = ["hold", "carton"]
        reads.clear()
        start = perf_counter()
        before = planning_calls
        output = store.run_or_learn(
            "audit_pack_batch",
            {"batch": "B1"},
            build=build,
            verify=lambda out: out.get("decisions") == expected and all(reads[k] for k in ("manifests", "scans")),
            event_id=phase,
        )
        runs.append(
            {
                "phase": phase,
                "seconds": round(perf_counter() - start, 4),
                "planner_invocations": planning_calls - before,
                "reads": dict(reads),
                **grade(output["decisions"], expected),
            }
        )
    return {"model": model, "planning_usage": usage, "runs": runs}


def claude_run(arm: str, model: str, api_key: str, registry: ToolRegistry, metrics: dict, max_tokens: int) -> object:
    hosted = arm in {"ptc", "ptc-recipe"}
    tools = [tool.to_anthropic_format() for tool in registry.definitions]
    if hosted:
        for tool in tools:
            tool["allowed_callers"] = [EXECUTOR]
        tools.append({"type": EXECUTOR, "name": "code_execution"})
    messages = [{"role": "user", "content": PROMPT}]
    body = {"model": model, "max_tokens": max_tokens, "tools": tools, "messages": messages}
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    for _ in range(8):
        metrics["requests"] += 1
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=(15, 120),
            allow_redirects=False,
        )
        if response.status_code != 200:
            error = response.json().get("error", {})
            raise RuntimeError(
                f"Anthropic HTTP {response.status_code}: {error.get('type')}: {error.get('message', '')}".replace(
                    api_key, "[redacted]"
                )
            )
        data = response.json()
        usage = data["usage"]
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            metrics[key] += usage.get(key, 0) or 0
        metrics["usage_by_request"].append(usage)
        content = data["content"]
        messages.append({"role": "assistant", "content": content})
        if data.get("container"):
            body["container"] = data["container"]["id"]
        results = []
        for block in content:
            if block["type"] != "tool_use":
                continue
            caller = block.get("caller", {}).get("type", "direct")
            if hosted and not caller.startswith("code_execution_"):
                raise RuntimeError("Expected a programmatic caller")
            metrics["hosted_callbacks"] += int(caller.startswith("code_execution_"))
            value = registry.execute(block["name"], block["input"], raise_on_invalid=True)
            results.append({"type": "tool_result", "tool_use_id": block["id"], "content": json.dumps(value)})
        if results:
            messages.append({"role": "user", "content": results})
        elif data["stop_reason"] == "pause_turn":
            continue
        elif data["stop_reason"] == "end_turn":
            text = "".join(block["text"] for block in content if block["type"] == "text").strip()
            metrics["final_text"] = text
            return parse_answer(text)
        else:
            raise RuntimeError(f"Unexpected stop reason: {data['stop_reason']}")
    raise RuntimeError("Exceeded eight Messages requests")


def compare(
    path: Path, *, live: bool, model: str, arms: list[str], max_tokens: int = 6144, extra_lines: int = 0
) -> dict:
    manifests, scans, expected = fixture(extra_lines)
    reads: Counter = Counter()
    tools = tools_for(manifests, scans, reads)
    store = RecipeStore(path, scope="warehouse-demo", registry=tools, contract="packing-v1")
    fingerprint = store.publish(pack_graph(), aliases=["pack_scan"])

    # Validate the receipt contract at dispatch. Only the external grader knows the expected decisions.
    def complete(output: dict) -> bool:
        decisions = output.get("decisions")
        return isinstance(decisions, dict) and decisions.keys() == manifests.keys()

    recipe_tool = store.as_tool("pack_scan", verify=complete)
    report = {
        "model": model,
        "max_tokens_per_request": max_tokens,
        "extra_lines_per_shipment": extra_lines,
        "recipe": fingerprint,
        "arms": {},
    }
    headers_key = None
    if live:
        from prompture.drivers.claude_driver import env_api_key

        headers_key = env_api_key()
        if not headers_key:
            raise RuntimeError("Configure CLAUDE_API_KEY or ANTHROPIC_API_KEY for --live")
    for arm in ["python", "local-recipe", *(arms if live else [])]:
        exposed = ToolRegistry()
        if arm in {"recipe", "ptc-recipe"}:
            exposed.add(recipe_tool)
        else:
            for name in ("read_manifests", "read_scans"):
                exposed.add(tools.get(name))
        reads.clear()
        metrics = dict(
            requests=0,
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            hosted_callbacks=0,
            usage_by_request=[],
        )
        start = perf_counter()
        try:
            if arm == "python":
                answer = audit(*(tools.execute(n, {"batch": "B1"}) for n in ("read_manifests", "read_scans")))
            elif arm == "local-recipe":
                answer = store.run("pack_scan", {"batch": "B1"}, verify=complete)["decisions"]
            else:
                answer = claude_run(arm, model, headers_key, exposed, metrics, max_tokens)
            metrics.update(grade(answer, expected))
            metrics["qualified"] = metrics["exact"] and bool(reads["scans"] and reads["manifests"])
        except (requests.RequestException, ValueError, RuntimeError, KeyError) as exc:
            metrics.update(grade(None, expected), qualified=False, error=str(exc))
        metrics.update(seconds=round(perf_counter() - start, 4), reads=dict(reads))
        report["arms"][arm] = metrics
        print(
            arm,
            json.dumps({k: v for k, v in metrics.items() if k not in {"usage_by_request", "final_text"}}),
            flush=True,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-tokens", type=int, default=6144)
    parser.add_argument("--extra-lines", type=int, default=0, help="Additional matched SKU lines per shipment")
    parser.add_argument("--learn-only", action="store_true", help="Use the live Prompture architect once, then replay.")
    args = parser.parse_args()
    if args.learn_only and not args.live:
        parser.error("--learn-only requires --live")
    with TemporaryDirectory() as folder:
        path = Path(folder) / "recipes.sqlite"
        if args.learn_only:
            report = learn_and_replay(path, args.model)
            print(json.dumps(report), flush=True)
        else:
            report = compare(
                path,
                live=args.live,
                model=args.model,
                arms=args.arms,
                max_tokens=args.max_tokens,
                extra_lines=args.extra_lines,
            )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.learn_only and not all(r["qualified"] for r in report["arms"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
