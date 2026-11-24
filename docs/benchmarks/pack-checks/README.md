# Prompture recipes versus Anthropic programmatic tool calling

**For this repeated packing decision, Claude Sonnet 4.6 used fewer tokens when it called a saved Prompture workflow.** On a 12-shipment batch every route returned the correct decision and reason for every shipment. Calling the recipe directly cost 2,157 model tokens, compared with 4,370 for ordinary raw tools and 24,415 for raw tools invoked through Anthropic's hosted programmatic tool calling (PTC). Giving hosted PTC the same recipe reduced its usage to 7,189 tokens and yielded the lowest wall time of the four Claude routes on that run.

### Live results

The larger batch adds 32 matched SKU lines per shipment. Expected decisions and the output schema are identical. Times cover one batch and include tool callbacks and API latency. Each cell shows summed input plus output tokens, exact shipment grades, and elapsed seconds.

| Route | Small batch · tokens / correct / time | Larger batch · tokens / correct / time |
| --- | --- | --- |
| Claude selects ordinary manifest and scan tools | 4,370 / 12 of 12 / 17.88 s | 12,931 / 12 of 12 / 17.63 s |
| Claude hosted PTC operates on the same raw tools | 24,415 / 12 of 12 / 47.06 s | 33,125 / 11 of 12 / 28.86 s |
| Claude calls the saved Prompture recipe directly | **2,157 / 12 of 12 / 8.03 s** | **2,139 / 12 of 12 / 6.71 s** |
| Claude hosted PTC calls the same saved recipe | 7,189 / 12 of 12 / 6.48 s | 7,189 / 12 of 12 / 5.66 s |
| Application calls the saved Prompture recipe | 0 / 12 of 12 / 0.0101 s | 0 / 12 of 12 / 0.0136 s |
| Application calls a fixed Python handler | 0 / 12 of 12 / 0.0005 s | 0 / 12 of 12 / 0.0010 s |

The two Claude recipe routes share the same deterministic business logic and fresh reads. Comparing them isolates the hosted execution route on a known job. On the small batch, direct invocation used **70.0% fewer tokens** than hosted invocation of the same recipe. Comparing direct recipe invocation with raw PTC includes a second advantage: a developer had already written the packing rule. On the larger raw PTC run, Claude held a correct carton containing a repeated observation with the same scan ID. The other recipes handled it correctly. The fixed Python handler was fastest for applications that already know the operation.

**Method.** Each Claude arm received the same synthetic batch identifier, policy, expected response shape, model, and 6,144-token per-request output limit. Raw arms could call only the two data readers. Recipe arms could call only the workflow that combines them. The grader saw all expected IDs and independent target decisions. It required exact decisions, reason codes, no extra IDs, and evidence of reading both data sources. Each final arm read each source once. Hosted callbacks were checked for a code-execution caller, with the container and raw assistant blocks retained across continuations. Token totals sum the Messages API usage fields across requests. Both cache-token counters were zero. The tables report tokens, not dollar savings. [Small batch and per-request usage](matched-small.json) · [Larger batch and per-request usage](matched-large.json) · [Anthropic hosted PTC protocol](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)

**Cold start.** In a separate live run, Prompture's architect used the same Claude model to connect the two read tools and an existing domain reducer into a new graph. It consumed 1,854 input and 512 output tokens, **2,366 total**, and took 7.29 s including execution. Verification against the fixture passed 12/12. After one carton scan changed, the saved graph read both sources again, switched that shipment from pass to hold, and graded 12/12 in 0.0102 s with zero planner calls. This tests learning the wiring of known tools. The reducer and independent oracle were supplied by the application. [Cold-start and changed-scan results](cold-and-replay.json)

**Earlier trials.** The initial scorer required an answer parseable as bare JSON or a single fenced JSON block. Three arms failed that parser. They cannot be graded retrospectively because their final text was not retained. The revised scorer accepts one unambiguous JSON object with surrounding prose and preserves final answers. In the second trial, raw PTC exhausted the 2,048-token output allowance. All final arms were rerun with the same higher limit. Keep these failed attempts when evaluating development spend or reliability. The table describes individual matched runs across two sizes of the same fixture. It does not estimate production percentiles or generalize to unfamiliar tasks. [Initial trial](trial-1-strict-json.json) · [Second trial](trial-2-output-limit.json)

### Using saved recipes

Pass the host's read-only tool registry, a project scope, and a version for tool behavior and policy. The graph hash and tool parameter schemas detect definition changes. Change the contract when the same schema acquires different semantics. A verifier decides whether an output is valid business evidence. The host can publish a reviewed graph explicitly or provide a builder that runs on an exact route miss.

```python
from collections import Counter
from pathlib import Path

from prompture.workflow import RecipeStore
from scripts.pack_checks import fixture, pack_graph, tools_for

manifests, scans, expected = fixture()
tools = tools_for(manifests, scans, Counter())
store = RecipeStore(Path("recipes.sqlite"), scope="warehouse-A", registry=tools, contract="packing-v1")
receipt = store.run_or_learn(
    "audit_pack_batch", {"batch": "B1"}, build=pack_graph,
    verify=lambda output: output["decisions"] == expected, event_id="example-1",
)
print(receipt["decisions"]["S11"], store.popular())
```

Here the verifier uses a fixture oracle. In a deployment, supply an independent check appropriate for the business operation. A repeated event executes against fresh data and increments the verified-use count once per recipe version. To publish confirmed phrases, call `store.publish(graph, aliases=[...])`. Register `store.as_tool(name, verify=...)` in an agent's ToolRegistry to expose the recipe through native or simulated tool calling. Hosts can export graph definitions with `store.load(name).to_json()` and review a deserialized graph before publishing it in another scope. The store executes synchronous, read-only tools and local transforms. Ranking is local to the project. Semantic matching and warehouse writes remain later milestones.

### Reproduce

The first two commands run locally. Commands with `--live` send synthetic data to the configured Anthropic API and incur usage. The comparison exits nonzero if any arm fails the full-batch grade; the JSON report is still written.

```sh
python -m scripts.benchmark_fulfillment_reuse
python -m scripts.benchmark_pack_checks
python -m scripts.benchmark_pack_checks --live --json-out small.json
python -m scripts.benchmark_pack_checks --live --extra-lines 32 --json-out large.json
python -m scripts.benchmark_pack_checks --live --learn-only --json-out learning.json
```

The 60-event pilot also runs on the SDK's persistent recipe store. It still yields 51 saved-graph hits, three graph builds, 108 fresh child calls and 60 correct outcomes, including six appropriate reviews. That percentage measures the specific event mix. It is independent of the Claude token comparison. [Fulfillment plan and original pilot](../../FULFILLMENT_RECIPE_PLAN.md)
