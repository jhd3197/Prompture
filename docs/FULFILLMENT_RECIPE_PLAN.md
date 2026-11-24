# Local workflow reuse in Prompture

The first implementation ships a scoped read-only recipe store, a multi-line pack verifier, and a live Claude comparison at two batch sizes. [Implementation, measurements, and reproduction commands](benchmarks/pack-checks/README.md)

## Decision and first milestone

Build a project-scoped library of verified, parameterized workflows on each machine. Use a fulfillment center as the reference workload. It has frequent repeated requests, rapidly changing facts, and a clear consequence for a wrong answer. Keep the library domain-independent. A warehouse installs its tools and recipes through the same interfaces an invoice-processing or research application would use.

**The first deliverable is a read-only answer to “Can this package leave this station?”** Given a fresh scan and an expected shipment, return `pass`, `hold`, or `review`, the reason, and the identifiers of the records used. A pass is a verification result. Reserving inventory, buying a label, or releasing a parcel to a carrier are separate actions for a later release.

**Success metric:** of all recurring requests, what fraction complete correctly using a verified saved recipe without generating a new plan? Report this separately for scanner events, previously verified wording, and new natural-language requests. Include misses and abstentions in the denominator. Compare with the simplest available alternative, including a fixed event handler.

## What the video contributes

I retrieved the English caption track for Tech Vision's *Inside Amazon's Smart Warehouse* through Prompture's `youtube/transcript_api` backend. These timestamped notes capture the design-relevant text in condensed form. The operating figures and labor claims in the video describe its historical setting. They supply no performance targets for Prompture.

| Video | What the narration describes | Design choice |
| --- | --- | --- |
| [1:33–3:05](https://www.youtube.com/watch?v=IMPbKVb8y8s&t=93s) | Forecasts influence where stock sits before orders arrive. Fulfillment and sortation serve different purposes. | Give stock-position queries, order fulfillment, and downstream sorting separate intents and data sources. |
| [3:05–4:06](https://www.youtube.com/watch?v=IMPbKVb8y8s&t=185s) | Mobile shelving brings goods to a picker at a workstation. | Track location, pod or bin, workstation, and item as separate IDs. A pod's arrival still leaves the item to be verified. |
| [5:08–6:39](https://www.youtube.com/watch?v=IMPbKVb8y8s&t=308s) | Routes change with congestion and charging needs. Equipment handles bulk goods and package labels at other stations. | Reuse the checking procedure. Read current lane, station availability, and label revision on every execution. |
| [7:10–8:13](https://www.youtube.com/watch?v=IMPbKVb8y8s&t=430s) | Receiving, stowing, and picking depend on distinct human observations before an item reaches packing. | Allow entry at each station. Correlate an observed barcode and quantity with the expected order line. |
| [8:13–10:48](https://www.youtube.com/watch?v=IMPbKVb8y8s&t=493s) | The video discusses throughput pressure and gives damaged goods and product condition as examples requiring judgment. | Include exceptions and appropriate handoff to a person in the scoring. A correct hold counts as success. |

Amazon's published process description also distinguishes inbound receiving, stowing, picking, packing, labeling, and outbound transport. Its account of the delivery network distinguishes the fulfillment center, sort center, and delivery station. The fixtures below are proposed simulations inspired by that separation. [Amazon fulfillment-center walkthrough](https://www.aboutamazon.com/news/operations/inside-amazons-fulfillment-centers-what-you-can-expect-to-see-on-a-warehouse-tour) · [Journey of a package](https://www.aboutamazon.com/news/operations/how-do-amazon-packages-get-delivered)

To extract the original timestamped English text into a local file with the checkout's caption reader, run the following in the project environment:

```python
import os
from pathlib import Path
from prompture.tools.web import read_url

os.environ["PROMPTURE_READER_YOUTUBE_BACKENDS"] = "transcript_api"
video = read_url("https://www.youtube.com/watch?v=IMPbKVb8y8s", reader="youtube",
                 fallback=False, use_cache=False, max_chars=0, languages=("en",))
Path("warehouse-captions.txt").write_text(video.content, encoding="utf-8")
```

## One shipment, many entry points

Consider order `O17` with two blue mugs and a book. Shipment `S17a` contains the mugs from facility A, and `S17b` contains the book from facility B. At A, a picker scans two blue mugs. Packing still has to confirm that those mugs went into the carton for `S17a`. A carton containing exactly the right products can still carry yesterday's label, another customer's label, or the wrong carrier lane. Each decision needs different evidence.

| Entry point | Question or event | Evidence and expected behavior |
| --- | --- | --- |
| Supplier, transfer, or return arrives | “Did we receive what was expected?” | Match receipt or return authorization, facility, SKU, lot when applicable, and observed quantity. Route damaged goods to inspection. |
| Stow or replenishment | “Is this bin eligible for this unit?” | Read bin capacity and current restrictions. Detect a duplicate scan or unit already stowed elsewhere. |
| Order or allocation request | “Can we fulfill both lines by the promise date?” | Read available stock net of reservations across sites. Consider split shipments and cutoff times. Return candidate allocations. |
| Pick station | “Is this the right item, quantity, and tote?” | Compare scanned SKU or serial and tote against outstanding shipment lines. Exercise same-name variants and partial picks. |
| Pack station | “Does this carton contain exactly this shipment?” | Compare expected and observed item quantities as multisets. Match carton, shipment, and destination reference separately. Hold missing, extra, wrong-address, and cancelled orders. |
| Sorter, outbound dock, or carrier manifest | “Can this package enter this lane?” | Verify latest label revision, service, lane, manifest, and shipment state. A late relabel invalidates an earlier pass. |
| Tracking, delivery, or return | “Where did custody last change?” | Reconstruct accepted handoffs, distinguish dispatch from delivery, and open a review when proof is absent. A return has its own disposition and stock status. |

Use two independent streams in the fixture: **expected business records** (orders, allocations, manifests) and **physical observations** (scans, quantities, timestamps). Include `tenant/project`, `site`, `event_id`, `observed_at`, `received_at`, and record revision. Link order line, shipment, carton, label, and tracking IDs explicitly. Generate independent distractors so a wrong route cannot pass simply because all mismatches occur on the same rows.

## How local reuse works

1. **Receive and route.** A typed station event selects an eligible intent and supplies IDs directly. For free text, try a verified scoped alias, then retrieve candidates by intent and input schema. If evidence is weak, request missing information, fall back to planning, or abstain. Keep a write request distinct from a similarly worded verification.
2. **Load and validate.** Start with existing `Graph.to_json/from_json`, `graph_version`, and `run_graph(..., options={"tool_registry": ...})`. Rebind the current scoped tools and resolve fresh inputs. Prompture already exposes compiled graphs as agent tools. That adapter currently describes every input as a string, so preserve real input schemas before exposing quantity, list, or object parameters. [Workflow model](../prompture/workflow/model.py) · [Runner and tool adapter](../prompture/workflow/engine.py) · [Tool node](../prompture/workflow/nodes.py)
3. **Identify a recipe.** Freeze a parameterized graph with stable node IDs and named slots such as `shipment_id`. Use its existing content hash for graph integrity. Add a compatibility fingerprint over recipe format, input/output contracts, tool behavior/schema versions, and policy version. A changed tool may keep its name. Keep timestamps, invocation counts, and actual order numbers outside the definition. The current graph hash includes graph ID and metadata, so consistently author or normalize descriptive fields before deduplicating.
4. **Run and verify.** Return reason codes, observed/expected IDs, record versions, and coverage. Require typed success from each child. The current tool node wraps a returned value as a completed result, and some registry validation failures are returned as strings. An independent business verifier must agree before counting success. Incomplete scans or a stale observation produce `review`. Use Tukuy skills for optional transforms or integrations through the existing bridge. [Tool registry](../prompture/agents/tools_schema.py) · [Tukuy bridge](../prompture/extraction/tukuy_bridge.py)
5. **Record local experience.** Store recipes, aliases, and run outcomes in SQLite. Key recipes by scope and fingerprint, and store a source event or request ID with each attempt. Atomically record the outcome and update counters once. Keep attempts, verified successes, verified holds, errors, last success, and representative latency/cost. Popularity breaks ties among compatible, relevant recipes. Quarantine a version that fails verification. Promote new workflows after fixture replay and explicit acceptance or an independent oracle check.
6. **Handle misses.** Supply a small tool catalog to the existing `architect`, validate its graph against the allowed operations, then execute in a read-only fixture or shadow run. Save a reviewed, parameterized definition. A text-only model can select a named workflow through simulated calls. Native tool callers can select the same one via its schema. For models unable to generalize from prose, expose typed entry points or a short list of constrained choices. [Architect](../prompture/workflow/architect.py) · [Prior investigation and token measurements](PROGRAMMATIC_TOOL_CALLING.md)

Persist **which procedure worked**. A previous result may establish that a particular parcel passed at a specified time. Every new verification reads the current scans, order status, and manifest again. Keep separate identifiers for a reusable definition, an invocation of that definition, and a real-world effect. For later write operations, connect Prompture's existing operation ledger and durable recovery to a warehouse business idempotency key stable across retries and replacement runs. Reconcile unknown outcomes with the external system before retrying. [Recovery, graph versioning, and operation ledger](../prompture/workflow/recovery.py)

## First measurements

The [reproducible benchmark](../scripts/benchmark_fulfillment_reuse.py) runs from the repository root with `python -m scripts.benchmark_fulfillment_reuse`. The data and registry tools are synthetic. The graph runner, JSON persistence, fingerprint checks, template binding, and tool dispatch are real. Each supported event reads its own scan and performs the selected check.

| 60 typed events on one machine | Rebuild graph for each event | Reuse verified graphs |
| --- | ---: | ---: |
| Graphs constructed | 54 | 3 |
| Hits on a previously saved graph | 0 | 51 |
| Fresh child tool calls | 108 | 108 |
| Expected outcomes matched | 60 | 60 |
| Mismatches correctly held | 23 | 23 |
| Unsupported return requests routed to review | 6 | 6 |

There were 30 pack, 18 receipt, 6 dock, and 6 return events. The six successive blocks of ten yielded **6, 9, 9, 9, 9, 9** saved-graph hits. That is **51/60 (85%) of arrivals** and **51/54 (94.4%) of eligible events**. The hit rate plateaus once the three recipes are known. SQLite was reopened between blocks. This arm starts from an authoritative event type and constructs deterministic graphs on a miss. Both arms made zero model calls. It measures reuse opportunity and correctness; it cannot establish a token or wall-clock saving over a fixed event handler.

### Free-text routing

The optional model run accepts `--model-dir PATH` for an already downloaded Qwen3-0.6B checkpoint. On CPU with four threads it scores one of four allowed labels per question: pack, receipt, dock, or unsupported. The 16 authored prompts include four per class, including a refund and a request to move a parcel. The model receives only descriptions of the classes. The fixture labels simulate a person verifying which successful routes may be remembered. The repeat comparison uses exact phrases. This deliberately isolates routing cost from graph execution and from search across paraphrases.

| Same 16 phrases on the repeat pass | Route each through Qwen | Reuse verified exact phrases |
| --- | ---: | ---: |
| Actual model calls | 16 | 6 |
| Actual input tokens, including chat template | 1,350 | 500 |
| Total routing wall time, model already loaded | 9.17 s | 3.42 s |
| Correct routing decisions | 12/16 | 12/16 |
| Unsupported requests assigned a supported recipe | 2/4 | 2/4 |

**The second pass used 63.0% fewer routing input tokens.** Counting the first learning pass as well gives 1,850 tokens for 32 requests versus 2,700 with repeated model routing, a **31.5% reduction**. Ten correct supported phrases were eligible for reuse. The two correct unsupported decisions were intentionally left out of memory. The timing is a single local run, excludes model startup, graph execution and verification, and uses an in-memory alias lookup. The warehouse event arm above measures SQLite persistence separately. No provider costs were incurred.

The cold model chose the right recipe on 10/12 supported prompts. It sent a pack check to review, confused a dock check with an inbound receipt, and misrouted two unsupported requests: a refund to receipt verification and a request to move a parcel to lane verification. Exact reuse accelerated known successes and left those four errors intact. This is why the first release should prioritize typed events and verified phrases, and why a later semantic router needs independent tests for lookalike requests.

## Delivery sequence

The first unfamiliar-phrasing benchmark now covers six warehouse recipes, two local models and Claude. It measures candidate retrieval, recipe selection and appropriate abstention on the same requests. [Discovery implementation and results](benchmarks/recipe-discovery/README.md)

| Increment | Work and dependency | Evidence required to finish |
| --- | --- | --- |
| 1. Ground truth and baselines | Extend the existing inventory harness with shipment lines, independent physical scans, split shipments, labels and custody. Implement pack verification first. Add a direct Python handler, current tool loop, fresh architect graph, and pre-authored graph as baselines. | Same input snapshot yields the same externally graded decision and reason. Faults include correct SKU in wrong carton, short quantity, stale label, duplicate scan and cancellation. |
| 2. Local recipe storage and typed entry points | Add project-scoped storage, record/version fingerprinting, schema validation, deduplication, counters, export/import, and invalidation. Bind current tools and inputs on every run. | Correct replay after restart, parallel inserts deduplicate, repeated events count once, changed tool contracts invalidate, and a second project cannot reuse the first project's private aliases or event data. |
| 3. Natural-language discovery | Add human-confirmed aliases and candidate retrieval, with tests for new wording, unsupported intents, ambiguity and missing IDs. Expose a compact callable recipe to both native and simulated tool routes. | Blind evaluation by task, model, and entry point. Report wrong matches and abstentions separately. An unsupported write request must never silently become a successful read check. |
| 4. Multi-site shadow traffic | Feed synthetic events in chronological order with rotating popular intents, new SKUs, shift changes, returns, and tool upgrades. Run the candidate and the best existing route against the same snapshot. | Learning curve for empty, warm, restarted, and invalidated stores. Count every miss, retry, fallback, and verification failure. Confirm a rule-based router remains available for known typed events. |
| 5. Controlled writes | After read verification passes, add simulated reserve, transfer, release, and carrier handoff with effect-specific keys and reconciliation. Test concurrent workers and a timeout after remote commit. | Zero unauthorized or duplicate effects in the fault suite, conserved stock, one accepted handoff per shipment revision, and a resumed run reaches a verified state or a clearly identified review queue. |

## How to get decision-grade numbers

- **Build the held-out set before tuning.** Start with 300 distinct requests across at least three sites, including 100 exception, ambiguous, or unsupported cases. Keep orders, paraphrase families, and at least one site out of training and alias seeding. Replay a separate 10,000-event chronological stream for throughput and popularity drift. These are proposed sample sizes.
- **Measure the cost that can actually disappear.** Log planning, intent routing, argument extraction, execution, verification and explanation separately. For each arm, report planner calls, measured input/output/cache tokens, cost per correct completed request, child tool calls, and end-to-end p50/p95 latency. Record receipts and model/endpoint versions. Local runs report CPU time or wall time and token counts; cloud runs use provider billing metadata. Compare identical output quality and freshness.
- **Publish denominators.** Report correct saved-recipe completions divided by all arrivals and by eligible recurring requests; wrong accepted routes divided by accepted routes; incorrect releases divided by true exceptions; and appropriate reviews divided by true unsupported cases. Break down by typed event, exact alias, unseen text, and each task family. Count human review time separately.
- **Initial promotion gate.** On the blind set, require zero observed false releases and zero scope violations, and no loss in exact decision plus reason versus the best current route. For text requests, aim for at least 25% lower model tokens per correct completion or 20% lower p95 latency with matched coverage. Report uncertainty and the actual sample counts. These are proposed targets; the pilot establishes neither production safety nor an expected hit rate.
- **First engineering decision.** If typed event dispatch wins and novel-text matching remains unreliable, ship verified recipe storage with explicit event and application selection. If semantic routing also clears the gates, enable it by model and intent. Expanding a general-purpose program interpreter can wait until the measured workflows require it.
