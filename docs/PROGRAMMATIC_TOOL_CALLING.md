# Programmatic tool calling in Prompture

Engineering investigation for Prompture and Tukuy · November 23, 2026

**Build one portable execution path above the drivers.** Prompture can already ask a text-only model to choose a tool. Let that tool run a short workflow over registered tools, keep intermediate values in the application, and return a small, verifiable result. Native function-calling models can invoke the same workflow through their ordinary tool API. Tukuy is a good fit for the transformations inside it. The first release should support developer-authored workflows and a small, validated plan format. General model-written Python and Anthropic's hosted execution are separate extensions.

The best first workload is an exact aggregation over tool data. It has a checkable answer, often needs the entire dataset, and already exposes the limits of truncation, search, and pagination. A single lookup with a short answer should keep its current direct route.

## The video and what the evidence supports

I retrieved the English captions directly with Prompture's YouTube reader. The backend was `youtube/transcript_api`. The chapter links below point to the corresponding caption groups. A third party also hosts a readable, lightly edited transcript. [Original video](https://www.youtube.com/watch?v=8dVCSPXG6Mw&vl=en) · [Readable transcript](https://github.com/kb4ai/mcp-considered-suboptimal-pub-kb/blob/master/archived-resources/prompt-engineering-yt--anthropic-killed-tool-calling--transcript.md)

| Jump to the video | What matters for Prompture |
| --- | --- |
| [2:02 · Context growth](https://www.youtube.com/watch?v=8dVCSPXG6Mw&t=122s) | Repeated tool schemas and large intermediate results consume context for different reasons. Prompture already has mechanisms for both. |
| [3:34 · Execution loop](https://www.youtube.com/watch?v=8dVCSPXG6Mw&t=214s) | A model can specify several operations at once. Code can take the first result, choose the next arguments, and reduce the output before the next inference. |
| [5:06 · Prior implementations](https://www.youtube.com/watch?v=8dVCSPXG6Mw&t=306s) | The pattern exists outside one provider. Cloudflare describes executing generated code with bindings to MCP-backed tools. |
| [8:12 · Search filtering](https://www.youtube.com/watch?v=8dVCSPXG6Mw&t=492s) | Filtering search results before inference is one application. For research agents, preserve source URLs and supporting passages in the filtered output. |
| [10:51 · Cost caveat](https://www.youtube.com/watch?v=8dVCSPXG6Mw&t=651s) | Generating the program consumes output tokens. A smaller input can still produce a more expensive run. |
| [12:24 · Tool declarations](https://www.youtube.com/watch?v=8dVCSPXG6Mw&t=744s) | A hosted implementation needs an explicit way to expose callable tools to its execution environment. |

To retrieve the timestamped text from this checkout, use the same caption backend. Prompture groups the original cues into roughly 30-second paragraphs. This reads an existing English-language track without sending audio to a transcription provider.

```python
import os
from prompture.tools.web import read_url

os.environ["PROMPTURE_READER_YOUTUBE_BACKENDS"] = "transcript_api"
video = read_url(
    "https://www.youtube.com/watch?v=8dVCSPXG6Mw",
    reader="youtube", fallback=False, use_cache=False, max_chars=0,
)
print(video.content)
```

Anthropic reports a drop from **43,588 to 27,297 tokens** on its complex research workloads, about 37%. Its separate web-search study across BrowseComp and DeepSearchQA reports an average **11% performance improvement and 24% fewer input tokens**. Price-weighted tokens decreased for Sonnet 4.6 and increased for Opus 4.6 on those benchmarks. These results motivate measuring Prompture's own workloads. The video's claim about models generally being better at code than tool JSON is too broad to use as a routing rule for older checkpoints. [Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) · [Dynamic filtering study](https://claude.com/resources/articles/improved-web-search-with-dynamic-filtering)

## MCP and programmatic calling in the same request

Suppose a user asks which open invoices exceed each customer's credit limit. An MCP server exposes invoice and customer tools. Prompture discovers their schemas with `tools/list`, calls them with `tools/call`, and receives their results. A program can page through invoices, fetch the distinct customers, join on customer ID, and return the matching invoice IDs and amounts. Those MCP calls still happen. The savings come from avoiding a model inference to coordinate each page or join, and from keeping the underlying records out of subsequent prompts. A server that already offers a filtered invoice query can solve this with one direct call.

MCP defines a protocol for discovery and invocation and supports structured results. The client chooses which schemas and results to show the model. Anthropic and Cloudflare both describe using code alongside MCP. One product detail matters here: Anthropic's hosted PTC docs exclude tools supplied by its **MCP connector**. Prompture can still invoke an MCP server through its own host registry. A hosted integration would expose eligible operations as application-owned client tools and service the callbacks itself. [MCP tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) · [Anthropic's MCP architecture](https://www.anthropic.com/engineering/code-execution-with-mcp) · [Cloudflare Code Mode](https://blog.cloudflare.com/code-mode/) · [Hosted PTC restrictions](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)

## What Prompture already has

| Surface in this checkout | Existing behavior | Integration consequence |
| --- | --- | --- |
| [Conversation](../prompture/agents/conversation.py) and [AsyncConversation](../prompture/agents/async_conversation.py) | With tools registered, `simulated_tools="auto"` uses the driver's native-tool flag. A plain-text driver can emit a JSON call. The base driver can flatten message history for completion-only backends. | One plan tool can use both routes. There is no need to implement an interpreter in every driver. [Base driver](../prompture/drivers/base.py) |
| [Simulated JSON](../prompture/agents/simulated_tools.py) and [prompted streaming grammars](../prompture/agents/tool_grammars.py) | Ordinary simulated calls alternate between a tool request and a final answer. Unparseable prose can become a final answer. Streaming has a separate prompted path. | A final-looking answer is insufficient evidence that an explicitly requested calculation ran. Check the execution receipt. Exercise ordinary, event, and live paths separately. |
| [Registry](../prompture/agents/tools_schema.py) and [agent wrappers](../prompture/agents/agent.py) | The registry validates and coerces arguments and exposes sync and async dispatch. Agent wrappers add run context, callbacks and approval handling. Some validation failures return text. | Child operations must enter the same effective, wrapped registry. Preserve result types and failure status before rendering the result into chat. |
| [Tool context](../prompture/agents/tool_context.py) | Deferred schemas, tool search/load, result offload, previews, and scoped `read_tool_result` already exist. Tool discovery can switch on automatically at an estimated 8,000 schema tokens. | Retain these as the baseline. Discovery cuts schema costs. Offload lets the model inspect a slice. A local reducer can inspect every row without sending each slice back. |
| [MCP hub](../prompture/mcp/hub.py) and [bridge](../prompture/integrations/mcp_bridge.py) | Named servers yield registered tools. The hub serializes returned content and renders MCP errors as strings. | Add a structured execution result for programs. Prefer MCP `structuredContent`, verify `outputSchema` when supplied, and preserve `isError`. Keep the existing text rendering for direct callers. |
| [Tukuy bridge](../prompture/extraction/tukuy_bridge.py) and [PythonSandboxTool](../prompture/tools/code_exec.py) | Prompture already maps skills to tools, supports `ainvoke`, wraps chains as pipeline steps, and can expose `python_execute`. The Python tool passes code to the sandbox and returns formatted output. | Reuse the skill bridge. The existing Python tool has no facility for calling the conversation's registered tools. The pipeline step also stringifies its result, which loses types needed by the next operation. |

There is one compatibility trap. For example, the Ollama driver advertises tool use at driver level. The base capability check can warn when model metadata disagrees. A gateway or local server can therefore land on the native route for a checkpoint that rejects it. `Conversation(..., simulated_tools=True)` is an existing override. For the new feature, propagate an explicit transport override through Agent and AsyncAgent, and remember a confirmed unsupported-tools response per endpoint and model. Retry through text only before any child has executed. [Ollama driver](../prompture/drivers/ollama_driver.py) · [Capability check](../prompture/drivers/base.py)

## Where Tukuy fits

**Put the authority to call tools in Prompture and the reusable data transformations in Tukuy.** Prompture knows the current agent, available tools, provider, budget, and MCP sessions. Tukuy already provides `Chain.run/arun`, `Skill.invoke/ainvoke`, and branch and parallel composition. A host-authored skill can already fetch through a scoped adapter, reduce typed records, and return a scalar or a short list. The reverse `TukuyLLMBackend` bridge is for a skill that requests another LLM completion. An exact filter or sum needs zero such calls. [Tukuy chain](https://github.com/jhd3197/Tukuy/blob/3040f7e9cf73c867f5b838b3889f0ba25c3ed92d/tukuy/chain.py) · [Tukuy skill](https://github.com/jhd3197/Tukuy/blob/3040f7e9cf73c867f5b838b3889f0ba25c3ed92d/tukuy/skill.py) · [Prompture backend bridge](../prompture/bridges/tukuy_backend.py)

Compile a validated step to a known operation. Supply a restricted registry of pure transforms and explicit host callbacks for tool calls. Tukuy's chain can also accept arbitrary Python callables and resolve plugin names through a default registry. Those are valuable developer interfaces. Model-selected operations need a smaller vocabulary. Tukuy-backed safety wrappers cover only the tools they wrap, so the host must enforce eligibility for plain Python and MCP tools too.

Treat free-form Python as a later backend for models that demonstrate sufficient code reliability. Prompture's current wrapper uses Tukuy's in-process restricted execution and AST scoring. Its documented timeout and memory controls have platform limits, particularly on Windows. Executing arbitrary generated Python with access to the application registry would require an isolated worker with enforceable resource limits and a narrow tool-call channel back to the host. Credentials and authorization decisions belong on the host. A bounded JSON interpreter avoids that deployment dependency for v1. [Python tool implementation](../prompture/tools/code_exec.py) · [Tukuy sandbox](https://github.com/jhd3197/Tukuy/blob/3040f7e9cf73c867f5b838b3889f0ba25c3ed92d/tukuy/sandbox/sandbox.py)

## A portable calling contract

The proposed `run_tool_plan` below is a new meta-tool. An older model can emit this through the current simulated envelope. A native model can put the same `steps` in a function call. A developer-authored `total_inventory(warehouse)` recipe offers the shortest interface for a known workflow.

```json
{
  "type": "tool_call",
  "name": "run_tool_plan",
  "arguments": {
    "steps": [
      {"op": "call", "tool": "list_inventory"},
      {"op": "sum", "path": ["*", "A"]}
    ]
  }
}
```

Here the first step produces a map keyed by SKU. The second sums warehouse A's numeric value across its entries. The host returns the total, and can attach the contributing row count and source to the final receipt. Specify in the operation contract whether an absent warehouse means zero or invalid data. Missing values, empty inputs, pagination limits, and partial failures must have explicit semantics for an exact answer.

For dependent calls, extend the same contract with named results, references to earlier typed values, and bounded iteration. In the invoice example, a later step consumes customer IDs extracted from earlier rows. Validate the whole plan's structure and tool names first, then validate each resolved argument immediately before dispatch. Leave intermediate rows in run-scoped memory. Emit only the selected result, coverage, and evidence identifiers. If a cap prevents full coverage, expose that fact to the model.

| Model or task | First route to try | Fallback or limit |
| --- | --- | --- |
| Text-only or older completion model | Simulated JSON selects a named recipe with simple arguments. Offer a short plan and one worked example for novel combinations. | Allow one bounded repair for a malformed plan, before dispatch. If a plan remains invalid, try a suitable recipe or the existing stepwise loop and report uncompleted work. |
| Native function callers, including compatible OpenAI-style, Gemini and local endpoints | Register the same recipe or plan as an ordinary function. Prompture executes it locally. | Select by concrete model and endpoint. Native parallel calls already solve independent small lookups in one model turn. |
| Claude with supported hosted PTC | Optional hosted backend for workflows that benefit from model-written Python. | Keep portable execution available. Hosted support needs versioned code-execution tools, `allowed_callers`, caller IDs, container lifetime, and tool-result-only continuations. The current Claude response normalizer keeps text and ordinary tool calls. [Driver](../prompture/drivers/claude_driver.py) · [Hosted protocol](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) |
| Tiny lookup, ambiguous next step, or decision requiring human judgment | Direct function call, or return a bounded excerpt and let the model reason. | A program pays for its own description and generation. It can also discard evidence the next decision needs. |

**Eligibility for execution and discovery are different decisions.** Resolve each child against an explicit per-run allowlist, independently of whether a schema is currently visible. Exclude the plan runner itself from child calls. Limit steps, fan-out, total calls, bytes, and elapsed time across the whole run. Use bounded concurrency for independent reads. Trace parent and child IDs, tool duration, completion status, and result size while keeping full data out of default model messages and callbacks that feed history.

Start with reads. For later writes, authorize the resolved operation and arguments at the child boundary. Record completion and an idempotency key before making any automated retry decision. A timeout can arrive after the remote side committed. A subsequent model repair, fallback, or container restart must consult that record. The same rule applies when the operator stops a running plan. [Existing fault-injection inventory world](../prompture/execution/sandbox.py)

## What the local experiment measured

A scripted text driver exercised the real `Conversation.ask` simulated loop against Prompture's `InventoryWorld`. The question asked for total units in warehouse A. For zero-based row index `i`, A had `i % 11 + 1` units and B had one. Each route invoked the underlying inventory tool once. The recipe and plan used installed Tukuy 0.0.35 to reduce the same data. The plan prototype recognized only a read and a sum over a specified warehouse. No production feature was enabled.

| Rows | Route | Model requests | Input tokens | Output tokens | Answer |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Current direct call | 2 | 380 | 32 | 1 |
| 1 | Fixed Tukuy recipe | 2 | 348 | 32 | 1 |
| 1 | Two-step plan | 2 | 424 | 63 | 1 |
| 1,000 | Direct, offload disabled | 2 | 20,541 | 35 | 5,995 |
| 1,000 | Current default offload, read all pages | 7 | 73,722 | 470 | 5,995 |
| 1,000 | Fixed Tukuy recipe | 2 | 351 | 35 | 5,995 |
| 1,000 | Two-step plan | 2 | 427 | 66 | 5,995 |

Counts use the **Qwen3-0.6B tokenizer** on each message's content across all requests and on all scripted model replies. They exclude chat-template framing, provider caching, billing, and execution time. The paging route requested 7,000 characters per page and accumulated them in conversation history, so its large total reflects repeated input. A query for one matching SKU could stop after one excerpt. The recipe exposes a single purpose-built tool, and the prototype plan has a tiny two-operation schema. A production catalog or repair prompt will cost more. Artifact identifiers may also change token counts slightly between runs.

On the one-row case the plan spent **75 extra input-plus-output tokens** over direct calling. On 1,000 rows the plan used **493** input-plus-output tokens against **20,576** even for direct calling with offload disabled. This isolates the effect of moving an exact reduction outside chat. It does not establish average production savings, model accuracy, or the cost of arbitrary generated Python.

### What happened with an actual small model

I ran the locally cached Qwen3-0.6B checkpoint on CPU, with greedy decoding, thinking disabled, and a text-only adapter into the same `Conversation.ask` path. The prompt explicitly requested a tool lookup and the exact warehouse A total. With a fixed recipe available, the model returned `1234567890` and made zero calls. With a two-step plan tool available, it returned the same number and again made zero calls. A fresh recipe run reproduced the failure. One corrective turn, including a worked call for warehouse B and an explicit request for A, still produced zero calls.

This failure would survive a JSON syntax check. The model returned a valid final-answer envelope. For tasks that explicitly require tool evidence, the host can check for a successful child invocation with the correct scope before accepting a number. For a known aggregation it can render the verified scalar directly. For open-ended research it can require source evidence and let the model explain it. General chat still needs a legitimate no-tool final-answer path.

**A narrower interface succeeded on the same model.** In a separate test the application selected the inventory recipe. Qwen received the question and the allowed warehouse values, A and B. With a JSON argument prefix already present, the local adapter scored the two valid next tokens and selected the higher-scoring one. Prompture validated and dispatched the recipe, and Tukuy summed the actual records. Asked for A, the model selected A and the host returned **5,995**. Asked for B, it selected B and the host returned **1,000**. No second generation was necessary. This tests two argument selections on a known intent. Open-ended workflow selection, identifiers outside a fixed vocabulary, and endpoints that expose only a text-completion API need separate evaluation.

This suggests a useful feature for older models: let an application explicitly select a recipe when its UI or calling workflow already knows the intent. Give the model just the missing arguments. Use a provider's constrained decoding or a local token constraint when available. Otherwise validate its text output against the same narrow schema, repair once, and surface a failure if it still cannot supply valid arguments. When the user has already supplied typed arguments, dispatch the recipe directly.

## Implementation order and release criteria

1. **Ship one read-only recipe through existing APIs.** Start with inventory totals or another exact reduction. Give it a small parameter schema and a typed result. Support selection by the application for known intents and selection by the model for open-ended requests. Exercise it through Conversation, AsyncConversation, Agent, and AsyncAgent. This is useful to small models even if arbitrary plan generation never meets the accuracy target.
2. **Add typed child dispatch and an opt-in plan tool.** Preserve the existing public text results while exposing success, typed value, error, and provenance internally. Build a typed path for MCP and Tukuy failures. Reuse the agent-wrapped registry, active tool catalog, result policy, artifact scope, budgets, and callbacks. Reject prohibited names and invalid references before effects. Keep the first vocabulary to calls, projections, filters, and reductions. Test against Prompture's pinned `tukuy==0.0.30` TOON dependency as well as the newer checkout before choosing a minimum version. [Dependencies](../pyproject.toml)
3. **Benchmark before automatic routing.** Use matched prompts and inputs across a legacy text endpoint, a small local model forced through simulated mode, a native cloud model, and a hosted PTC model if available. Compare direct, native parallel, deferred/offloaded, fixed recipe, and plan. Include one lookup, a full reduction, two dependent tools, a broad search with citations, malformed plans, missing fields, and interrupted calls. Grade exact answers, source coverage, and side effects. Use the existing deterministic fixtures and reporting harness. [Inventory fixture](../prompture/execution/data/tool_task_inventory.json) · [Evaluation harness](../prompture/execution/harness.py)
4. **Promote by model and workload.** Record first-pass and post-repair success, incorrect answers that claim success, billed input/output/cache tokens, tool and executor cost, tool calls, model turns, and median/p95 latency. Compare cost **per correct completed task**, including failed attempts. A reasonable proposed gate is no loss in exact-match or evidence coverage against the best current route, zero unauthorized or duplicated effects in fault tests, and a reproducible cost or latency improvement on the target workload. Retain a direct-route control for simple requests.
5. **Add hosted PTC or isolated Python only where the evaluation justifies it.** Preserve provider-specific blocks in sync, async, and streaming Claude responses, correlate nested calls, and handle expiry or cancellation. For a portable Python worker, test resource enforcement on each supported OS. Reuse the same host-side dispatch and accounting in both backends.

TOON remains useful at the last boundary when a model must inspect many homogeneous rows. A sum, count, or filtered set of IDs should usually be computed before any JSON or TOON rendering. That is the largest opportunity for Prompture: make the model choose the operation and give it only the information it needs to explain the result.

**Source scope.** Code references are to the local Prompture checkout at `2e8852d` and sibling Tukuy checkout at `3040f7e`. The cached Prompture `origin/dev` was five commits ahead, including stop and streaming changes relevant to eventual integration. The measurements describe this checkout.
