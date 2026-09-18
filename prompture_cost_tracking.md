# Prompture Cost & Usage Tracking

Prompture includes a built-in usage tracker that automatically records every LLM call as an individual event in a local SQLite database. No setup required -- it works out of the box.

The built-in reporting hooks omit raw responses and message bodies. They retain model names, token counts, estimated costs, timing, provider request IDs, pricing provenance, and usage diagnostics. Custom metadata and sinks are caller-controlled; avoid placing message content or secrets in them.

Per-call costs are **estimates**, not invoice amounts. OpenAI and Claude report whether an estimate is complete, partial, or unavailable. Optional organization billing reports provide a separate provider-reported view.

Database location: `~/.prompture/usage/usage.db`

---

## Table of Contents

- [Automatic Tracking](#automatic-tracking)
- [Context Scoping](#context-scoping)
- [Querying Usage](#querying-usage)
- [Budget Management](#budget-management)
- [Cost Calculation API](#cost-calculation-api)
- [OpenAI and Claude Reporting](#openai-and-claude-reporting)
- [Provider Token Counting](#provider-token-counting)
- [OpenAI Responses API](#openai-responses-api)
- [Organization Billing Reports](#organization-billing-reports)
- [Direct SQLite Access](#direct-sqlite-access)
- [DriverCallbacks Integration](#drivercallbacks-integration)
- [Configuration](#configuration)
- [Event Schema Reference](#event-schema-reference)
- [SQL Views Reference](#sql-views-reference)
- [Migration from Legacy Ledger](#migration-from-legacy-ledger)

---

## Automatic Tracking

Every LLM call made through Prompture is automatically tracked. This includes:

- Extraction functions (`extract_with_model`, `ask_for_json`, `render_output`, etc.)
- Conversations (`Conversation.ask()`, `Conversation.ask_for_json()`, etc.)
- Agents (`Agent.run()`, sub-agent calls)
- Async variants of all the above

```python
from prompture import extract_with_model
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

# Automatically tracked -- tokens, cost, timing, model all recorded
result = extract_with_model("Juan is 28 from Miami", Person, model_name="openai/gpt-4")
```

No manual `record_model_usage()` calls needed. The driver hook layer handles everything.

---

## Context Scoping

Tag events with hierarchical context using Python's `contextvars`. Context managers nest automatically -- no need to thread IDs through function signatures.

```python
from prompture.infra.tracker import get_tracker

tracker = get_tracker()

with tracker.session("batch-job-001"):
    with tracker.agent("data-extractor"):
        # This call is tagged with session_id + agent_id
        result = extract_with_model(...)

        with tracker.tool("web-search"):
            # Tagged with session_id + agent_id + tool_name
            other = extract_with_model(...)
```

### Available Scopes

| Scope | Purpose | Example |
|-------|---------|---------|
| `tracker.session(id)` | Group a batch of related work | `"batch-job-001"` |
| `tracker.agent(id)` | Which agent is running | `"pm-agent"` |
| `tracker.conversation(id)` | Which conversation thread | `"conv-abc123"` |
| `tracker.tool(name)` | Which tool triggered the call | `"web-search"` |
| `tracker.operation(name)` | Freeform label | `"ask_for_json"` |

All scopes are optional. If omitted, the field is `None` on the recorded event.

Session IDs are auto-generated (UUID) if you pass `None`:

```python
with tracker.session() as session_id:
    print(session_id)  # "a1b2c3d4-..."
    # all calls inside here share this session_id
```

Agent context is automatically applied inside `Agent._execute()`, so agent runs are tagged without any manual setup.

---

## Querying Usage

### Filtered Queries

```python
tracker = get_tracker()

# All events from a specific provider
events = tracker.query(provider="openai", limit=50)

# Events from a specific agent
events = tracker.query(agent_id="pm-agent", status="success")

# Events from a session
events = tracker.query(session_id="batch-job-001")

# Events in a time range (ISO 8601 strings)
events = tracker.query(start="2025-01-01", end="2025-01-31")

# Combine filters
events = tracker.query(provider="openai", agent_id="pm-agent", limit=100)
```

Each event is a dict with all fields from the [Event Schema](#event-schema-reference).

### Aggregated Summary

```python
summary = tracker.summary(provider="openai")

summary.total_events          # 12
summary.total_cost            # 0.0342
summary.total_tokens          # 15420
summary.total_prompt_tokens   # 10200
summary.total_completion_tokens  # 5220
summary.total_elapsed_ms      # 8340.5
summary.models                # {"openai/gpt-4": 0.03, "openai/gpt-3.5-turbo": 0.004}
summary.providers             # {"openai": 0.0342}
```

### Quick Cost Checks

```python
tracker.cost_today()          # Total USD spent today (UTC)
tracker.cost_this_month()     # Total USD spent this month (UTC)
tracker.cost_by_model()       # {"openai/gpt-4": 0.12, "claude/sonnet": 0.05, ...}
tracker.cost_by_provider()    # {"openai": 0.12, "claude": 0.05, ...}
```

---

## Budget Management

Set spending limits with automatic period tracking.

### Setting Budgets

```python
tracker = get_tracker()

# $5/month global budget
tracker.set_budget("global", limit_cost=5.00, period="monthly")

# Daily token cap
tracker.set_budget("daily-cap", limit_tokens=1_000_000, period="daily")

# Per-agent budget
tracker.set_budget("agent:pm", limit_cost=2.00, period="monthly")

# All-time budget (no reset)
tracker.set_budget("project-x", limit_cost=50.00, period="all")
```

### Checking Budgets

```python
status = tracker.check_budget("global")

status.scope            # "global"
status.exceeded         # False
status.current_cost     # 1.23
status.limit_cost       # 5.00
status.remaining_cost   # 3.77
status.current_tokens   # 45000
status.limit_tokens     # None (not set for this budget)
status.remaining_tokens # None
```

### Enforcing Budgets

```python
from prompture.infra.tracker import get_tracker, BudgetExceededError

tracker = get_tracker()
status = tracker.check_budget("global")

if status.exceeded:
    raise BudgetExceededError(status)
    # "Budget exceeded for scope 'global': cost $5.0100 / $5.0000"
```

### Budget Periods

| Period | Resets | Use Case |
|--------|--------|----------|
| `"daily"` | Every UTC day | Rate limiting |
| `"monthly"` | Every UTC month | Spending caps |
| `"all"` | Never | Project-lifetime budgets |

---

## Cost Calculation API

Calculate costs without making inference API calls. Rates resolve from the local model registry and then models.dev fallback data. Use the detailed estimate below when missing prices or billing modifiers matter; the legacy numeric helpers return zero when rates are unavailable.

```python
from prompture.infra.tracker import UsageTracker

cost = UsageTracker.calculate_cost(
    model_name="openai/gpt-4",
    input_tokens=1000,
    output_tokens=500,
)
# Returns: 0.045 (USD)
```

Also available as a standalone function:

```python
from prompture.infra.cost_mixin import calculate_cost

cost = calculate_cost("openai/gpt-4", input_tokens=1000, output_tokens=500)
```

This is the **public** replacement for the previously private `CostMixin._calculate_cost()`.

---

## OpenAI and Claude Reporting

Driver responses keep the existing `meta.cost`, `prompt_tokens`, `completion_tokens`, and `total_tokens` fields. The OpenAI and Claude drivers add:

| Field | Meaning |
| --- | --- |
| `cost_status` | `estimated`: all observed dimensions priced; `partial`: some dimensions or usage missing; `unknown`: no model rates |
| `rates_available` | Whether base model rates were found; explicit zero rates can represent a free model |
| `usage_complete` | Whether the provider supplied the required terminal usage counts |
| `cost_breakdown` | USD amounts for uncached input, cache reads, 5-minute and 1-hour cache writes, output, tools, and total |
| `pricing` | Rate source, base/effective rates per million tokens, applied rules, and unpriced dimensions |
| `usage_details` | Available cache, reasoning, audio, image, text, and prediction token details |
| `cache_savings` | Estimated net savings against uncached input, including cache write premiums |
| `requested_model`, `returned_model` | Requested model and the effective model reported by the provider |
| `request_id`, `response_id`, `service_tier` | Provider attribution where available |

Reasoning and prediction tokens are subsets of output usage; they are not added to the bill twice. Claude input totals include uncached input, cache reads, and cache writes. Actual mixed 5-minute/1-hour cache-write counts override a requested TTL. Missing terminal stream usage remains explicitly incomplete.

Pricing rules cover verified model-specific long-context thresholds, supported service-tier discounts, and supported inference geography premiums. Unknown modifiers are recorded in `pricing.unpriced` and make the estimate partial. Pricing snapshots are estimates of public rates; negotiated contracts, unsupported modalities, storage, and unobserved charges need provider billing reconciliation.

Cost budgets still operate on the numeric estimate. A partial or unknown estimate cannot establish a hard upper bound on the provider's eventual charge; inspect reporting status when enforcing a billing policy.

Known server tool counts can contribute separately priced charges: Claude web search and OpenAI file search / identifiable web search variants. Search-content token allocations, unknown tool variants, and unsupported tool fees remain partial rather than silently free. Cache savings are only aggregated for fully priced estimates and can be negative when write premiums outweigh reads.

```python
from prompture import estimate_call_cost
from prompture.infra.tracker import get_tracker

estimate = estimate_call_cost(
    "openai/gpt-5.5", 300_000,
    expected_completion_tokens=1_000,
    cached_tokens=200_000,
    service_tier="flex",
)
print(estimate.total_cost, estimate.cost_status, estimate.pricing)

tracker = get_tracker()
print(tracker.summary(provider="openai"))
print(tracker.efficiency_report(provider="openai"))
```

`summary()` aggregates **all matching events**, independently of `query()`'s default 1,000-row display limit. `efficiency_report()` includes component costs, pricing-status counts, incomplete usage, provider cache ratio, local cache hits, retry/fallback spend, cache savings, and cost per successful extraction. Missing evidence is `None`, not an invented zero.

Validated `extract_with_model`, `extract_with_models`, and stepwise extraction calls (sync and async) automatically tag actual retry and model-fallback attempts. A separate zero-cost `extraction_outcome` event records the final validation result. All attempts share an `extraction_id`; `total_calls` excludes outcome events, while `total_events` includes them. Cost per successful extraction includes failed-attempt spend for the observed extraction outcomes. A caller-supplied default after failure is not a validated success. Raw JSON parsing alone does not establish a validated extraction outcome. SDK-internal transport retries are not individually visible to these hooks.

Conversation and session usage retain per-call `usage_records`, including rich metadata, plus additive breakdowns. Their memory use grows with the number of retained calls. A normally completed conversation stream records usage once; early closure remains incomplete if final usage never arrived. Existing historical events without these fields are reported as unclassified; new metadata is not reconstructed retroactively.

## Provider Token Counting

Count the actual structured request before generation, including messages, tools, images, and supported schema instructions:

```python
from prompture import count_request_tokens, estimate_request_cost

messages = [{"role": "user", "content": "Extract the delivery date: October 12."}]
count = count_request_tokens("claude/claude-sonnet-4-6", messages)
forecast = estimate_request_cost(
    "openai/gpt-5.5", messages, expected_completion_tokens=100,
)
print(count.input_tokens, forecast.total_cost, forecast.cost_status)
```

These helpers make explicit network requests using ordinary inference credentials. They do not generate output or create generation usage events. Async equivalents are `acount_request_tokens()` and `aestimate_request_cost()`. Output length and future cache hits still require a forecast. Use a provider SDK version exposing the counting endpoint.

OpenAI counts the **Responses representation**, even when the selected driver's generation transport is Chat Completions. This is not an exact Chat Completions billing guarantee. Claude uses `messages.count_tokens`; provider counts are preflight estimates and may differ from final generation usage.

## OpenAI Responses API

Chat Completions remains the default. Select Responses explicitly on the driver or through `options={"api": "responses"}`:

```python
from prompture.drivers.openai_driver import OpenAIDriver

driver = OpenAIDriver(model="gpt-5.5", api="responses")
response = driver.generate("Summarize this delivery note.", options={"max_tokens": 100})
print(response["text"], response["meta"]["cost_breakdown"])
```

The transport supports sync/async generation, streaming, function calls, structured output, reasoning options, and terminal usage metadata. Storage defaults to `False`; enable `store` explicitly if using server-managed conversation state. Prompt-cache diagnostics, when returned by a supported model, are preserved as `meta.prompt_cache_diagnostics`. Pass provider-supported `prompt_cache_options` through options, including `comparison_response_id` when comparing cache behavior. Diagnostics availability is model-dependent; selecting Responses alone does not guarantee diagnostics or change pricing.

## Organization Billing Reports

`BillingClient` reads organization reports only when explicitly called. It uses **admin credentials**, never a fallback inference API key:

Keys may be supplied through `admin_key=`, the environment, or Prompture's `.env` settings. Explicit arguments take precedence; an explicitly empty environment value prevents fallback to `.env`. Settings represent these credentials as `SecretStr`.

- OpenAI: `OPENAI_ADMIN_KEY`, `/v1/organization/usage/completions`, and `/v1/organization/costs`.
- Anthropic: `ANTHROPIC_ADMIN_KEY`, `/v1/organizations/usage_report/messages`, and `/v1/organizations/cost_report`.

```python
from datetime import datetime, timezone
from prompture import BillingClient

start = datetime(2026, 9, 1, tzinfo=timezone.utc)
end = datetime(2026, 9, 2, tzinfo=timezone.utc)
with BillingClient("openai", organization_id="your-organization-id") as billing:
    usage = billing.get_usage(start, end, group_by=["model", "project_id"])
    costs = billing.get_costs(start, end, group_by=["project_id", "line_item"])
    print(costs.totals, costs.complete, costs.warnings)
```

Use timezone-aware, bucket-aligned bounds. Reports cover `[start, end)` in UTC. Usage supports minute/hour/day buckets; costs use daily buckets. All pages are fetched. `complete=True` means pagination completed, not that delayed billing records have arrived or an invoice is final. Errors raise `BillingAPIError` with the partial report attached; `allow_partial=True` returns it with errors instead. Async `aget_usage`, `aget_costs`, and `areport` offload the HTTP client to a worker thread.

Amounts use `Decimal` USD; Anthropic's decimal cents are converted to dollars. Anthropic costs support workspace/description grouping but no API filters and exclude Priority Tier costs. OpenAI usage and cost grouping capabilities differ. Unsupported groups and filters are rejected instead of ignored. Provider usage rows preserve their original dimensions and usage fields.

Reports are **separate from the local usage ledger**. They are never added to local estimates or divided into fictitious per-call charges. `reconcile_costs(LocalCostSummary(...), costs)` compares only matching provider, known organization identity, filters/exclusions, currency, and exact time range. `LocalCostSummary` is a caller-attested aggregate: include all activity in that scope, count unknown or partial-price events in `unknown_cost_events`, and mark missing coverage incomplete. Do not copy a provider scope unless local data was actually filtered to match. The local tracker's inclusive end filter differs from the billing API's exclusive end; enforce `[start, end)` when constructing a reconciliation aggregate. Differences can reflect reporting delays, negotiated rates, or calls made outside Prompture.

See [the runnable example](examples/cost_reporting_example.py). Reference contracts: [OpenAI pricing](https://developers.openai.com/api/docs/pricing), [OpenAI organization usage](https://developers.openai.com/api/reference/resources/admin/subresources/organization/subresources/usage), [OpenAI token counting](https://developers.openai.com/api/docs/guides/token-counting), [OpenAI cache diagnostics](https://developers.openai.com/api/docs/guides/prompt-caching/diagnostics), [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing), [Claude usage and cost API](https://platform.claude.com/docs/en/manage-claude/usage-cost-api), and [Claude token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting).

## Direct SQLite Access

The database is a standard SQLite file with WAL mode enabled for concurrent reads. External tools, dashboards, or scripts can query it directly.

**Location:** `~/.prompture/usage/usage.db`

```python
import sqlite3

conn = sqlite3.connect("~/.prompture/usage/usage.db")
conn.row_factory = sqlite3.Row

# Total spend this month
row = conn.execute(
    "SELECT SUM(cost) FROM usage_events WHERE timestamp LIKE ?",
    ("2025-06%",)
).fetchone()
print(f"June spend: ${row[0]:.2f}")

# Cost per conversation
rows = conn.execute("SELECT * FROM conversation_costs").fetchall()
for r in rows:
    print(f"  {r['conversation_id']}: ${r['total_cost']:.4f}")

conn.close()
```

### Using Pre-Built Views

```sql
-- Daily spend history
SELECT * FROM daily_costs ORDER BY day DESC;

-- Which models cost the most
SELECT * FROM model_costs ORDER BY total_cost DESC;

-- Provider breakdown
SELECT * FROM provider_costs;

-- Per-conversation costs
SELECT * FROM conversation_costs;

-- Per-agent costs
SELECT * FROM agent_costs;

-- Backward-compatible with old ModelUsageLedger format
SELECT * FROM model_usage;
```

---

## DriverCallbacks Integration

Wire the tracker into any driver instance as callbacks:

```python
from prompture.infra.tracker import get_tracker

tracker = get_tracker()

# Create callbacks with fixed context
callbacks = tracker.as_callbacks(
    session_id="my-session",
    agent_id="my-agent",
)

# Attach to a driver -- every call through this driver is now tracked
driver.callbacks = callbacks
```

Use this adapter for integrations that do not invoke Prompture's automatic driver hooks. Attaching a tracker callback to a driver already using those hooks records each call twice. `UsageSession` callbacks can collect a separate in-memory report without adding another event to the SQLite ledger.

---

## Configuration

### Environment Variables

```env
USAGE_TRACKING_ENABLED=true          # Apply tracker settings at package import
USAGE_DB_PATH=/custom/path/usage.db  # Custom database path
USAGE_FLUSH_THRESHOLD=10             # Events buffered before auto-flush (default: 10)
```

### Programmatic Configuration

```python
from prompture.infra.tracker import configure_tracker

tracker = configure_tracker(
    enabled=True,
    db_path="/custom/path/usage.db",
    flush_threshold=5,
)
```

### Disabling Tracking

```python
configure_tracker(enabled=False)
```

When disabled, `record()` is a no-op. Use the programmatic switch to disable recording: the current initialization path does not apply `USAGE_TRACKING_ENABLED=false` to a lazily created tracker.

---

## Event Schema Reference

Every LLM call produces one `UsageEvent` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | UUID, auto-generated |
| `timestamp` | `str` | UTC ISO 8601, auto-generated |
| `model_name` | `str` | Full model identifier, e.g. `"openai/gpt-4"` |
| `provider` | `str` | Provider name, e.g. `"openai"`, `"claude"` |
| `api_key_hash` | `str` | 8-char SHA256 prefix of the API key used (empty for local models) |
| `prompt_tokens` | `int` | Input token count |
| `completion_tokens` | `int` | Output token count |
| `total_tokens` | `int` | Total tokens (prompt + completion) |
| `cost` | `float` | USD cost of the call |
| `elapsed_ms` | `float` | Wall-clock latency in milliseconds |
| `session_id` | `str?` | From `tracker.session()` context |
| `conversation_id` | `str?` | From `tracker.conversation()` context |
| `agent_id` | `str?` | From `tracker.agent()` context |
| `tool_name` | `str?` | From `tracker.tool()` context |
| `operation` | `str?` | From `tracker.operation()` context |
| `cache_hit` | `bool` | Whether the response came from cache |
| `status` | `str` | `"success"` or `"error"` |
| `error_type` | `str?` | Exception class name on failure |
| `tags` | `list[str]` | Custom string tags (stored as JSON) |
| `metadata` | `dict` | Custom key-value data (stored as JSON) |

---

## SQL Views Reference

The database includes these pre-built views:

### `daily_costs`

| Column | Description |
|--------|-------------|
| `day` | Date string (`YYYY-MM-DD`) |
| `total_cost` | Sum of costs for the day |
| `total_tokens` | Sum of tokens for the day |
| `event_count` | Number of LLM calls |

### `model_costs`

| Column | Description |
|--------|-------------|
| `model_name` | e.g. `"openai/gpt-4"` |
| `total_cost` | Lifetime cost for this model |
| `total_prompt_tokens` | Lifetime input tokens |
| `total_completion_tokens` | Lifetime output tokens |
| `total_tokens` | Lifetime total tokens |
| `event_count` | Total calls to this model |

### `provider_costs`

| Column | Description |
|--------|-------------|
| `provider` | e.g. `"openai"` |
| `total_cost` | Lifetime cost for this provider |
| `total_tokens` | Lifetime tokens |
| `event_count` | Total calls |

### `conversation_costs`

| Column | Description |
|--------|-------------|
| `conversation_id` | Conversation identifier |
| `total_cost` | Total cost for the conversation |
| `total_tokens` | Total tokens |
| `event_count` | Number of LLM calls |

### `agent_costs`

| Column | Description |
|--------|-------------|
| `agent_id` | Agent identifier |
| `total_cost` | Total cost for the agent |
| `total_tokens` | Total tokens |
| `event_count` | Number of LLM calls |

### `model_usage` (backward compatibility)

Matches the old `ModelUsageLedger` schema:

| Column | Description |
|--------|-------------|
| `model_name` | Model identifier |
| `api_key_hash` | API key hash |
| `use_count` | Number of calls |
| `total_tokens` | Lifetime tokens |
| `total_cost` | Lifetime cost |
| `first_used` | Earliest timestamp |
| `last_used` | Latest timestamp |
| `last_status` | Always `"success"` |

---

## Migration from Legacy Ledger

The old `ModelUsageLedger` (`~/.prompture/usage/model_ledger.db`) stored only per-model aggregates. The new tracker stores individual events with full context.

- `record_model_usage()` still works but emits a `DeprecationWarning` and delegates to the new tracker.
- `get_recently_used_models()` still reads from the old ledger for backward compatibility.
- The `model_usage` SQL view in the new database provides the same schema as the old ledger table.
- Both databases coexist -- the old one is not deleted or modified.

To query the new system with the old schema shape:

```sql
-- Same columns as the old model_ledger.db
SELECT * FROM model_usage;
```
