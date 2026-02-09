# Prompture Cost & Usage Tracking

Prompture includes a built-in usage tracker that automatically records every LLM call as an individual event in a local SQLite database. No setup required -- it works out of the box.

**Privacy first:** The tracker stores zero message content. Only metadata -- model name, token counts, cost, timing, and opaque IDs for grouping.

Database location: `~/.prompture/usage/usage.db`

---

## Table of Contents

- [Automatic Tracking](#automatic-tracking)
- [Context Scoping](#context-scoping)
- [Querying Usage](#querying-usage)
- [Budget Management](#budget-management)
- [Cost Calculation API](#cost-calculation-api)
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

Calculate costs without making API calls. Uses live rates from models.dev.

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

This is useful when you have a standalone driver outside of the normal extraction/conversation flow and want it tracked.

---

## Configuration

### Environment Variables

```env
USAGE_TRACKING_ENABLED=true          # Enable/disable tracking (default: true)
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
# Via environment
# USAGE_TRACKING_ENABLED=false

# Or programmatically
configure_tracker(enabled=False)
```

When disabled, `record()` is a no-op with zero overhead.

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
