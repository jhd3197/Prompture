# New Features

## Multi-Agent Debate Groups

Structured multi-agent debates with rounds, assigned positions, and optional judge arbitration.

- **`DebateGroup`** / **`AsyncDebateGroup`** — Run N agents through configurable debate rounds on a given topic. Each round, every agent sees the full transcript of prior arguments before responding.
- **`DebateConfig`** — Configure rounds count, per-agent position labels (e.g. `"FOR"` / `"AGAINST"`), judge agent, judge prompt template, and whether to inject position instructions into prompts.
- **`DebateEntry`** — Typed transcript entry capturing round number, agent name, position, content, timestamp, and per-turn usage.
- **`DebateResult`** — Extends `GroupResult` with the full transcript, rounds completed count, and the judge's verdict string.
- **Judge phase** — After all rounds, an optional judge agent receives the formatted transcript and produces a summary/verdict.
- **Graceful stop** — Call `.stop()` mid-debate to finish the current agent and exit cleanly.
- Full async support via `AsyncDebateGroup` with identical API.

## Router Agent Strategies

`RouterAgent` now supports three routing strategies instead of LLM-only routing.

- **`RoutingStrategy.llm`** (default) — LLM classifies which specialist agent should handle the request.
- **`RoutingStrategy.keyword`** — Match keywords in the user message against per-agent keyword lists. Routes to the agent with the most keyword hits. Zero-LLM-call routing.
- **`RoutingStrategy.round_robin`** — Rotate through agents in order. Stateful index persists across calls.
- **Fuzzy name matching** — LLM routing now uses multi-pass fuzzy matching (exact, substring, word-overlap) to resolve agent names from LLM output.
- **Routing callbacks** — `GroupCallbacks.on_route_decision(agent_name, reason, confidence)` fires on every routing decision with the selected agent, reasoning string, and confidence score (0.0–1.0).
- **`keywords` parameter** — Pass a `dict[str, list[str]]` mapping agent names to keyword lists for keyword-based routing.
- **Fallback agent** — Optional `fallback` agent used when no strategy produces a match.
- Full async support via `AsyncRouterAgent`.

## Sequential Group Step Conditions (Waterfall Mode)

Per-step conditional execution for `SequentialGroup` and `AsyncSequentialGroup`.

- **`step_conditions`** parameter — List of callables `(output_text, shared_state) -> bool`, one per agent. After each agent runs, its condition is evaluated. If it returns `False`, the group stops and returns results collected so far.
- **Escalation tracking** — When a step condition halts execution, `_escalation_stopped_at` is written to shared state with the stopping agent's name.
- **Skip notifications** — `GroupCallbacks.on_step_skipped(agent_name, reason)` fires for every remaining agent that was skipped after a condition returned `False`.
- Enables escalation-style workflows: an L1 agent handles simple cases, and only escalates to L2/L3 when its condition signals the need.

## Group Lifecycle Callbacks

New observability hooks on `GroupCallbacks` for round-based and routing events.

- **`on_round_start(round_number)`** — Fires at the start of each debate round or loop iteration.
- **`on_round_complete(round_number)`** — Fires at the end of each debate round or loop iteration.
- **`on_step_skipped(agent_name, reason)`** — Fires when a waterfall step condition skips an agent.
- **`on_route_decision(agent_name, reason, confidence)`** — Fires when a router selects (or fails to select) an agent.
- These join the existing `on_agent_start`, `on_agent_complete`, `on_agent_error`, and `on_state_update` hooks.

## Budget Enforcement

Cost and token budget tracking with pre-flight estimation and policy-based enforcement across Agents and Conversations.

- **`BudgetPolicy`** enum with three modes:
  - **`hard_stop`** — Raises `BudgetExceededError` immediately when the budget is exceeded.
  - **`warn_and_continue`** — Logs a warning and allows execution to continue past the limit.
  - **`degrade`** — Proactively switches to a cheaper fallback model at 80% budget consumption. Fires `on_model_fallback` callback on switch.
- **`BudgetState`** — Immutable snapshot of budget consumption vs. limits. Exposes `exceeded`, `cost_remaining`, `tokens_remaining`, and `cost_fraction` properties.
- **`enforce_budget()`** — Core enforcement function. Accepts budget state, policy, fallback models list, and optional `on_model_fallback` callback. Returns a new model string when degrading, `None` otherwise.
- **`estimate_tokens(text)`** — Token estimation using tiktoken `cl100k_base` when available, with a `len(text) // 4` heuristic fallback.
- **`estimate_cost(model, input_tokens, output_tokens)`** — USD cost estimation using cached model rates from models.dev.
- **`BudgetExceededError`** — New exception carrying the `budget_state` snapshot for programmatic inspection.
- **Agent/Conversation wiring** — `Agent`, `AsyncAgent`, `Conversation`, and `AsyncConversation` all accept `max_cost`, `max_tokens`, `budget_policy`, and `fallback_models` parameters. Budget is checked before every LLM call.

## Package Exports

All new types are exported from their respective package `__init__.py` files:

- `prompture.groups` — `DebateGroup`, `AsyncDebateGroup`, `DebateConfig`, `DebateEntry`, `DebateResult`, `RoutingStrategy`
- `prompture.infra` — `BudgetPolicy`, `BudgetState`, `enforce_budget`, `estimate_cost`, `estimate_tokens`
- `prompture.exceptions` — `BudgetExceededError`

## Test Coverage

- **`tests/test_debate.py`** — 414 lines covering debate rounds, positions, judge phase, transcript structure, graceful stop, error handling, and async variants.
- **`tests/test_router_strategies.py`** — 268 lines covering LLM, keyword, and round-robin routing, fuzzy matching, fallback behavior, and routing callbacks.
- **`tests/test_step_conditions.py`** — 209 lines covering waterfall stop, escalation tracking, skip callbacks, and edge cases.
- **`tests/test_budget.py`** — 448 lines covering all three budget policies, token/cost estimation, degrade threshold, fallback selection, and the `BudgetExceededError` exception.
