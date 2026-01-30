# Prompture Roadmap

## Completed Work

### v0.0.1–v0.0.21: Core Extraction Engine
- Initial project structure, `ask_for_json()`, JSON schema enforcement
- OpenAI, Claude, Azure, Ollama drivers
- JSON cleaning and AI-powered cleanup fallback
- `extract_and_jsonify()`, `manual_extract_and_jsonify()`
- Driver flexibility: `get_driver()` interface
- Example scripts for each provider

### v0.0.22–v0.0.24: Pydantic & Stepwise Extraction
- `extract_with_model()`: one-shot Pydantic model extraction
- `stepwise_extract_with_model()`: per-field extraction with smart type coercion
- `tools.py` utilities: parsing, schema generation, shorthand numbers, multilingual booleans
- Structured logging and verbose control

### v0.0.25–v0.0.28: Multi-Provider & Field System
- LM Studio, Google Gemini, Groq, OpenRouter, Grok drivers (12 total)
- Sphinx documentation site
- Field definitions registry with 50+ predefined fields
- Enum field support and validation utilities
- Template variables (`{{current_year}}`, `{{current_date}}`, etc.)
- Text classification and analysis examples

### v0.0.29–v0.0.32: TOON, Discovery & AirLLM
- TOON output format support (compact token-oriented notation)
- TOON input conversion via `extract_from_data()` / `extract_from_pandas()` (45-60% token savings)
- `get_available_models()` auto-discovery across configured providers
- `render_output()` for raw text/HTML/markdown generation
- AirLLM driver for local inference
- Live model rates with caching and `get_model_rates()` API

### v0.0.33–v0.0.34: Async, Caching & Conversations
- `AsyncDriver` base class and async driver implementations
- `AsyncConversation` for non-blocking multi-turn interactions
- Response caching with memory, SQLite, and Redis backends
- `Conversation` class: stateful multi-turn sessions with system prompts and message history
- Message-based driver APIs: `generate_messages()`, `generate_messages_stream()`
- Native JSON mode detection per provider (OpenAI `json_schema`, Claude tool-use, Gemini `response_mime_type`)
- `DriverCallbacks` with `on_request`, `on_response`, `on_error`, `on_stream_delta` hooks
- `UsageSession` for accumulated token/cost tracking across calls
- `configure_logging()` with `JSONFormatter` for structured log output

### v0.0.35: Tool Use, Streaming & Plugin System
- `ToolRegistry` and `ToolDefinition`: register Python functions as LLM-callable tools
- `tool_from_function()`: auto-generate JSON schemas from type hints
- Tool use in conversations with multi-round execution (`max_tool_rounds`)
- Streaming via `ask_stream()` and `generate_messages_stream()`
- Pluggable driver registry with entry-point discovery
- `register_driver()` / `register_async_driver()` for third-party provider plugins

### v0.0.36 (current): Vision Support
- `ImageContent` frozen dataclass and `make_image()` smart constructor for bytes, base64, file path, URL inputs
- `image_from_bytes()`, `image_from_base64()`, `image_from_file()`, `image_from_url()` constructors
- `conv.ask("describe", images=[screenshot_bytes])` API on `Conversation` and `AsyncConversation`
- Image support in `ask_for_json()`, `extract_with_model()`, `ask_stream()`, `add_context()`
- Image support in standalone core functions: `render_output()`, `ask_for_json()`, `extract_and_jsonify()`, `extract_with_model()`
- Driver-level `_prepare_messages()` with provider-specific wire formats (OpenAI, Claude, Gemini, Ollama)
- Shared `vision_helpers.py` module for OpenAI-compatible drivers (Groq, Grok, Azure, LM Studio, OpenRouter)
- `supports_vision` capability flag on all drivers (sync and async)
- Universal internal format: `{"type": "image", "source": ImageContent(...)}` content blocks
- Backward compatible: string-only messages unchanged

### v0.0.37: Conversation Persistence
- `conv.export() -> dict` serialization (messages, system prompt, tool definitions, usage)
- `Conversation.from_export(data)` restoration with driver reconstruction
- File-based persistence: `conv.save("path.json")` / `Conversation.load("path.json")`
- SQLite `ConversationStore` backend with tag search, listing, and CRUD
- Optional auto-save on every turn via `auto_save` parameter
- Conversation metadata: `conversation_id`, `tags`, `created_at`, `last_active`, `turn_count`
- Export/import of `UsageSession` alongside conversation state
- `ImageContent` serialization/deserialization with `strip_images` option
- Versioned export format (`EXPORT_VERSION = 1`) with validation on import
- `serialization.py` (pure data transforms) and `persistence.py` (storage backends) modules
- Full async support: mirrored on `AsyncConversation`

---

## Upcoming

### Phase 3: Agent Framework
**Goal**: Higher-level agent abstraction with a ReAct loop, typed context, structured output, and two-tier execution API (simple `run()` + step-by-step `iter()`).

#### Phase 3a: Core Agent Class ✅
- [x] `Agent` class composing `Conversation` + `ToolRegistry` + system prompt + output type
- [x] Constructor: `Agent(model, *, driver, tools, system_prompt, output_type, max_iterations, options)`
- [x] Tool registration via constructor injection (list or `ToolRegistry`) and `@agent.tool` decorator
- [x] `output_type: type[BaseModel]` for structured agent output with JSON parse + `model_validate()` retry (up to 3 attempts)
- [x] `agent.run(prompt) -> AgentResult` — high-level, hides the ReAct loop entirely
- [x] `AgentResult` containing: `output` (typed or str), `output_text`, `messages`, `usage`, `steps: list[AgentStep]`, `all_tool_calls`, `state`
- [x] `AgentStep` dataclass with `step_type` (think/tool_call/tool_result/output), `timestamp`, `content`, `tool_name`, `tool_args`, `tool_result`, `duration_ms`
- [x] `AgentState` enum: `idle`, `running`, `stopped`, `errored`
- [x] `agent.stop()` graceful shutdown flag
- [x] Fresh `Conversation` per `run()` — no state leakage between runs
- [x] Internal ReAct loop delegates to `Conversation._ask_with_tools` via `max_tool_rounds`
- [x] System prompt augmented with JSON schema instructions when `output_type` is set
- [x] `ModelRetry` exception defined (used in Phase 3b guardrails)
- [x] Shared types module: `agent_types.py` (AgentState, StepType, AgentStep, AgentResult, ModelRetry)
- [x] 24 unit tests covering construction, run (no tools / with tools / with output_type), system prompt, stop, state, options
- [x] Example script: `examples/agent_example.py`

#### Phase 3b: Context, Guardrails & Callbacks (next)
- [ ] `RunContext[DepsType]` dataclass passed to tools and system prompt functions: carries deps, model info, usage, message history, iteration count
- [ ] `deps_type` generic on `Agent` for type-safe dependency access in tools
- [ ] Dynamic system prompts: `system_prompt: str | Callable[[RunContext], str]` for context-aware persona rendering
- [ ] `AgentCallbacks` extending `DriverCallbacks` with: `on_step`, `on_tool_start(name, args)`, `on_tool_end(name, result)`, `on_iteration(step_number)`, `on_output(result)`
- [ ] Input validators: `input_guardrails: list[Callable[[RunContext, str], str | None]]` — transform or reject input before loop starts
- [ ] Output validators: `output_guardrails: list[Callable[[RunContext, AgentResult], AgentResult | None]]` — validate final output, raise `ModelRetry` to feed error back to LLM
- [ ] `ModelRetry` exception integration: raised from tools or validators to send error message back to the model with retry budget
- [ ] Per-run `UsageSession` tracking (tokens, cost, errors across all iterations)
- [ ] Iteration limits: `max_cost` (USD budget via `UsageSession`)

#### Phase 3c: Streaming, Iteration & Async
- [ ] `agent.iter(prompt) -> AgentIterator` — low-level step-by-step control, yields `AgentStep` per iteration
- [ ] `agent.run_stream(prompt) -> StreamedAgentResult` — streaming with deltas for each step
- [ ] `AsyncAgent` mirroring `Agent` with `async run()`, `async iter()`, `async run_stream()`
- [ ] Async tool support: tools can be sync or async callables (auto-detected)

### Phase 4: Persona Templates
**Goal**: Reusable, composable system prompt definitions with template variables, layered composition, and a thread-safe registry following the `field_definitions.py` pattern.

#### Persona Data Model
- [ ] `Persona` dataclass with structured fields: `name`, `system_prompt` (template text), `description` (metadata for registries/docs), `traits: list[str]` (behavioral tags), `variables: dict[str, Any]` (default template values), `constraints: list[str]` (rules and guardrails), `model_hint: str | None` (suggested model), `settings: dict[str, Any]` (temperature, max_tokens, etc.)
- [ ] Layered prompt structure within `system_prompt`: role/identity section, behavioral rules, output format constraints, and domain knowledge — following OpenAI/Anthropic prompt engineering best practices (markdown headings or XML sections)

#### Template Rendering
- [ ] Reuse `field_definitions.py` `{{variable}}` substitution engine for persona prompts
- [ ] Built-in runtime variables: `{{current_date}}`, `{{current_year}}`, `{{current_datetime}}`, `{{current_weekday}}` (same as field definitions)
- [ ] Custom per-render variables: `persona.render(user_name="Alice", company="Acme")`
- [ ] `Persona.render(**kwargs) -> str` produces final system prompt with all variables resolved

#### Composition & Extension
- [ ] `persona.extend(additional_instructions) -> Persona` — returns new persona with appended instructions (immutable pattern)
- [ ] Trait composition: `Persona(base="analyst", traits=["concise", "technical"])` merges base persona's prompt with trait-defined behavioral modifiers
- [ ] Trait registry: `register_trait("concise", "Keep responses under 3 sentences.")` — reusable behavioral fragments (inspired by Datasette LLM fragments)
- [ ] Constraint injection: `persona.with_constraints(["Never discuss competitors", "Always cite sources"])` — appends rules without modifying the base prompt
- [ ] `persona + other_persona` merge operator for combining two personas (concatenates prompts, merges traits/variables, warns on conflicts)

#### Thread-Safe Global Registry
- [ ] `register_persona(name, persona)` / `get_persona(name)` with `threading.Lock` (mirrors `field_definitions.py` pattern exactly)
- [ ] `get_persona_names()`, `get_registry_snapshot()`, `reset_registry()`
- [ ] `_PersonaRegistryProxy` for dict-like access: `PERSONAS["analyst"]`
- [ ] Auto-initialization with built-in personas on import

#### Built-in Personas
- [ ] `json_extractor` — precise structured data extraction, low temperature, JSON-only output
- [ ] `data_analyst` — quantitative analysis, cites sources, uses precise numerical language
- [ ] `text_summarizer` — concise summaries, configurable length via variables
- [ ] `code_reviewer` — technical, identifies issues and suggests fixes, structured feedback format
- [ ] `concise_assistant` — general purpose, brief responses, no unnecessary elaboration
- [ ] Each built-in persona includes appropriate `model_hint` and `settings` defaults

#### Integration with Conversation & Agent
- [ ] `Conversation(persona="json_extractor")` shorthand — looks up registry, renders, sets as system prompt
- [ ] `Conversation(persona=my_persona)` — accepts `Persona` instance directly
- [ ] `Agent(persona="analyst")` shorthand (Phase 3 integration) — renders via `RunContext` for dynamic variable injection
- [ ] Dynamic persona support for agents: `system_prompt: str | Persona | Callable[[RunContext], str]` — Persona objects auto-render with RunContext variables
- [ ] `description` field available for multi-agent routing (AutoGen pattern: LLM-facing prompt vs orchestrator-facing metadata are separate concerns)

#### Serialization & Persistence
- [ ] `persona.to_dict() -> dict` / `Persona.from_dict(data) -> Persona` with version field
- [ ] YAML file support: `persona.save_yaml("path.yaml")` / `Persona.load_yaml("path.yaml")` — YAML is the standard for human-authored prompt definitions
- [ ] JSON file support: `persona.save_json("path.json")` / `Persona.load_json("path.json")` — for programmatic generation and API transport
- [ ] `load_personas_from_directory("personas/")` — bulk-load a folder of YAML/JSON persona files into the registry

### Phase 5: Multi-Agent Coordination
**Goal**: Enable multiple agents to collaborate via deterministic workflow groups (sequential, parallel, router) and agent-as-tool composition, with explicit scoped state sharing and aggregate usage tracking.

#### Agent-as-Tool (Foundation Pattern)
- [ ] `agent.as_tool(name, description) -> ToolDefinition` — wraps any `Agent` as a callable tool for another agent
- [ ] Coordinator retains conversation control; sub-agent runs independently and returns result as tool output
- [ ] Optional `custom_output_extractor: Callable[[AgentResult], str]` for transforming sub-agent results before returning to coordinator
- [ ] Sub-agent inherits no conversation history from coordinator (maximum isolation)
- [ ] Sub-agent `AgentResult` captured for tracing even when used as tool

#### Deterministic Workflow Groups
- [ ] `SequentialGroup(agents, *, state, error_policy, max_total_turns)` — agents execute in order, each receiving shared state with outputs from prior agents
- [ ] `ParallelGroup(agents, *, state, error_policy, timeout_ms)` — independent agents run concurrently via `asyncio.gather`, results collected into shared state
- [ ] `LoopGroup(agents, *, exit_condition, max_iterations)` — generator-critic cycle: agents execute in sequence repeatedly until `exit_condition(state) -> bool` returns True
- [ ] All groups accept `state: dict[str, Any]` as initial shared context
- [ ] Groups are composable: a `SequentialGroup` can contain a `ParallelGroup` as a step (nested workflows)

#### Shared State via Named Keys
- [ ] Each agent reads from shared `state: dict[str, Any]` (injected as part of system prompt or `RunContext.deps`)
- [ ] Each agent writes output to a named key: `Agent(output_key="research_data")` — result stored in `state["research_data"]`
- [ ] Template variable injection: agent system prompts can reference `{research_data}` to read other agents' outputs (Google ADK pattern)
- [ ] Explicit data flow: traceable which agent produces and consumes which state keys
- [ ] No shared conversation history — each agent gets only the state keys it needs (minimum necessary context)

#### LLM-Driven Router (Optional)
- [ ] `RouterAgent(model, agents, routing_prompt)` — uses a (cheap) LLM to classify input and delegate to the appropriate specialist agent
- [ ] Routing based on agent `description` fields from Persona metadata (Phase 4 integration)
- [ ] Fallback agent when no specialist matches
- [ ] Router runs a single LLM call for classification, not a full ReAct loop (minimal overhead)

#### Error Handling
- [ ] `ErrorPolicy` enum: `fail_fast` (abort group on first failure), `continue_on_error` (skip failed agent, proceed with partial results), `retry_failed` (retry N times with backoff)
- [ ] Per-agent error state captured in `GroupResult.agent_results`
- [ ] Failed agent's error message available in shared state for downstream agents to handle
- [ ] `max_total_turns` across entire group to prevent runaway costs from agents bouncing between each other

#### Group-Level Usage & Observability
- [ ] `GroupResult` dataclass: `agent_results: dict[str, AgentResult]`, `aggregate_usage: UsageSession`, `shared_state: dict[str, Any]`, `elapsed_ms: float`, `timeline: list[GroupStep]`
- [ ] `GroupStep` dataclass: `agent_name`, `step_type` (started/completed/errored), `timestamp`, `duration_ms`, `usage_delta`
- [ ] Aggregate `UsageSession` across all agents with per-agent bucketing (`_per_agent: dict[str, UsageSession]`)
- [ ] `GroupCallbacks` extending `AgentCallbacks` with: `on_agent_start(name)`, `on_agent_complete(name, result)`, `on_agent_error(name, error)`, `on_state_update(key, value)`
- [ ] Interleaved timeline view: all agent steps merged chronologically for debugging

#### Timeout & Cancellation
- [ ] Per-agent timeout: `Agent(timeout_ms=30000)` — enforced via `asyncio.wait_for`
- [ ] Group-level timeout: `group.run(timeout_ms=120000)` — cancels all remaining agents
- [ ] Cooperative shutdown: `group.stop()` calls `agent.stop()` on all running agents, waits for current iterations to complete
- [ ] `max_total_cost` budget across the group (aggregate `UsageSession` enforced)

#### Async Support
- [ ] `AsyncSequentialGroup`, `AsyncParallelGroup`, `AsyncLoopGroup` mirroring sync variants
- [ ] `ParallelGroup` uses `asyncio.gather` internally (async-native); sync wrapper available via `group.run()` with event loop management
- [ ] `AsyncRouterAgent` for non-blocking routing

#### Serialization & Persistence
- [ ] `GroupResult.export() -> dict` with per-agent results, shared state, aggregate usage, and timeline
- [ ] `group.save("path.json")` for full group run persistence (reuses `serialization.py` patterns)
- [ ] Tag-based storage: `ConversationStore.save(group_id, export, tags=["group_run", "2025-01-30"])`

### Phase 6: Cost Budgets & Guardrails
**Goal**: Prevent runaway costs with pre-flight estimation and enforcement, manage context windows with token-aware history truncation/summarization, rate-limit requests, and validate input/output content — building on the existing `UsageSession`, `DriverCallbacks`, and `CostMixin` infrastructure.

#### Pre-Flight Cost Estimation
- [ ] `estimate_tokens(text) -> int` using tiktoken (OpenAI models) with fallback to character-based heuristic (~4 chars/token) for other providers
- [ ] `estimate_cost(prompt, model, options) -> float` — pre-call cost estimate using `get_model_rates()` pricing data
- [ ] Token count available in `on_request` callback payload: `{"estimated_tokens": int, "estimated_cost": float}` for pre-call decision making
- [ ] Optional tiktoken dependency: graceful fallback to heuristic when not installed

#### Budget Limits & Enforcement
- [ ] `Conversation(max_cost=0.50, max_tokens=10000)` — per-conversation budget caps
- [ ] `Agent(max_cost=1.00, max_tokens=50000)` — per-agent-run budget caps (checked between iterations via `UsageSession`)
- [ ] `BudgetPolicy` enum: `hard_stop` (raise `BudgetExceeded` before the call that would exceed), `warn_and_continue` (fire `on_budget_warning` callback, proceed), `degrade` (switch to cheaper fallback model)
- [ ] `BudgetExceeded` exception with `usage_at_limit: dict` containing tokens/cost consumed when the limit was hit
- [ ] Pre-call budget check: compare `estimate_cost()` against remaining budget before each LLM call — reject if estimated cost would exceed remaining budget
- [ ] Post-call budget check: after each response, update `UsageSession` and check against limits for the *next* call
- [ ] `on_budget_warning(usage, limit, remaining)` callback fired when usage exceeds configurable threshold (default 80%)

#### Model Fallback Chains
- [ ] `Conversation(model="openai/gpt-4o", fallback_models=["openai/gpt-4o-mini", "groq/llama-3.1-8b"])` — ordered list of progressively cheaper models
- [ ] Fallback triggers: `BudgetPolicy.degrade` switches to next model in chain when budget threshold reached
- [ ] Fallback on error: retry with next model on provider errors (rate limit, timeout, 5xx) — configurable via `fallback_on_errors: bool`
- [ ] `on_model_fallback(from_model, to_model, reason)` callback for observability
- [ ] Fallback state tracked in `UsageSession`: which models were used and why

#### Per-Session & Per-Conversation Tracking
- [ ] Automatic `UsageSession` on every `Conversation` and `Agent` (no manual callback wiring required)
- [ ] `conversation.usage_session` property exposing the session with per-model bucketing
- [ ] `conversation.remaining_budget -> dict` with `{"cost": float, "tokens": int}` remaining before limits
- [ ] Cross-conversation session: `UsageSession` can be shared across multiple conversations via constructor injection for global budget enforcement

#### Rate Limiting
- [ ] `RateLimiter` class with token bucket algorithm: `RateLimiter(requests_per_minute=60, tokens_per_minute=100000)`
- [ ] Per-conversation rate limiting: `Conversation(rate_limiter=my_limiter)`
- [ ] Per-model rate limiting: `RateLimiter` scoped to a specific `"provider/model"` string
- [ ] Backpressure behavior: `block` (sleep until bucket refills), `reject` (raise `RateLimitExceeded` immediately)
- [ ] Rate limiter state exposed: `limiter.available_requests`, `limiter.available_tokens`, `limiter.next_available_at`

#### Context Window Management
- [ ] `ContextWindowManager` for token-aware message history management
- [ ] Token budget allocation: configurable split between system prompt, conversation history, and response — `ContextWindowManager(system_reserve=500, response_reserve=1000, max_context=128000)`
- [ ] Context window sizes loaded from `get_model_info()` per-model metadata (falls back to configurable default)
- [ ] Overflow strategy enum: `truncate_oldest` (drop oldest messages first), `summarize` (LLM-compress old messages), `sliding_window` (keep last N messages)
- [ ] `truncate_oldest`: removes oldest non-system messages until history fits within budget, preserving system prompt and most recent messages
- [ ] `sliding_window`: keeps last N turns (configurable `window_size`), drops everything before
- [ ] `Conversation(context_manager=my_manager)` integration — automatically applied before each LLM call

#### Conversation Summarization
- [ ] `summarize` overflow strategy: when history exceeds token budget, compress older messages into a summary using a (cheap) LLM call
- [ ] Summary inserted as a system-level context message: `{"role": "system", "content": "Previous conversation summary: ..."}`
- [ ] Configurable summarization model: `ContextWindowManager(summarize_model="openai/gpt-4o-mini")` — use a cheap/fast model for summarization
- [ ] Hybrid approach: keep last N messages verbatim + summary of everything before (LangChain `ConversationSummaryBufferMemory` pattern)
- [ ] Summary token budget: summary itself has a max token allocation to prevent unbounded growth
- [ ] `on_summarize(original_tokens, summary_tokens, messages_removed)` callback for observability

#### Content Guardrails
- [ ] `InputGuardrail` protocol: `check(content: str) -> GuardrailResult` returning `passed`, `blocked` (with reason), or `modified` (with transformed content)
- [ ] `OutputGuardrail` protocol: `check(content: str, context: dict) -> GuardrailResult` — same return types, with access to conversation context
- [ ] Built-in input guardrails: `RegexBlocker(patterns: list[str])` for blocking patterns (PII, secrets, profanity), `RegexRequirer(patterns: list[str])` for requiring patterns in output
- [ ] Built-in output guardrails: `JsonSchemaValidator(schema)` for format compliance, `MaxLengthValidator(max_chars)` for response length
- [ ] `Conversation(input_guardrails=[...], output_guardrails=[...])` — applied automatically before/after each LLM call
- [ ] `GuardrailResult.blocked` raises `ContentBlocked(reason, content)` exception
- [ ] `GuardrailResult.modified` transparently transforms content and proceeds
- [ ] Guardrail chain: multiple guardrails execute in order; first `blocked` result stops the chain
- [ ] `on_guardrail_triggered(guardrail_name, result, direction)` callback for logging/observability

#### Integration with Agent & Multi-Agent (Phase 3 & 5)
- [ ] Agent inherits conversation-level budgets and guardrails
- [ ] `Agent(input_guardrails, output_guardrails)` — Phase 3b guardrails implemented using this Phase 6 infrastructure
- [ ] Group-level budgets from Phase 5 (`max_total_cost`) enforced via shared `UsageSession` with Phase 6 `BudgetPolicy`
- [ ] `ModelRetry` (from Phase 3b) integrates with output guardrails: guardrail returns `retry` result → feeds error back to LLM

#### Settings & Configuration
- [ ] `Settings` additions: `default_max_cost`, `default_max_tokens`, `default_rate_limit_rpm`, `default_context_overflow_strategy`
- [ ] Environment variable support: `PROMPTURE_MAX_COST=0.50`, `PROMPTURE_MAX_TOKENS=10000`, `PROMPTURE_RATE_LIMIT_RPM=60`
- [ ] All budget/guardrail settings overridable per-conversation or per-agent (constructor params take precedence over Settings defaults)

### Phase 7: Async Tool Execution
**Goal**: Non-blocking tool execution for long-running operations.

- [ ] `@registry.async_tool` decorator for async tool functions
- [ ] Tool timeout configuration per tool
- [ ] Parallel tool execution when LLM requests multiple tools in one turn
- [ ] Tool status polling: tool returns "pending" and agent checks back
- [ ] Tool cancellation support
- [ ] Progress reporting from tools back to the conversation

### Phase 8: Middleware & Interceptors
**Goal**: Pluggable pipeline between conversation and driver for cross-cutting concerns.

- [ ] `Middleware` protocol: `process(message, next) -> message`
- [ ] Built-in middleware: content filtering, prompt compression, rate limiting
- [ ] History summarization middleware: compress old messages to save tokens
- [ ] Logging middleware: structured request/response logging
- [ ] Retry middleware: automatic retry with backoff on transient errors
- [ ] `Conversation(middleware=[filter, compress, log])` configuration
- [ ] Middleware ordering and priority

### Phase 9: Structured Observation Input
**Goal**: Typed input models for feeding structured context to conversations and agents.

- [ ] `Observation` base model for structured input (screen state, metrics, events)
- [ ] Observation-to-prompt template rendering
- [ ] Built-in observation types: `ScreenObservation`, `MetricsObservation`, `EventObservation`
- [ ] Custom observation models via Pydantic
- [ ] `conv.observe(ScreenObservation(app="Chrome", elements=[...]))` API
- [ ] Automatic observation diffing: only send what changed since last observation
- [ ] Observation history alongside message history
