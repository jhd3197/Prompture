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

### v0.0.38–v0.0.39: New Provider Drivers & Reasoning
- **Moonshot AI (Kimi) driver**: OpenAI-compatible API with vision, tool use, streaming; China endpoint support via `https://api.moonshot.cn/v1`
- **Z.ai (Zhipu GLM) driver**: Chinese LLM provider with GLM-4 models
- **ModelScope driver**: Alibaba Cloud AI models
- Async variants for all new drivers (`AsyncMoonshotDriver`, `AsyncZaiDriver`, `AsyncModelScopeDriver`)
- **Reasoning extraction**: `reasoning_content` field extracted from thinking models (Claude extended thinking, Moonshot kimi-thinking, DeepSeek R1)
- `last_reasoning` property on `Conversation` and `AsyncConversation` for accessing model's reasoning chain
- Reasoning support in Ollama and LM Studio drivers for local reasoning models
- Fallback pricing from OpenRouter for Moonshot models when models.dev data unavailable

### v0.0.40–v0.0.41: Simulated Tools & Azure Multi-Backend
- **Simulated tool calling** (`simulated_tools.py`): Prompt-based tool use for drivers without native support
  - `build_tool_prompt(tools)` generates tool descriptions for system prompt
  - `parse_simulated_response(text)` extracts tool calls or final answers from JSON
  - Enables tool use with any model via structured JSON response parsing
- **Azure multi-backend support** (`azure_config.py`):
  - Per-model configuration: `register_azure_config("gpt-4o", {...})`
  - Dynamic resolver: `set_azure_config_resolver(callback)`
  - Backend classification: `classify_backend(model)` → `"openai"` | `"claude"` | `"mistral"`
  - Routes Claude and Mistral models through Azure to their respective APIs
- Strict JSON schema preparation via `prepare_strict_schema()` for OpenAI-style `json_schema` mode
- `json_schema` response format support for Ollama and LM Studio drivers
- `shared_state` parameter and `inject_state()` method on group classes for explicit state management
- Tool use and streaming support added to Google Gemini drivers (sync and async)

### v0.0.42–v0.0.43: Ledger, Discovery & Analysis
- **Persistent model usage ledger** (`ledger.py`):
  - SQLite-backed tracking at `~/.prompture/usage/model_ledger.db`
  - `record_model_usage(model, tokens, cost, status)` — fire-and-forget API
  - `get_recently_used_models(limit)` — query recent usage
  - Per-model stats: call count, total tokens, total cost, first/last used timestamps
  - Thread-safe with internal locking
- **Model capability metadata**:
  - `get_available_models(include_capabilities=True)` returns enriched dicts
  - Capabilities include: context window, vision support, tool use, streaming, JSON mode
  - `verified_only=True` filters to models with successful usage history
  - Integration with models.dev live pricing cache
- **Code analysis module** (`prompture.analysis`):
  - `analyze_python(source)` → `CodeAnalysis` with AST-based security assessment
  - Feature extraction: imports, file ops, network calls, system calls, exec/eval usage
  - Risk scoring: `RiskLevel.LOW`, `MEDIUM`, `HIGH`, `CRITICAL`
  - `is_safe` determination based on configurable threshold
- **Python sandbox** (`prompture.sandbox`):
  - `PythonSandbox` class for safe LLM-generated code execution
  - Import restrictions via allowlist/blocklist with `ImportRestrictions`
  - Path restrictions for file access with `PathRestrictions`
  - Resource limits (timeout, memory) via `ResourceLimits`
  - `SandboxResult` dataclass with output, error, return value, locals
- **History utilities** (`prompture.history`):
  - `filter_steps(steps, step_type, tool_name, timestamp_range)` — filter AgentSteps
  - `search_messages(messages, role, content_contains, has_tool_calls)` — search message history
  - `get_tool_call_summary(result)` — summarize all tool calls with arguments and results
  - `calculate_cost_breakdown(usage)` — detailed token/cost breakdown
  - `export_result_json(result)` — serialize AgentResult to JSON

### v1.0.0: Stable API Release
- Version bump to 1.0.0 indicating stable public API
- All core extraction, conversation, agent, and multi-agent features production-ready
- 15 provider drivers (sync + async): OpenAI, Claude, Google, Groq, Grok, Azure, Ollama, LM Studio, OpenRouter, HuggingFace, AirLLM, Local HTTP, Moonshot, Z.ai, ModelScope
- Comprehensive test coverage across all modules

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

#### Phase 3b: Context, Guardrails & Callbacks ✅
- [x] `RunContext[DepsType]` dataclass passed to tools and system prompt functions: carries deps, model info, usage, message history, iteration count
- [x] `deps_type` generic on `Agent` for type-safe dependency access in tools
- [x] Dynamic system prompts: `system_prompt: str | Callable[[RunContext], str]` for context-aware persona rendering
- [x] `AgentCallbacks` extending `DriverCallbacks` with: `on_step`, `on_tool_start(name, args)`, `on_tool_end(name, result)`, `on_iteration(step_number)`, `on_output(result)`
- [x] Input validators: `input_guardrails: list[Callable[[RunContext, str], str | None]]` — transform or reject input before loop starts
- [x] Output validators: `output_guardrails: list[Callable[[RunContext, AgentResult], AgentResult | None]]` — validate final output, raise `ModelRetry` to feed error back to LLM
- [x] `ModelRetry` exception integration: raised from tools or validators to send error message back to the model with retry budget
- [x] Per-run `UsageSession` tracking (tokens, cost, errors across all iterations)
- [x] Iteration limits: `max_cost` (USD budget via `UsageSession`)

#### Phase 3c: Streaming, Iteration & Async ✅
- [x] `agent.iter(prompt) -> AgentIterator` — low-level step-by-step control, yields `AgentStep` per iteration
- [x] `agent.run_stream(prompt) -> StreamedAgentResult` — streaming with deltas for each step
- [x] `AsyncAgent` mirroring `Agent` with `async run()`, `async iter()`, `async run_stream()`
- [x] Async tool support: tools can be sync or async callables (auto-detected)
- [x] `StreamEvent` and `StreamEventType` types for streaming event classification
- [x] `AsyncAgentIterator` and `AsyncStreamedAgentResult` wrapper classes with result capture
- [x] All new types exported from `prompture.__init__`
- [x] Example script: `examples/async_agent_example.py`

### Phase 4: Persona Templates ✅
**Goal**: Reusable, composable system prompt definitions with template variables, layered composition, and a thread-safe registry following the `field_definitions.py` pattern.

#### Persona Data Model ✅
- [x] `Persona` frozen dataclass with structured fields: `name`, `system_prompt` (template text), `description` (metadata for registries/docs), `traits: tuple[str, ...]` (behavioral tags), `variables: dict[str, Any]` (default template values), `constraints: list[str]` (rules and guardrails), `model_hint: str | None` (suggested model), `settings: dict[str, Any]` (temperature, max_tokens, etc.)
- [x] Layered prompt structure within `system_prompt`: role/identity section, behavioral rules, output format constraints, and domain knowledge — constraints rendered as `## Constraints` section

#### Template Rendering ✅
- [x] Reuse `field_definitions.py` `_apply_templates()` and `_get_template_variables()` for persona prompts
- [x] Built-in runtime variables: `{{current_date}}`, `{{current_year}}`, `{{current_datetime}}`, `{{current_weekday}}` (same as field definitions)
- [x] Custom per-render variables: `persona.render(user_name="Alice", company="Acme")`
- [x] `Persona.render(**kwargs) -> str` produces final system prompt with all variables resolved
- [x] Variable precedence: built-in < `self.variables` < kwargs

#### Composition & Extension ✅
- [x] `persona.extend(additional_instructions) -> Persona` — returns new persona with appended instructions (immutable via `dataclasses.replace`)
- [x] Trait composition: `Persona(traits=("concise", "technical"))` resolves traits from registry during `render()`
- [x] Trait registry: `register_trait("concise", "Keep responses under 3 sentences.")`, `get_trait()`, `get_trait_names()`, `reset_trait_registry()` with `threading.Lock`
- [x] Constraint injection: `persona.with_constraints(["Never discuss competitors"])` — appends rules without modifying the base prompt
- [x] `persona + other_persona` merge operator: concatenates prompts, dedupes traits, merges variables (right wins + warning on conflict), merges constraints and settings

#### Thread-Safe Global Registry ✅
- [x] `register_persona(persona)` / `get_persona(name)` with `threading.Lock` (mirrors `field_definitions.py` pattern)
- [x] `get_persona_names()`, `get_persona_registry_snapshot()`, `clear_persona_registry()`, `reset_persona_registry()`
- [x] `_PersonaRegistryProxy` for dict-like access: `PERSONAS["analyst"]`
- [x] Auto-initialization with built-in personas on import

#### Built-in Personas ✅
- [x] `json_extractor` — precise structured data extraction, `temperature: 0.0`, JSON-only output constraints
- [x] `data_analyst` — quantitative analysis, cites sources, confidence level constraints
- [x] `text_summarizer` — concise summaries, configurable `{{max_sentences}}` variable (default 3)
- [x] `code_reviewer` — structured feedback format (Summary/Issues/Suggestions sections)
- [x] `concise_assistant` — brief responses, no unnecessary elaboration

#### Integration with Conversation & Agent ✅
- [x] `Conversation(persona="json_extractor")` shorthand — looks up registry, renders, sets as system prompt
- [x] `Conversation(persona=my_persona)` — accepts `Persona` instance directly
- [x] `ValueError` when both `persona` and `system_prompt` provided
- [x] `persona.settings` applied as default options (explicit options override)
- [x] `persona.model_hint` used if `model_name` not provided
- [x] Dynamic persona support for agents: `system_prompt: str | Persona | Callable[[RunContext], str]` — Persona objects auto-render with RunContext variables
- [x] `description` field available for multi-agent routing
- [x] Full async support: mirrored on `AsyncConversation` and `AsyncAgent`
- [x] All new symbols exported from `prompture.__init__`

#### Serialization & Persistence ✅
- [x] `persona.to_dict() -> dict` / `Persona.from_dict(data) -> Persona` with `version: 1`
- [x] JSON file support: `persona.save_json()` / `Persona.load_json()`
- [x] YAML file support: `persona.save_yaml()` / `Persona.load_yaml()` (optional `pyyaml` dependency)
- [x] `load_personas_from_directory("personas/")` — bulk-load `.json`/`.yaml`/`.yml` files into the registry
- [x] 72 unit tests covering all sub-phases

### Phase 5: Multi-Agent Coordination ✅
**Goal**: Enable multiple agents to collaborate via deterministic workflow groups (sequential, parallel, router) and agent-as-tool composition, with explicit scoped state sharing and aggregate usage tracking.

#### Agent-as-Tool (Foundation Pattern) ✅
- [x] `agent.as_tool(name, description) -> ToolDefinition` — wraps any `Agent` as a callable tool for another agent
- [x] Coordinator retains conversation control; sub-agent runs independently and returns result as tool output
- [x] Optional `custom_output_extractor: Callable[[AgentResult], str]` for transforming sub-agent results before returning to coordinator
- [x] Sub-agent inherits no conversation history from coordinator (maximum isolation)
- [x] Sub-agent `AgentResult` captured for tracing even when used as tool

#### Deterministic Workflow Groups ✅
- [x] `SequentialGroup(agents, *, state, error_policy, max_total_turns)` — agents execute in order, each receiving shared state with outputs from prior agents
- [x] `ParallelGroup(agents, *, state, error_policy, timeout_ms)` — independent agents run concurrently via `asyncio.gather`, results collected into shared state
- [x] `LoopGroup(agents, *, exit_condition, max_iterations)` — generator-critic cycle: agents execute in sequence repeatedly until `exit_condition(state) -> bool` returns True
- [x] All groups accept `state: dict[str, Any]` as initial shared context
- [x] Groups are composable: a `SequentialGroup` can contain a `ParallelGroup` as a step (nested workflows via `GroupAsAgent` adapter)

#### Shared State via Named Keys ✅
- [x] Each agent reads from shared `state: dict[str, Any]` (injected via template variable substitution in prompts)
- [x] Each agent writes output to a named key: `Agent(output_key="research_data")` — result stored in `state["research_data"]`
- [x] Template variable injection: agent system prompts can reference `{research_data}` to read other agents' outputs (Google ADK pattern)
- [x] Explicit data flow: traceable which agent produces and consumes which state keys
- [x] No shared conversation history — each agent gets only the state keys it needs (minimum necessary context)

#### LLM-Driven Router ✅
- [x] `RouterAgent(model, agents, routing_prompt)` — uses a (cheap) LLM to classify input and delegate to the appropriate specialist agent
- [x] Routing based on agent `description` fields from Persona metadata (Phase 4 integration)
- [x] Fallback agent when no specialist matches
- [x] Router runs a single LLM call for classification, not a full ReAct loop (minimal overhead)

#### Error Handling ✅
- [x] `ErrorPolicy` enum: `fail_fast` (abort group on first failure), `continue_on_error` (skip failed agent, proceed with partial results), `retry_failed` (retry N times with backoff)
- [x] Per-agent error state captured in `GroupResult.agent_results` and `GroupResult.errors` (list of `AgentError`)
- [x] Failed agent's error message available in shared state for downstream agents to handle
- [x] `max_total_turns` across entire group to prevent runaway costs from agents bouncing between each other

#### Group-Level Usage & Observability ✅
- [x] `GroupResult` dataclass: `agent_results: dict[str, AgentResult]`, `aggregate_usage: dict`, `shared_state: dict[str, Any]`, `elapsed_ms: float`, `timeline: list[GroupStep]`
- [x] `GroupStep` dataclass: `agent_name`, `step_type` (agent_run/agent_error), `timestamp`, `duration_ms`, `usage_delta`
- [x] Aggregate usage across all agents via `_aggregate_usage()` helper (prompt_tokens, completion_tokens, total_tokens, total_cost, call_count, errors)
- [x] `GroupCallbacks` with: `on_agent_start(name, prompt)`, `on_agent_complete(name, result)`, `on_agent_error(name, error)`, `on_state_update(key, value)`
- [x] Interleaved timeline view: all agent steps merged chronologically for debugging

#### Timeout & Cancellation ✅
- [x] Per-agent timeout in `ParallelGroup(timeout_ms=30000)` — enforced via `asyncio.wait_for`
- [x] Cooperative shutdown: `group.stop()` calls `agent.stop()` on all running agents
- [x] `max_total_cost` budget across the group (aggregate `UsageSession` enforced)

#### Async Support ✅
- [x] `AsyncSequentialGroup`, `AsyncLoopGroup` mirroring sync variants
- [x] `ParallelGroup` uses `asyncio.gather` internally (async-native); sync wrapper available via `group.run()` with event loop management
- [x] `AsyncRouterAgent` for non-blocking routing

#### Serialization & Persistence ✅
- [x] `GroupResult.export() -> dict` with per-agent results, shared state, aggregate usage, and timeline
- [x] `GroupResult.save("path.json")` for full group result persistence (reuses `serialization.py` patterns)

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
