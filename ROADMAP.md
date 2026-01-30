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

#### Core Agent Class
- [ ] `Agent` class composing `Conversation` + `ToolRegistry` + system prompt + output type
- [ ] Constructor: `Agent(model, *, tools, system_prompt, output_type, deps_type, max_iterations, max_cost)`
- [ ] Tool registration via constructor injection and `@agent.tool` decorator
- [ ] `output_type: type[BaseModel]` for structured agent output with validation retry (reuses `extract_with_model()` internally)

#### Typed Context & Dependency Injection
- [ ] `RunContext[DepsType]` dataclass passed to tools and system prompt functions: carries deps, model info, usage, message history, iteration count
- [ ] `deps_type` generic on `Agent` for type-safe dependency access in tools
- [ ] Dynamic system prompts: `system_prompt: str | Callable[[RunContext], str]` for context-aware persona rendering

#### Two-Tier Execution API
- [ ] `agent.run(prompt, *, deps, context) -> AgentResult` — high-level, hides the ReAct loop entirely
- [ ] `agent.run_sync(prompt, *, deps) -> AgentResult` — sync wrapper for non-async contexts
- [ ] `agent.run_stream(prompt, *, deps) -> StreamedAgentResult` — streaming with deltas for each step
- [ ] `agent.iter(prompt, *, deps) -> AgentIterator` — low-level step-by-step control, yields `AgentStep` (think/tool_call/tool_result/output) per iteration
- [ ] `AgentResult` containing: `output` (typed or str), `messages`, `usage`, `steps: list[AgentStep]`, `all_tool_calls`

#### Agent Loop & State
- [ ] Internal ReAct loop: send messages → check for tool calls → execute tools → re-send (reuses `Conversation._ask_with_tools` pattern)
- [ ] Loop modes: `autonomous` (run until output or max_iterations), `single_step` (one LLM call, return)
- [ ] Agent state enum: `idle`, `running`, `paused`, `stopped`, `errored`
- [ ] Graceful shutdown: `agent.stop()` completes current iteration then exits
- [ ] Iteration limits: `max_iterations` (tool rounds) and `max_cost` (USD budget via `UsageSession`)

#### Lifecycle Hooks & Observability
- [ ] `AgentCallbacks` extending `DriverCallbacks` with: `on_step`, `on_tool_start(name, args)`, `on_tool_end(name, result)`, `on_iteration(step_number)`, `on_output(result)`
- [ ] Per-run `UsageSession` tracking (tokens, cost, errors across all iterations)
- [ ] Structured step log: `AgentStep` dataclass with `step_type`, `timestamp`, `content`, `tool_name`, `duration_ms`

#### Guardrails
- [ ] Input validators: `input_guardrails: list[Callable[[RunContext, str], str | None]]` — transform or reject input before loop starts
- [ ] Output validators: `output_guardrails: list[Callable[[RunContext, AgentResult], AgentResult | None]]` — validate final output, raise `ModelRetry` to feed error back to LLM
- [ ] `ModelRetry` exception: raised from tools or validators to send error message back to the model with retry budget

#### Async Support
- [ ] `AsyncAgent` mirroring `Agent` with `async run()`, `async iter()`, `async run_stream()`
- [ ] Async tool support: tools can be sync or async callables (auto-detected)

### Phase 4: Persona Templates
**Goal**: Reusable, composable personality definitions for agents and conversations.

- [ ] `Persona` class with structured fields: name, traits, tone, constraints, knowledge domains
- [ ] Persona registry (like field registry): `register_persona("cautious_explorer", {...})`
- [ ] Composable traits: `Persona(base="explorer", traits=["tech_savvy", "minimalist"])`
- [ ] Persona-to-system-prompt rendering with template support
- [ ] Built-in personas: assistant, analyst, coder, reviewer, device_agent
- [ ] `Conversation(persona="cautious_explorer")` shorthand
- [ ] Persona serialization for storage/sharing

### Phase 5: Multi-Agent Coordination
**Goal**: Enable multiple agents to share context and collaborate.

- [ ] `AgentGroup` for managing a set of agents with a shared objective
- [ ] Shared message bus: agents can send typed messages to each other
- [ ] Coordinator pattern: one agent delegates tasks to others
- [ ] Shared memory: a common context store agents can read/write
- [ ] Agent discovery: agents can query what other agents exist and their capabilities
- [ ] Synchronization primitives: wait for agent, barrier, handoff
- [ ] Group-level usage tracking (aggregate tokens/cost across all agents)

### Phase 6: Cost Budgets & Guardrails
**Goal**: Prevent runaway costs and enforce safety constraints on conversations and agents.

- [ ] `Conversation(max_cost=0.50, max_tokens=10000)` budget caps
- [ ] Budget enforcement modes: hard stop, warn and continue, graceful degrade (switch to cheaper model)
- [ ] Per-session and per-conversation budget tracking
- [ ] Rate limiting: max requests per minute per conversation
- [ ] Content guardrails: blocked patterns, required patterns, output validators
- [ ] Token budget allocation: reserve tokens for system prompt vs history vs response
- [ ] Automatic conversation summarization when history exceeds token budget

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
