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

### v0.0.35 (current): Tool Use, Streaming & Plugin System
- `ToolRegistry` and `ToolDefinition`: register Python functions as LLM-callable tools
- `tool_from_function()`: auto-generate JSON schemas from type hints
- Tool use in conversations with multi-round execution (`max_tool_rounds`)
- Streaming via `ask_stream()` and `generate_messages_stream()`
- Pluggable driver registry with entry-point discovery
- `register_driver()` / `register_async_driver()` for third-party provider plugins

---

## Upcoming

### Phase 1: Vision Support
**Goal**: Native multimodal input in conversations and extraction.

- [ ] `ImageContent` type for passing images (bytes, base64, file path, URL) in messages
- [ ] `conv.ask(text="What's on screen?", images=[screenshot_bytes])` API
- [ ] Image support in `extract_with_model()` for vision-based structured extraction
- [ ] Driver-level image encoding per provider (Claude base64, OpenAI URL/base64, Gemini inline)
- [ ] Automatic image resizing/compression to stay within provider token limits
- [ ] Vision examples: screenshot analysis, document extraction, chart reading

### Phase 2: Conversation Persistence
**Goal**: Save and restore conversations across process restarts.

- [ ] `conv.export() -> dict` serialization (messages, system prompt, tool definitions, usage)
- [ ] `Conversation.from_export(data)` restoration
- [ ] File-based persistence: `conv.save("path.json")` / `Conversation.load("path.json")`
- [ ] SQLite backend for conversation storage with search by ID/tag
- [ ] Optional auto-save on every turn (configurable)
- [ ] Conversation metadata: tags, created_at, last_active, turn_count
- [ ] Export/import of `UsageSession` alongside conversation state

### Phase 3: Agent Framework
**Goal**: Higher-level agent abstraction for observe-think-act loops.

- [ ] `Agent` class wrapping `Conversation` + `ToolRegistry` + observation function
- [ ] Configurable loop: `Agent(observe=fn, tools=registry, persona="...", max_cycles=50)`
- [ ] Built-in loop modes: autonomous (continuous), command (task-driven, stops on completion), single (one cycle)
- [ ] Cycle callbacks: `on_observe`, `on_think`, `on_act`, `on_cycle_end`
- [ ] Agent state machine: idle, observing, thinking, acting, paused, stopped
- [ ] Graceful shutdown with `agent.stop()` and current-cycle completion
- [ ] Agent logging: structured action log with timestamps and tool call traces

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
