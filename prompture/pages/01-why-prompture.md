# Why Prompture?

## The Problem

Getting reliable structured JSON from LLMs is harder than it looks.

- **Provider-native JSON mode** (`response_format: json_object`) only works on recent models from a few providers. Older models, small local models, and many hosted providers have no native support.
- **Tool/function calling** can simulate structured output, but not every model supports it, and the schema mapping is non-trivial.
- **Prompting alone** works everywhere but produces malformed JSON often enough that production systems need repair logic.

Prompture solves this with a **3-tier fallback strategy** that makes *any* LLM return valid, schema-conforming JSON:

1. **Provider native** -- use `json_mode` or `json_schema` when the model supports it.
2. **Tool call** -- wrap the JSON schema as a function definition and extract via tool calling.
3. **Prompted repair** -- prompt for JSON, then clean and repair the output with AI-assisted fixing.

You can let Prompture choose automatically (`strategy="auto"`) or force a specific tier when you know what your model supports.

## Core Strengths

| Feature | Detail |
|---------|--------|
| Any model works | 3-tier fallback with explicit `StructuredOutputStrategy` control |
| Explicit cost tracking | Every response includes `prompt_tokens`, `completion_tokens`, `total_tokens`, and `cost`. Aggregate across calls with `UsageSession` |
| Pydantic-native | `extract_with_model()` returns validated Pydantic instances directly |
| 20+ providers | OpenAI, Anthropic, Google, Groq, Grok, Ollama, LM Studio, Azure, OpenRouter, HuggingFace, Moonshot, and more |
| TOON input | 45-60% token savings when sending structured data to LLMs via Token-Oriented Object Notation |
| Cross-model testing | Spec-driven runner to compare extraction quality, cost, and token usage across models |
| Stable API | Core extraction functions are covered by semantic versioning with a deprecation policy |

## Compared to Alternatives

### vs LangChain

**LangChain** is a full agent engineering platform: 700+ integrations, RAG pipelines, vector stores, LangGraph for stateful agents, LangSmith for observability.

**Prompture** is a focused extraction library: explicit cost tracking, works with weak/old/local models, no framework overhead.

| Dimension | LangChain | Prompture |
|-----------|-----------|-----------|
| Scope | Full orchestration platform | Structured extraction |
| Structured output | `with_structured_output()` on chat models | 3-tier fallback on any model |
| Cost tracking | Via callbacks / LangSmith | Built into every response |
| Weak/local models | Limited JSON support | First-class via prompted repair |
| Learning curve | Steeper (Runnables, LCEL, LangGraph) | Flat (function calls) |

### vs Instructor

**Instructor** patches provider SDKs directly to add Pydantic validation and retry logic. Elegant for supported providers.

**Prompture** uses its own driver layer instead of patching SDKs, supports more providers (especially local models), and adds TOON input for token savings and a cross-model test runner.

| Dimension | Instructor | Prompture |
|-----------|-----------|-----------|
| Approach | SDK monkey-patching | Provider-agnostic drivers |
| Provider coverage | OpenAI, Anthropic, Google, + a few more | 20+ providers including local |
| Token savings | No | TOON input (45-60%) |
| Cross-model testing | No | Built-in spec runner |

### vs Provider SDKs Directly

Using the OpenAI or Anthropic SDK directly gives you the tightest integration and access to the latest features immediately.

Prompture adds value when you need:

- **Multi-provider** support with a single API surface.
- **Cost tracking** across providers without building it yourself.
- **Fallback strategies** for models without native JSON mode.
- **Pydantic integration** without vendor lock-in.

## When NOT to Use Prompture

- You need a **full agent framework** with memory, RAG, and complex tool orchestration -- use LangChain/LangGraph.
- You only use **one provider** and want zero abstraction overhead -- use the provider SDK directly.
- You need **production agent observability** with traces and evaluations -- add LangSmith, Langfuse, or similar.

Prompture is best when your core need is: *get reliable structured data out of LLMs, across providers, with cost visibility*.
