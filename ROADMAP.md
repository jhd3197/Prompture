# Prompture Roadmap

Current state: **v0.0.29.dev** | 12 drivers | 50+ field definitions | 171 tests | 28 examples

---

## Where Prompture Stands Today

Prompture has a solid foundation: a clean driver abstraction across 12 LLM providers, a thread-safe field registry with 50+ predefined fields, Pydantic model extraction (one-shot and stepwise), TOON input conversion for 45-60% token savings, cost tracking, and a spec-driven cross-model test runner. The CI/CD pipeline auto-publishes to PyPI on merge to main.

What follows is an honest assessment of what would move the project forward, organized from highest to lowest impact.

---

## 1. Streaming Support

**Why it matters:** Every major provider now supports streaming. For extraction tasks on larger inputs, users currently wait for the full response before getting anything back. Streaming unlocks real-time progress feedback and partial result handling.

**What this looks like:**
- Add a `stream=True` parameter to `generate()` in the base `Driver` class
- Implement streaming in OpenAI, Claude, Google, and Groq drivers first (they all have native streaming APIs)
- Yield chunks or partial JSON as they arrive
- For extraction functions, accumulate streamed text and parse once complete
- Ollama already returns `stream: False` explicitly -- flip this and handle the NDJSON response format

**Scope:** Medium-large. Each driver needs its own streaming implementation due to API differences.

---

## 2. Async Driver Support

**Why it matters:** Prompture is synchronous end-to-end. Anyone using it inside a FastAPI, Django Channels, or any async application has to wrap calls in `run_in_executor`. Batch extraction across multiple models is sequential.

**What this looks like:**
- Add an `AsyncDriver` base class with `async generate()`
- Provide async variants of core functions: `async_extract_with_model()`, `async_extract_and_jsonify()`
- OpenAI, Anthropic, and Google SDKs all have native async clients
- For drivers using `requests`, swap in `httpx.AsyncClient`
- Enable concurrent multi-model extraction with `asyncio.gather()`

**Scope:** Large. Touches the driver layer and core extraction functions. Could be done incrementally -- async drivers first, then async core wrappers.

---

## 3. Response Caching

**Why it matters:** Extraction tasks are often repeated with identical or near-identical prompts during development, testing, and batch processing. Every repeated call burns tokens and money.

**What this looks like:**
- Hash-based cache keyed on `(prompt, model, schema, options)`
- Pluggable backends: in-memory (dict/LRU), SQLite on disk, Redis for shared environments
- Cache-aware wrappers around `extract_with_model()` and `ask_for_json()`
- TTL support and manual invalidation
- Disabled by default, enabled via `cache=True` or a settings toggle

**Scope:** Medium. Self-contained module that wraps existing functions.

---

## 4. Retry and Resilience

**Why it matters:** LLM APIs fail. Rate limits, transient 500s, malformed responses, and timeouts are routine. Currently, a failed call just raises and the user has to handle retries themselves.

**What this looks like:**
- Configurable retry policy: max attempts, backoff strategy (exponential with jitter)
- Retry on transient HTTP errors (429, 500, 502, 503, 504)
- Retry on malformed JSON responses (re-prompt with stricter instructions)
- Circuit breaker pattern for providers that are consistently down
- Per-driver timeout configuration (some providers are slower than others)
- `ai_cleanup=True` already exists in core -- this extends the idea to the transport layer

**Scope:** Medium. Could be implemented as a decorator/wrapper around `Driver.generate()`.

---

## 5. Structured Output / JSON Mode

**Why it matters:** OpenAI, Google, and Anthropic now offer native JSON mode or structured output guarantees at the API level. This is more reliable than prompt-based schema enforcement and uses fewer tokens.

**What this looks like:**
- Detect when the target provider supports `response_format: { type: "json_object" }` (OpenAI) or equivalent
- Automatically use native JSON mode when available, fall back to prompt-based enforcement otherwise
- For OpenAI's structured outputs (`response_format: { type: "json_schema", json_schema: ... }`), pass the Pydantic schema directly
- Google Gemini has `response_mime_type: "application/json"` with schema support
- Claude has tool-use based structured output

**Scope:** Medium. Per-driver changes with a feature-detection layer in core.

---

## 6. Conversation / Multi-Turn Support

**Why it matters:** All drivers currently operate in single-turn mode (one prompt in, one response out). Multi-turn is needed for complex extraction tasks, clarification loops, and chain-of-thought reasoning.

**What this looks like:**
- Support message arrays (`[{"role": "system", ...}, {"role": "user", ...}]`) in driver `generate()`
- Add a `system_prompt` parameter to extraction functions
- Enable follow-up extraction: "Given your previous response, also extract X"
- Useful for stepwise extraction where context from earlier fields informs later ones

**Scope:** Medium. Requires updating the driver interface and core functions.

---

## 7. Better TOON Output

**Why it matters:** TOON output is marked experimental in the README because general-purpose models tend to emit verbose completions instead of compact TOON notation. This undercuts the token-saving benefit.

**What this looks like:**
- Few-shot examples in the TOON output prompt showing correct compact notation
- Model-specific prompt tuning (some models respond better to certain instruction styles)
- Validation that the output is actually valid TOON before returning
- Benchmarking suite comparing TOON output token counts vs JSON across models
- Consider whether TOON output is worth continuing or if native JSON mode (item 5) makes it obsolete for output

**Scope:** Small-medium. Mostly prompt engineering and testing.

---

## 8. Observability and Logging

**Why it matters:** Prompture has a custom `LogLevel` enum in `tools.py` that isn't stdlib `logging`. Debug output goes through `log_debug()` with a custom level system. This makes it hard to integrate with standard Python logging, APM tools, or structured logging pipelines.

**What this looks like:**
- Migrate to stdlib `logging` throughout (or at minimum, bridge the custom `LogLevel` to stdlib levels)
- Structured log output (JSON lines) for production use
- Request/response logging with configurable verbosity
- Token usage and cost aggregation across a session
- Optional callback hooks for monitoring: `on_request`, `on_response`, `on_error`
- Cost dashboard or summary function: "This session used X tokens across Y calls costing $Z"

**Scope:** Medium. The custom logging is spread across core and tools.

---

## 9. Linting, Formatting, and Type Checking

**Why it matters:** The CLAUDE.md explicitly notes "There is no configured linter or formatter." As the project grows and accepts contributions, inconsistent style and uncaught type errors become friction.

**What this looks like:**
- Add `ruff` for linting and formatting (single tool, fast, replaces flake8+black+isort)
- Add `mypy` or `pyright` for type checking (the codebase already uses type hints extensively)
- `pyproject.toml` configuration for both
- Pre-commit hooks via `pre-commit`
- CI check in GitHub Actions

**Scope:** Small-medium. Configuration plus fixing any issues the tools surface.

---

## 10. Schema Inference from Examples

**Why it matters:** Users currently need to define Pydantic models or JSON schemas upfront. For exploratory work, it would be useful to provide example outputs and have Prompture infer the schema.

**What this looks like:**
- `infer_schema(examples: list[dict]) -> dict` that produces a JSON schema from sample data
- `infer_model(examples: list[dict]) -> Type[BaseModel]` that produces a Pydantic model dynamically
- Use this as a quick-start path: provide examples, get a schema, refine it
- Combine with `extract_with_model()` for a zero-config extraction pipeline

**Scope:** Medium. Standalone utility module.

---

## 11. Batch and Parallel Extraction

**Why it matters:** Processing hundreds or thousands of texts is common (documents, reviews, emails). Currently this requires a manual loop.

**What this looks like:**
- `extract_batch(texts, model, schema)` that processes a list of inputs
- Configurable concurrency (thread pool or async)
- Progress callback or tqdm integration
- Rate limiting per provider (respect API limits)
- Partial results on failure (don't lose 99 results because #100 failed)
- Works with the caching layer (item 3) to skip already-processed inputs

**Scope:** Medium. Builds on async support (item 2) or uses threading.

---

## 12. Plugin / Custom Driver Registration

**Why it matters:** The driver registry is currently hardcoded in `drivers/__init__.py`. Third-party providers or custom endpoints require forking or modifying source.

**What this looks like:**
- `register_driver(name, factory_fn)` public API
- Entry point based discovery (`[project.entry-points."prompture.drivers"]` in pyproject.toml)
- Documentation for writing a custom driver
- Community drivers can be installed as separate packages and auto-discovered

**Scope:** Small-medium. The registry pattern is already there -- just needs a public registration API and entry point scanning.

---

## 13. Expanded Test Coverage

**Why it matters:** 171 tests is solid, but integration tests are skipped by default and require live API access. Some drivers have no dedicated unit tests.

**What this looks like:**
- Mock-based unit tests for each driver (test the request/response handling without live API calls)
- Property-based testing with `hypothesis` for JSON cleaning and type conversion
- Snapshot tests for prompt generation (catch unintended prompt changes)
- Coverage reporting in CI (currently not measured)
- Test matrix across Python 3.9-3.13

**Scope:** Medium. Ongoing effort.

---

## 14. Documentation Site

**Why it matters:** The README is 400+ lines and growing. Examples are in an `examples/` directory with 28 files. There's no searchable, navigable documentation.

**What this looks like:**
- MkDocs or Sphinx site with:
  - Getting started guide
  - Per-driver setup and configuration
  - API reference (auto-generated from docstrings)
  - Field definitions catalog
  - TOON usage guide
  - Migration guide for version upgrades
  - Cookbook with real-world recipes
- Hosted on GitHub Pages or Read the Docs
- The existing `documentation.yml` workflow can be extended

**Scope:** Medium-large. Content already exists scattered across README, examples, and docstrings.

---

## 15. pyproject.toml Migration

**Why it matters:** The project uses `setup.py` which is the legacy build configuration approach. The Python ecosystem has standardized on `pyproject.toml` for build configuration, dependencies, and tool settings.

**What this looks like:**
- Migrate `setup.py` to `[project]` table in `pyproject.toml`
- Move `setuptools_scm` config to `[tool.setuptools_scm]`
- Consolidate tool configs (ruff, mypy, pytest) into the same file
- Remove `setup.py` entirely
- Keep `setuptools` as build backend or consider `hatchling`

**Scope:** Small. Mechanical migration.

---

## 16. Provider-Specific Features

Some providers offer capabilities that the current uniform driver interface doesn't expose:

| Provider | Feature | Value |
|----------|---------|-------|
| OpenAI | Function calling / tool use | Structured extraction via native tools |
| Claude | Extended thinking | Better reasoning for complex extraction |
| Google | Grounding with Search | Extraction with fact-checking |
| Ollama | Model pulling | Auto-download models on first use |
| OpenAI | Batch API | 50% cost reduction for non-urgent work |
| Claude | PDF/image input | Extract from non-text sources |
| Google | Context caching | Reuse large contexts across calls |

**Scope:** Varies per feature. Each is an independent enhancement.

---

## Priority Matrix

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| **High** | Structured Output / JSON Mode | High | Medium |
| **High** | Retry and Resilience | High | Medium |
| **High** | Streaming Support | High | Medium-Large |
| **High** | Linting + Type Checking | Medium | Small |
| **Medium** | Async Driver Support | High | Large |
| **Medium** | Response Caching | Medium | Medium |
| **Medium** | Conversation / Multi-Turn | Medium | Medium |
| **Medium** | Batch Extraction | Medium | Medium |
| **Medium** | pyproject.toml Migration | Low | Small |
| **Lower** | Schema Inference | Medium | Medium |
| **Lower** | Plugin System | Medium | Small-Medium |
| **Lower** | TOON Output Improvements | Low-Medium | Small |
| **Lower** | Observability Overhaul | Medium | Medium |
| **Lower** | Documentation Site | Medium | Medium-Large |
| **Ongoing** | Test Coverage Expansion | Medium | Ongoing |
| **Ongoing** | Provider-Specific Features | Varies | Varies |

---

## Suggested First Moves

If starting tomorrow, the highest-leverage sequence would be:

1. **Linting + pyproject.toml** -- Low effort, cleans up the foundation for everything else
2. **Structured Output / JSON Mode** -- Directly improves the core value proposition (reliable JSON extraction) with less prompt engineering
3. **Retry and Resilience** -- Makes the library production-ready; without this, every user writes their own retry wrapper
4. **Streaming** -- Table stakes for modern LLM tooling; unblocks real-time UX in downstream applications

These four items turn Prompture from a capable dev tool into a production-grade extraction library.
