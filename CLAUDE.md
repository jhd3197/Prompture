# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Prompture is a Python library for extracting structured JSON output from LLMs, with schema enforcement, Pydantic model integration, TOON (Token-Oriented Object Notation) input/output conversion, and multi-provider driver support. Published on PyPI as `prompture`.

## Build & Development Commands

```bash
# Install in development mode
pip install -e ".[test,dev]"

# Run all tests (uses DEFAULT_MODEL from tests/conftest.py)
python test.py

# Run tests with pytest directly
pytest

# Run integration tests (require live LLM access)
pytest --run-integration
# or: RUN_INTEGRATION_TESTS=1 pytest

# Run a single test file
pytest tests/test_core.py

# Run a single test
pytest tests/test_core.py::TestCleanJsonText::test_basic_json

# Skip integration tests when credentials are missing
TEST_SKIP_NO_CREDENTIALS=true python test.py

# Build distribution
python -m build

# CLI entry point
prompture run <spec-file>
```

### Linting & Formatting (ruff)

```bash
# Format all files
ruff format .

# Check for lint violations
ruff check .

# Auto-fix lint violations
ruff check --fix .

# Auto-fix including unsafe fixes (e.g., typing modernization)
ruff check --fix --unsafe-fixes .
```

Configuration is in `pyproject.toml` under `[tool.ruff]`.

## Architecture

### Module Layout

- **`prompture/core.py`** — Primary business logic. All extraction functions live here: `ask_for_json()` (low-level schema enforcement), `extract_and_jsonify()` / `manual_extract_and_jsonify()` (text-to-JSON), `extract_with_model()` / `stepwise_extract_with_model()` (Pydantic-based), `extract_from_data()` / `extract_from_pandas()` (TOON input), `render_output()` (raw text formatting).
- **`prompture/drivers/`** — One module per LLM provider (openai, claude, google, groq, grok, azure, ollama, lmstudio, openrouter, local_http, huggingface, airllm). Each driver implements `generate(prompt, options)` returning a standardized response with token/cost metadata.
- **`prompture/drivers/__init__.py`** — Central `DRIVER_REGISTRY` dict mapping provider name to factory lambda. `get_driver_for_model("provider/model")` parses the string and instantiates the right driver. `get_driver("provider")` is the legacy interface.
- **`prompture/tools.py`** — Utilities: JSON/TOON text cleanup, type conversion (shorthand numbers, multilingual booleans, datetimes), field schema generation.
- **`prompture/logging.py`** — Library logging configuration: `JSONFormatter` for structured JSON-lines output, `configure_logging()` for enabling library-level logging via Python stdlib `logging`.
- **`prompture/callbacks.py`** — `DriverCallbacks` dataclass with `on_request`, `on_response`, `on_error` hooks for driver-level observability.
- **`prompture/session.py`** — `UsageSession` dataclass for tracking token counts, costs, and errors across multiple driver calls.
- **`prompture/field_definitions.py`** — Thread-safe global field registry with 50+ predefined fields, template variable substitution (`{{current_year}}`, `{{current_date}}`), and Pydantic Field generation via `field_from_registry()`.
- **`prompture/settings.py`** — Pydantic-settings `Settings` class loading provider API keys/endpoints from `.env`.
- **`prompture/discovery.py`** — `get_available_models()` auto-detects models from configured providers (static pricing tables + dynamic Ollama endpoint query).
- **`prompture/runner.py`** — Spec-driven test suite runner for cross-model comparison.
- **`prompture/validator.py`** — JSON schema validation via jsonschema with fallback.

### Key Patterns

- **Model strings** use `"provider/model"` format (e.g., `"ollama/llama3.1:8b"`, `"openai/gpt-4"`). The provider prefix routes to the correct driver.
- **Driver responses** always include metadata: `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost`, `raw_response`.
- **Output formats**: JSON (default) and TOON (experimental, for compact output). Controlled via `output_format` parameter.
- **TOON input conversion** uses `python-toon` and `tukuy` packages to reduce token usage by 45-60% when sending structured data to LLMs.

### Testing

Tests live in `tests/`. Integration tests that call live LLMs are marked with `@pytest.mark.integration` and skipped by default. The default test model is set in `tests/conftest.py` as `DEFAULT_MODEL`. Shared fixtures and assertion helpers (`assert_valid_usage_metadata`, `assert_jsonify_response_structure`) are in `conftest.py`.

### Configuration

Provider API keys and endpoints are configured via environment variables loaded from `.env` (see `.env.copy` for the template). The `Settings` class in `settings.py` manages all provider config.

### Versioning

Uses `setuptools_scm` for automatic version from git tags. The `VERSION` file contains the current dev version.
