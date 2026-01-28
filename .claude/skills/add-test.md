# Skill: Add Tests

When the user asks to add tests for new or existing functionality, follow these conventions.

## Test Infrastructure

- Framework: `pytest`
- Test directory: `tests/`
- Shared fixtures and helpers: `tests/conftest.py`
- Default test model: `DEFAULT_MODEL` from `conftest.py` (currently `"ollama/gpt-oss:20b"`)
- Integration marker: `@pytest.mark.integration` (skipped by default, run with `--run-integration`)

## Key Fixtures and Helpers (from conftest.py)

```python
# Fixtures
sample_json_schema      # Standard {"name": str, "age": int, "interests": list} schema
integration_driver      # Driver instance from DEFAULT_MODEL (skips if unavailable)

# Assertion helpers
assert_valid_usage_metadata(meta)          # Validates prompt_tokens, completion_tokens, total_tokens, cost, raw_response
assert_jsonify_response_structure(response) # Validates json_string, json_object, usage keys
```

## Test File Naming

- `tests/test_{module}.py` — maps to source module (e.g. `test_core.py`, `test_field_definitions.py`)
- `tests/test_{feature}.py` — for cross-cutting features (e.g. `test_toon_input.py`, `test_enhanced_extraction.py`)

## Test Class and Function Conventions

```python
import pytest
from prompture import extract_with_model, get_driver_for_model
from tests.conftest import DEFAULT_MODEL  # if needed


class TestFeatureName:
    """Tests for {feature description}."""

    def test_basic_behavior(self):
        """What the test verifies in plain English."""
        result = some_function(...)
        assert result["key"] == expected

    def test_edge_case(self):
        """Edge case: empty input."""
        ...

    def test_error_handling(self):
        """Should raise ValueError on invalid input."""
        with pytest.raises(ValueError, match="expected message"):
            some_function(bad_input)


class TestFeatureIntegration:
    """Integration tests requiring live LLM access."""

    @pytest.mark.integration
    def test_live_extraction(self, integration_driver, sample_json_schema):
        """End-to-end extraction with live model."""
        result = extract_and_jsonify(
            text="John is 30 years old",
            json_schema=sample_json_schema,
            model_name=DEFAULT_MODEL,
        )
        assert_jsonify_response_structure(result)
        assert_valid_usage_metadata(result["usage"])
```

## Rules

- **Unit tests** (no LLM call): No marker needed, should always pass
- **Integration tests** (live LLM call): Must have `@pytest.mark.integration`
- Use `conftest.py` fixtures and helpers — don't redefine them
- One test class per logical group, clear docstrings
- Test both happy path and error cases
- For driver tests, mock the HTTP layer (use `unittest.mock.patch` or `responses` library) for unit tests

## Running

```bash
pytest tests/                           # Unit tests only
pytest tests/ --run-integration         # Include integration tests
pytest tests/test_core.py -x -q         # Single file, stop on first failure
pytest tests/test_core.py::TestClass::test_method  # Single test
```
