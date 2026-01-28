---
name: add-test
description: Add unit and integration tests for Prompture functionality. Uses pytest conventions, shared fixtures from conftest.py, and the integration marker pattern. Use when writing tests for new or existing features.
metadata:
  author: prompture
  version: "1.0"
---

# Add Tests

Creates tests in `tests/` following project conventions.

## Infrastructure

- Framework: **pytest**
- Shared fixtures: `tests/conftest.py`
- Default model: `DEFAULT_MODEL` (currently `"ollama/gpt-oss:20b"`)
- Integration marker: `@pytest.mark.integration` (skipped by default)

## Available Fixtures and Helpers

```python
# Fixtures
sample_json_schema       # {"name": str, "age": int, "interests": list}
integration_driver       # Driver from DEFAULT_MODEL (skips if unavailable)

# Assertion helpers
assert_valid_usage_metadata(meta)           # Checks prompt_tokens, completion_tokens, total_tokens, cost, raw_response
assert_jsonify_response_structure(response)  # Checks json_string, json_object, usage
```

## File Naming

- `tests/test_{module}.py` — maps to source module
- `tests/test_{feature}.py` — cross-cutting features

## Test Structure

```python
import pytest
from prompture import extract_with_model


class TestFeatureName:
    """Tests for {feature}."""

    def test_basic_behavior(self):
        """What this test verifies."""
        result = some_function(...)
        assert result["key"] == expected

    def test_error_handling(self):
        """Should raise ValueError on invalid input."""
        with pytest.raises(ValueError, match="expected"):
            some_function(bad_input)


class TestFeatureIntegration:
    """Integration tests requiring live LLM access."""

    @pytest.mark.integration
    def test_live_extraction(self, integration_driver, sample_json_schema):
        result = extract_and_jsonify(
            text="John is 30 years old",
            json_schema=sample_json_schema,
            model_name=DEFAULT_MODEL,
        )
        assert_jsonify_response_structure(result)
        assert_valid_usage_metadata(result["usage"])
```

## Rules

- **Unit tests** (no LLM): no marker, must always pass
- **Integration tests** (live LLM): must have `@pytest.mark.integration`
- Use conftest fixtures — don't redefine them
- One test class per logical group with docstrings
- Test both happy path and error cases
- Mock HTTP layer for driver unit tests

## Running

```bash
pytest tests/ -x -q                          # Unit only
pytest tests/ --run-integration -x -q        # Include integration
pytest tests/test_core.py::TestClass::test_method  # Single test
```
