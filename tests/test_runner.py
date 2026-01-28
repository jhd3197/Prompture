import os

import pytest

from prompture import run_suite_from_spec


@pytest.mark.integration
def test_run_suite_from_spec(integration_driver):
    """Test running a test suite with a real LLM driver."""
    provider = os.getenv("AI_PROVIDER", "").lower()
    if not provider:
        pytest.skip("AI_PROVIDER environment variable not set")

    # Test spec with real model and simple user info extraction test
    spec = {
        "meta": {"project": "test"},
        "models": [{"id": "test-model", "driver": provider, "options": integration_driver.options}],
        "tests": [
            {
                "id": "t1",
                "prompt_template": "Extract user info: '{text}'",
                "inputs": [{"text": "Juan is 28 and lives in Miami. He likes basketball and coding."}],
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                        "location": {"type": "string"},
                        "interests": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "interests"],
                },
            }
        ],
    }

    # Setup drivers with real driver
    drivers = {provider: integration_driver}

    # Run the test suite
    report = run_suite_from_spec(spec, drivers)

    # Verify report structure and content
    assert report["meta"]["project"] == "test"
    assert len(report["results"]) == 1

    result = report["results"][0]
    assert result["test_id"] == "t1"
    assert result["model_id"] == "test-model"
    assert result["validation"]["ok"] is True

    # Verify response structure
    response = result["response"]
    assert "name" in response
    assert "interests" in response
    assert isinstance(response["name"], str)
    assert isinstance(response["interests"], list)
    assert len(response["interests"]) > 0
