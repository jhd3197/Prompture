import pytest
from typing import Dict, Any
from prompture.drivers import get_driver_for_model

# Default model configuration for all tests
DEFAULT_MODEL = "ollama/gpt-oss:20b"  # Change this to use a different model

@pytest.fixture
def sample_json_schema() -> Dict[str, Any]:
    """Sample JSON schema for testing core functions."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "interests": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name"]
    }


@pytest.fixture
def integration_driver(request):
    """Returns a driver instance with default model configuration."""
    try:
        return get_driver_for_model(DEFAULT_MODEL)
    except ValueError as e:
        pytest.skip(str(e))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests that use real LLM APIs")


def pytest_collection_modifyitems(config, items):
    """Configure test collection behavior."""
    # Integration tests are controlled by the TEST_SKIP_NO_CREDENTIALS env var
    # in test.py, no additional configuration needed here
    pass


def assert_valid_usage_metadata(meta: Dict[str, Any]):
    """Helper function to validate usage metadata structure."""
    required_keys = {"prompt_tokens", "completion_tokens", "total_tokens", "cost", "raw_response"}

    for key in required_keys:
        assert key in meta, f"Missing required metadata key: {key}"

    # Validate types
    assert isinstance(meta["prompt_tokens"], int), "prompt_tokens must be int"
    assert isinstance(meta["completion_tokens"], int), "completion_tokens must be int"
    assert isinstance(meta["total_tokens"], int), "total_tokens must be int"
    assert isinstance(meta["cost"], (int, float)), "cost must be numeric"
    assert isinstance(meta["raw_response"], dict), "raw_response must be dict"

    # Validate reasonable totals
    assert meta["total_tokens"] == meta["prompt_tokens"] + meta["completion_tokens"], "total_tokens should equal prompt + completion"


def assert_jsonify_response_structure(response: Dict[str, Any]):
    """Helper function to validate the structure of jsonify responses."""
    required_keys = {"json_string", "json_object", "usage"}

    for key in required_keys:
        assert key in response, f"Missing required response key: {key}"

    # Validate types
    assert isinstance(response["json_string"], str), "json_string must be string"
    assert isinstance(response["json_object"], dict), "json_object must be dict"
    assert isinstance(response["usage"], dict), "usage must be dict"

    # Validate usage metadata
    assert_valid_usage_metadata(response["usage"])


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory path for test data files."""
    return "tests/data"