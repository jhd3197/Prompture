import pytest
import os
from typing import Dict, Any
from prompture.drivers import MockDriver


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
def mock_driver_with_metadata():
    """Mock driver that returns realistic token usage and cost metadata."""
    class MockDriverWithMetadata(MockDriver):
        def generate(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
            # Base response from parent
            response = super().generate(prompt, options or {})

            # Enhanced metadata with token usage and cost tracking
            prompt_tokens = len(prompt.split()) * 2  # Rough estimate
            completion_tokens = len(response["text"].split()) * 2

            response["meta"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": 0.0,  # Mock is free
                "raw_response": {
                    "mock_response": True,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
            }

            return response

    return MockDriverWithMetadata()


@pytest.fixture
def integration_driver():
    """Returns a real driver if environment variables are set, otherwise MockDriver."""
    # Check for Ollama
    if os.getenv("OLLAMA_URI"):
        from prompture.drivers import OllamaDriver
        return OllamaDriver(
            endpoint=os.getenv("OLLAMA_URI"),
            model=os.getenv("OLLAMA_MODEL", "gemma3:latest")
        )

    # Check for OpenAI
    if os.getenv("OPENAI_API_KEY"):
        from prompture.drivers import OpenAIDriver
        return OpenAIDriver(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )

    # Fallback to mock
    return MockDriver()


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests that use real LLM APIs")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests by default unless environment is configured."""
    skip_integration = pytest.mark.skip(reason="integration test - requires API credentials")

    for item in items:
        if "integration" in item.keywords:
            # Skip if no integration credentials are available
            has_ollama = bool(os.getenv("OLLAMA_URI"))
            has_openai = bool(os.getenv("OPENAI_API_KEY"))

            if not (has_ollama or has_openai):
                item.add_marker(skip_integration)


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