import pytest
import os
from typing import Dict, Any


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


def get_provider_credentials(provider: str) -> Dict[str, Any]:
    """Get credentials and configuration for a specific provider."""
    credentials = {}
    
    if provider == "ollama":
        # Check for endpoint using both possible env var names
        endpoint = os.getenv("OLLAMA_ENDPOINT") or os.getenv("OLLAMA_URI")
        if endpoint:
            credentials = {
                "endpoint": endpoint,
                "model": os.getenv("OLLAMA_MODEL", "gemma:latest")
            }
    elif provider == "openai":
        if api_key := os.getenv("OPENAI_API_KEY"):
            credentials = {
                "api_key": api_key,
                "model": os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
            }
    elif provider == "claude":
        if api_key := os.getenv("ANTHROPIC_API_KEY"):
            credentials = {
                "api_key": api_key,
                "model": os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
            }
    
    return credentials


@pytest.fixture
def integration_driver(request):
    """Returns a real driver based on AI_PROVIDER environment variable."""
    provider = os.getenv("AI_PROVIDER", "")
    
    if not provider:
        pytest.skip("AI_PROVIDER environment variable not set")
    
    # Convert to lowercase after logging original value
    provider = provider.lower()
            
    credentials = get_provider_credentials(provider)
        
    if provider == "ollama":
        from prompture.drivers import OllamaDriver
        return OllamaDriver(**credentials)
    elif provider == "openai":
        from prompture.drivers import OpenAIDriver
        return OpenAIDriver(**credentials)
    elif provider == "claude":
        from prompture.drivers import ClaudeDriver
        return ClaudeDriver(**credentials)
    else:
        pytest.skip(f"Unsupported provider: {provider}")


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests that use real LLM APIs")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless environment is properly configured."""
    for item in items:
        if "integration" in item.keywords:
            provider = os.getenv("AI_PROVIDER", "")
            
            if not provider:
                item.add_marker(pytest.mark.skip(reason="AI_PROVIDER environment variable not set"))
                continue
            
            # Convert to lowercase after logging
            provider = provider.lower()
            credentials = get_provider_credentials(provider)
            
            if not credentials:
                skip_reason = f"Missing credentials for {provider} provider - check environment variables"
                if provider == "ollama":
                    skip_reason += " (OLLAMA_ENDPOINT or OLLAMA_URI required)"
                item.add_marker(pytest.mark.skip(reason=skip_reason))


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