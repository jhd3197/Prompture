import os
from typing import Any

import pytest
import requests

from prompture.drivers import get_driver_for_model

# Default to running integration tests unless the env var explicitly disables them.
os.environ.setdefault("RUN_INTEGRATION_TESTS", "1")

# Default model configuration for all tests
DEFAULT_MODEL = "ollama/gpt-oss:20b"  # Change this to use a different model


def pytest_addoption(parser):
    """Add CLI flag to opt into slow integration tests."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.integration (requires live LLM access)",
    )


@pytest.fixture
def default_model() -> str:
    """Provide the default model string for integration tests."""
    return DEFAULT_MODEL


@pytest.fixture
def sample_json_schema() -> dict[str, Any]:
    """Sample JSON schema for testing core functions."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "interests": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name"],
    }


@pytest.fixture
def integration_driver(request):
    """Returns a driver instance with default model configuration.

    Skips the test if the driver cannot be created or the backend
    server is unreachable (e.g. Ollama not running).
    """
    try:
        driver = get_driver_for_model(DEFAULT_MODEL)
    except (ValueError, Exception) as e:  # noqa: BLE001
        pytest.skip(f"Could not create driver: {e}")

    # Verify the backend is actually reachable before handing the driver
    # to a test — avoids ConnectionError deep inside test logic.
    endpoint = getattr(driver, "endpoint", None)
    if endpoint:
        try:
            base_url = endpoint.split("/api/")[0] if "/api/" in endpoint else endpoint.rstrip("/")
            requests.head(base_url, timeout=3)
        except requests.exceptions.ConnectionError:
            pytest.skip(f"Backend server not reachable at {base_url}")
        except Exception:  # noqa: BLE001
            pass  # Non-connection errors (e.g. 404) are fine — server is up

    return driver


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: marks tests that use real LLM APIs")


def pytest_collection_modifyitems(config, items):
    """Configure test collection behavior."""
    run_integration = config.getoption("--run-integration") or os.getenv("RUN_INTEGRATION_TESTS", "").lower() in {
        "1",
        "true",
        "yes",
    }

    if run_integration:
        return

    skip_marker = pytest.mark.skip(reason="use --run-integration to run these tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


def assert_valid_usage_metadata(meta: dict[str, Any]):
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
    assert meta["total_tokens"] == meta["prompt_tokens"] + meta["completion_tokens"], (
        "total_tokens should equal prompt + completion"
    )


def assert_jsonify_response_structure(response: dict[str, Any]):
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
