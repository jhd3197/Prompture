import pytest
import warnings
import os

from prompture.core import (
    extract_and_jsonify,
    extract_with_model,
    stepwise_extract_with_model,
    manual_extract_and_jsonify,
)
from prompture.drivers import get_driver_for_model

# Simple schema for extraction
SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

SIMPLE_TEXT = "Juan is 28 years old."

def get_available_model():
    # Try common models, fallback to None if not configured
    for model in [
        "ollama/llama3",
        "openai/gpt-4-turbo-preview",
        "openai/gpt-3.5-turbo",
        "claude/claude-3-opus-20240229",
        "hugging/mistral-7b-instruct-v0.2"
    ]:
        try:
            get_driver_for_model(model)
            return model
        except Exception:
            continue
    return None

@pytest.mark.integration
def test_extract_and_jsonify_autopick_success():
    model = get_available_model()
    if not model:
        pytest.skip("No supported provider/model available for testing.")
    result = extract_and_jsonify(model, SIMPLE_TEXT, SIMPLE_SCHEMA)
    assert "json_string" in result
    assert "json_object" in result
    assert "usage" in result
    assert isinstance(result["json_object"], dict)
    assert set(result["json_object"].keys()) >= {"name", "age"}

def test_extract_and_jsonify_invalid_model_string():
    with pytest.raises(ValueError):
        extract_and_jsonify("invalidmodel", SIMPLE_TEXT, SIMPLE_SCHEMA)
    with pytest.raises(ValueError):
        extract_and_jsonify("", SIMPLE_TEXT, SIMPLE_SCHEMA)
    with pytest.raises(ValueError):
        extract_and_jsonify(None, SIMPLE_TEXT, SIMPLE_SCHEMA)

def test_extract_and_jsonify_unsupported_provider():
    with pytest.raises(ValueError):
        extract_and_jsonify("unknownprovider/model", SIMPLE_TEXT, SIMPLE_SCHEMA)

@pytest.mark.integration
def test_extract_with_model_and_stepwise_extract_with_model():
    model = get_available_model()
    if not model:
        pytest.skip("No supported provider/model available for testing.")
    result1 = extract_with_model(model, SIMPLE_TEXT, SIMPLE_SCHEMA)
    assert "json_string" in result1
    assert "json_object" in result1
    assert "usage" in result1

    result2 = stepwise_extract_with_model(model, SIMPLE_TEXT, SIMPLE_SCHEMA)
    assert "json_string" in result2
    assert "json_object" in result2
    assert "usage" in result2

def test_manual_extract_and_jsonify_explicit_driver():
    # Use OpenAI as an example, skip if not configured
    try:
        driver = get_driver_for_model("openai/gpt-3.5-turbo")
    except Exception:
        pytest.skip("OpenAI driver not available.")
    result = manual_extract_and_jsonify(driver, SIMPLE_TEXT, SIMPLE_SCHEMA)
    assert "json_string" in result
    assert "json_object" in result
    assert "usage" in result

def test_deprecation_warning_for_old_extract_and_jsonify():
    try:
        from prompture.core import _old_extract_and_jsonify
    except ImportError:
        pytest.skip("Old extract_and_jsonify not available.")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        driver = get_driver_for_model("openai/gpt-3.5-turbo")
        _old_extract_and_jsonify(driver, SIMPLE_TEXT, SIMPLE_SCHEMA)
        assert any("deprecated" in str(warn.message).lower() for warn in w)