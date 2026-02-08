import pytest

from prompture.extraction.core import (
    extract_and_jsonify,
    extract_with_model,
    manual_extract_and_jsonify,
    stepwise_extract_with_model,
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
    for model in ["ollama/gpt-oss:20b", "ollama/gemma3:latest", "lmstudio/deepseek/deepseek-r1-0528-qwen3-8b"]:
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
    result = extract_and_jsonify(text=SIMPLE_TEXT, json_schema=SIMPLE_SCHEMA, model_name=model)
    assert "json_string" in result
    assert "json_object" in result
    assert "usage" in result
    assert isinstance(result["json_object"], dict)
    assert set(result["json_object"].keys()) >= {"name", "age"}


def test_extract_and_jsonify_invalid_model_string():
    with pytest.raises(ValueError):
        extract_and_jsonify(text=SIMPLE_TEXT, json_schema=SIMPLE_SCHEMA, model_name="invalidmodel")
    with pytest.raises(ValueError):
        extract_and_jsonify(text="", json_schema=SIMPLE_SCHEMA, model_name=SIMPLE_TEXT)
    with pytest.raises(ValueError):
        extract_and_jsonify(text=None, json_schema=SIMPLE_SCHEMA, model_name=SIMPLE_TEXT)


def test_extract_and_jsonify_unsupported_provider():
    with pytest.raises(ValueError):
        extract_and_jsonify(text=SIMPLE_TEXT, json_schema=SIMPLE_SCHEMA, model_name="unknownprovider/model")


@pytest.mark.integration
def test_extract_with_model_and_stepwise_extract_with_model():
    model = get_available_model()
    if not model:
        pytest.skip("No supported provider/model available for testing.")
    from pydantic import BaseModel

    class SimpleModel(BaseModel):
        name: str
        age: int

    result1 = extract_with_model(
        model_cls=SimpleModel,  # BaseModel class defining the schema
        text=SIMPLE_TEXT,  # Text to extract from
        model_name=model,  # Model name like "ollama/mistral"
    )
    assert "json_string" in result1
    assert "json_object" in result1
    assert "usage" in result1

    result2 = stepwise_extract_with_model(model_cls=SimpleModel, text=SIMPLE_TEXT, model_name=model)
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
