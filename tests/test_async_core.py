"""Tests for async core extraction functions."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from prompture.async_core import (
    ask_for_json,
    clean_json_text_with_ai,
    extract_and_jsonify,
    extract_with_model,
    gather_extract,
    manual_extract_and_jsonify,
    render_output,
    stepwise_extract_with_model,
)
from prompture.async_driver import AsyncDriver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_driver_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    return {
        "text": text,
        "meta": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": 0.001,
            "raw_response": {},
        },
    }


def _mock_async_driver(text: str = '{"name": "Alice", "age": 30}') -> AsyncDriver:
    driver = AsyncMock(spec=AsyncDriver)
    driver.generate = AsyncMock(return_value=_make_driver_response(text))
    driver.model = "test-model"
    return driver


# ---------------------------------------------------------------------------
# clean_json_text_with_ai
# ---------------------------------------------------------------------------


class TestAsyncCleanJsonTextWithAI:
    async def test_already_valid_json_returns_unchanged(self):
        driver = _mock_async_driver()
        result = await clean_json_text_with_ai(driver, '{"key": "value"}')
        assert result == '{"key": "value"}'
        driver.generate.assert_not_called()

    async def test_malformed_json_calls_driver(self):
        driver = _mock_async_driver(text='{"key": "fixed"}')
        result = await clean_json_text_with_ai(driver, '{"key": broken}')
        driver.generate.assert_called_once()
        assert json.loads(result) == {"key": "fixed"}


# ---------------------------------------------------------------------------
# render_output
# ---------------------------------------------------------------------------


class TestAsyncRenderOutput:
    async def test_text_output(self):
        driver = _mock_async_driver(text="Hello world")
        result = await render_output(driver, "Say hello", output_format="text")
        assert result["text"] == "Hello world"
        assert result["output_format"] == "text"
        assert "usage" in result

    async def test_invalid_format_raises(self):
        driver = _mock_async_driver()
        with pytest.raises(ValueError, match="Unsupported output_format"):
            await render_output(driver, "test", output_format="xml")


# ---------------------------------------------------------------------------
# ask_for_json
# ---------------------------------------------------------------------------


class TestAsyncAskForJson:
    async def test_valid_json_response(self):
        driver = _mock_async_driver()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await ask_for_json(driver, "Extract name", schema)
        assert "json_object" in result
        assert result["json_object"]["name"] == "Alice"
        assert result["output_format"] == "json"

    async def test_ai_cleanup_on_malformed_json(self):
        driver = AsyncMock(spec=AsyncDriver)
        driver.model = "test"
        # First call returns malformed JSON, second call (cleanup) returns valid
        driver.generate = AsyncMock(
            side_effect=[
                _make_driver_response("{name: Alice}"),
                _make_driver_response('{"name": "Alice"}'),
            ]
        )
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await ask_for_json(driver, "Extract", schema, ai_cleanup=True)
        assert result["json_object"]["name"] == "Alice"

    async def test_invalid_output_format_raises(self):
        driver = _mock_async_driver()
        with pytest.raises(ValueError, match="Unsupported output_format"):
            await ask_for_json(driver, "test", {}, output_format="xml")


# ---------------------------------------------------------------------------
# extract_and_jsonify
# ---------------------------------------------------------------------------


class TestAsyncExtractAndJsonify:
    async def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            await extract_and_jsonify(text="", json_schema={}, model_name="openai/gpt-4")

    async def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            await extract_and_jsonify(text="some text", json_schema={}, model_name="")

    async def test_invalid_model_format_raises(self):
        with pytest.raises(ValueError, match="Expected format"):
            await extract_and_jsonify(text="some text", json_schema={}, model_name="invalid")

    @patch("prompture.async_core.get_async_driver_for_model")
    async def test_successful_extraction(self, mock_get_driver):
        driver = _mock_async_driver()
        mock_get_driver.return_value = driver
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await extract_and_jsonify(
            text="Alice is 30 years old",
            json_schema=schema,
            model_name="openai/gpt-4",
        )
        assert result["json_object"]["name"] == "Alice"


# ---------------------------------------------------------------------------
# manual_extract_and_jsonify
# ---------------------------------------------------------------------------


class TestAsyncManualExtractAndJsonify:
    async def test_empty_text_raises(self):
        driver = _mock_async_driver()
        with pytest.raises(ValueError, match="cannot be empty"):
            await manual_extract_and_jsonify(driver, "", {})

    async def test_successful_extraction(self):
        driver = _mock_async_driver()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await manual_extract_and_jsonify(driver, "Alice is 30", schema)
        assert result["json_object"]["name"] == "Alice"


# ---------------------------------------------------------------------------
# extract_with_model
# ---------------------------------------------------------------------------


class TestAsyncExtractWithModel:
    async def test_empty_text_raises(self):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        with pytest.raises(ValueError, match="cannot be empty"):
            await extract_with_model(Person, "", "openai/gpt-4")

    @patch("prompture.async_core.get_async_driver_for_model")
    async def test_returns_model_instance(self, mock_get_driver):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int = 0

        driver = _mock_async_driver()
        mock_get_driver.return_value = driver

        result = await extract_with_model(Person, "Alice is 30", "openai/gpt-4")
        assert result["model"].name == "Alice"
        assert isinstance(result["json_object"], dict)


# ---------------------------------------------------------------------------
# stepwise_extract_with_model
# ---------------------------------------------------------------------------


class TestAsyncStepwiseExtractWithModel:
    async def test_empty_text_raises(self):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        with pytest.raises(ValueError, match="cannot be empty"):
            await stepwise_extract_with_model(Person, "", model_name="openai/gpt-4")

    @patch("prompture.async_core.get_async_driver_for_model")
    async def test_extracts_fields_sequentially(self, mock_get_driver):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str = "unknown"
            age: int = 0

        driver = AsyncMock(spec=AsyncDriver)
        driver.model = "test"
        driver.generate = AsyncMock(
            side_effect=[
                _make_driver_response('{"value": "Alice"}'),
                _make_driver_response('{"value": 30}'),
            ]
        )
        mock_get_driver.return_value = driver

        result = await stepwise_extract_with_model(
            Person,
            "Alice is 30 years old",
            model_name="openai/gpt-4",
        )
        assert result["model"].name == "Alice"


# ---------------------------------------------------------------------------
# gather_extract
# ---------------------------------------------------------------------------


class TestGatherExtract:
    @patch("prompture.async_core.get_async_driver_for_model")
    async def test_concurrent_extraction(self, mock_get_driver):
        driver = _mock_async_driver()
        mock_get_driver.return_value = driver

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        results = await gather_extract(
            "Alice is 30",
            schema,
            ["openai/gpt-4", "openai/gpt-4o"],
        )
        assert len(results) == 2
        for r in results:
            assert "json_object" in r
