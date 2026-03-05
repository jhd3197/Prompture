"""Tests for extract_with_models() — multi-model cascade extraction."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from prompture.exceptions import ExtractionError
from prompture.extraction.core import ExtractResult, extract_with_models


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------
class PersonModel(BaseModel):
    name: str
    age: int
    city: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_driver(*, json_mode: bool = True, json_schema: bool = True):
    """Create a mock driver with given capability flags."""
    driver = MagicMock()
    driver.supports_json_mode = json_mode
    driver.supports_json_schema = json_schema
    return driver


def _make_success_result(model_cls=PersonModel, data=None):
    """Create a successful ExtractResult."""
    data = data or {"name": "Juan", "age": 30, "city": "Lima"}
    import json

    return ExtractResult({
        "json_string": json.dumps(data),
        "json_object": data,
        "model": model_cls(**data),
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.001,
            "model_name": "test/model",
        },
    })


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------
class TestExtractWithModelsValidation:
    def test_empty_models_list_raises(self):
        with pytest.raises(ValueError, match="models list cannot be empty"):
            extract_with_models(PersonModel, "some text", models=[])

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="Text input cannot be empty"):
            extract_with_models(PersonModel, "", models=["openai/gpt-4"])

    def test_whitespace_text_raises(self):
        with pytest.raises(ValueError, match="Text input cannot be empty"):
            extract_with_models(PersonModel, "   \n  ", models=["openai/gpt-4"])


# ---------------------------------------------------------------------------
# Driver resolution tests
# ---------------------------------------------------------------------------
class TestDriverResolution:
    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_skips_model_when_driver_fails(self, mock_extract, mock_get_driver):
        """Model is skipped (not failed) when driver can't be created."""
        mock_get_driver.side_effect = [
            ValueError("No API key"),  # First model fails
            _make_driver(),  # Second model succeeds
        ]
        mock_extract.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel,
            "Juan is 30 from Lima",
            models=["bad/model", "openai/gpt-4"],
        )

        assert result["selected_model"] == "openai/gpt-4"
        assert len(result["attempts"]) == 2
        assert result["attempts"][0]["status"] == "skipped"
        assert "No API key" in result["attempts"][0]["reason"]
        assert result["attempts"][1]["status"] == "success"

    @patch("prompture.extraction.core.get_driver_for_model")
    def test_all_drivers_fail_raises(self, mock_get_driver):
        """When all drivers fail to resolve, raises ExtractionError."""
        mock_get_driver.side_effect = ValueError("No API key")

        with pytest.raises(ExtractionError) as exc_info:
            extract_with_models(
                PersonModel,
                "some text",
                models=["bad/one", "bad/two"],
            )
        assert hasattr(exc_info.value, "attempts")


# ---------------------------------------------------------------------------
# Strategy selection tests
# ---------------------------------------------------------------------------
class TestStrategySelection:
    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_uses_single_when_json_mode_supported(self, mock_extract, mock_get_driver):
        mock_get_driver.return_value = _make_driver(json_mode=True, json_schema=True)
        mock_extract.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel, "Juan is 30", models=["openai/gpt-4"]
        )

        assert result["strategy_used"] == "single"
        mock_extract.assert_called_once()

    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_uses_single_when_json_mode_only(self, mock_extract, mock_get_driver):
        """json_mode without json_schema still uses single strategy."""
        mock_get_driver.return_value = _make_driver(json_mode=True, json_schema=False)
        mock_extract.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel, "Juan is 30", models=["groq/llama"]
        )

        assert result["strategy_used"] == "single"

    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.stepwise_extract_with_model")
    def test_uses_stepwise_when_no_json_support(self, mock_stepwise, mock_get_driver):
        mock_get_driver.return_value = _make_driver(json_mode=False, json_schema=False)
        mock_stepwise.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel, "Juan is 30", models=["weak/model"]
        )

        assert result["strategy_used"] == "stepwise"
        mock_stepwise.assert_called_once()


# ---------------------------------------------------------------------------
# Cascade / fallthrough tests
# ---------------------------------------------------------------------------
class TestCascade:
    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_falls_through_on_failure(self, mock_extract, mock_get_driver):
        mock_get_driver.return_value = _make_driver()
        mock_extract.side_effect = [
            RuntimeError("API timeout"),
            _make_success_result(),
        ]

        result = extract_with_models(
            PersonModel,
            "Juan is 30 from Lima",
            models=["openai/gpt-4", "claude/sonnet"],
        )

        assert result["selected_model"] == "claude/sonnet"
        assert len(result["attempts"]) == 2
        assert result["attempts"][0]["status"] == "failed"
        assert result["attempts"][0]["reason"] == "API timeout"
        assert result["attempts"][1]["status"] == "success"

    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_first_success_wins(self, mock_extract, mock_get_driver):
        """Should not try remaining models after first success."""
        mock_get_driver.return_value = _make_driver()
        mock_extract.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel,
            "Juan is 30",
            models=["openai/gpt-4", "claude/sonnet", "groq/llama"],
        )

        assert result["selected_model"] == "openai/gpt-4"
        assert len(result["attempts"]) == 1
        mock_extract.assert_called_once()


# ---------------------------------------------------------------------------
# Tracking / observability tests
# ---------------------------------------------------------------------------
class TestTracking:
    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_total_cost_accumulates_across_attempts(self, mock_extract, mock_get_driver):
        mock_get_driver.return_value = _make_driver()

        # First call fails but has no cost (exception before response)
        # Second call succeeds with cost
        mock_extract.side_effect = [
            RuntimeError("timeout"),
            _make_success_result(),
        ]

        result = extract_with_models(
            PersonModel,
            "Juan is 30",
            models=["model/a", "model/b"],
        )

        assert result["total_cost"] == 0.001  # only successful attempt had cost
        assert result["total_tokens"] == 150
        assert result["total_attempts"] == 2  # both were actually called

    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_attempt_records_capabilities(self, mock_extract, mock_get_driver):
        mock_get_driver.return_value = _make_driver(json_mode=True, json_schema=False)
        mock_extract.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel, "Juan is 30", models=["groq/llama"]
        )

        caps = result["attempts"][0]["capabilities"]
        assert caps["json_mode"] is True
        assert caps["json_schema"] is False

    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_attempt_records_duration(self, mock_extract, mock_get_driver):
        mock_get_driver.return_value = _make_driver()
        mock_extract.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel, "Juan is 30", models=["openai/gpt-4"]
        )

        assert "duration_ms" in result["attempts"][0]
        assert isinstance(result["attempts"][0]["duration_ms"], float)

    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_result_has_all_tracking_keys(self, mock_extract, mock_get_driver):
        mock_get_driver.return_value = _make_driver()
        mock_extract.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel, "Juan is 30", models=["openai/gpt-4"]
        )

        assert "model" in result
        assert "selected_model" in result
        assert "strategy_used" in result
        assert "attempts" in result
        assert "total_cost" in result
        assert "total_tokens" in result
        assert "total_attempts" in result
        assert "usage" in result
        assert "json_string" in result
        assert "json_object" in result

    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_result_is_extract_result(self, mock_extract, mock_get_driver):
        """Result should be an ExtractResult with attribute access."""
        mock_get_driver.return_value = _make_driver()
        mock_extract.return_value = _make_success_result()

        result = extract_with_models(
            PersonModel, "Juan is 30", models=["openai/gpt-4"]
        )

        assert isinstance(result, ExtractResult)
        assert result.selected_model == "openai/gpt-4"
        assert result() is not None  # callable returns model


# ---------------------------------------------------------------------------
# Fallback tests
# ---------------------------------------------------------------------------
class TestFallback:
    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_fallback_used_when_all_fail(self, mock_extract, mock_get_driver):
        mock_get_driver.return_value = _make_driver()
        mock_extract.side_effect = RuntimeError("fail")

        fallback_person = PersonModel(name="Unknown", age=0)
        result = extract_with_models(
            PersonModel,
            "some text",
            models=["model/a", "model/b"],
            fallback=fallback_person,
        )

        assert result["model"].name == "Unknown"
        assert result["selected_model"] is None
        assert result["strategy_used"] is None
        assert result["usage"]["fallback_used"] is True
        assert len(result["attempts"]) == 2

    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.extract_with_model")
    def test_no_fallback_raises_with_tracking(self, mock_extract, mock_get_driver):
        mock_get_driver.return_value = _make_driver()
        mock_extract.side_effect = RuntimeError("fail")

        with pytest.raises(ExtractionError) as exc_info:
            extract_with_models(
                PersonModel,
                "some text",
                models=["model/a", "model/b"],
            )

        err = exc_info.value
        assert hasattr(err, "attempts")
        assert len(err.attempts) == 2
        assert hasattr(err, "total_cost")
        assert hasattr(err, "total_tokens")


# ---------------------------------------------------------------------------
# Field results passthrough
# ---------------------------------------------------------------------------
class TestFieldResults:
    @patch("prompture.extraction.core.get_driver_for_model")
    @patch("prompture.extraction.core.stepwise_extract_with_model")
    def test_stepwise_field_results_passed_through(self, mock_stepwise, mock_get_driver):
        mock_get_driver.return_value = _make_driver(json_mode=False, json_schema=False)
        stepwise_result = _make_success_result()
        stepwise_result["field_results"] = {
            "name": {"status": "success"},
            "age": {"status": "success"},
        }
        mock_stepwise.return_value = stepwise_result

        result = extract_with_models(
            PersonModel, "Juan is 30", models=["weak/model"]
        )

        assert "field_results" in result
        assert result["field_results"]["name"]["status"] == "success"


# ---------------------------------------------------------------------------
# Async version
# ---------------------------------------------------------------------------
class TestAsyncExtractWithModels:
    @pytest.mark.asyncio
    async def test_async_empty_models_raises(self):
        from prompture.extraction.async_core import (
            extract_with_models as async_extract_with_models,
        )

        with pytest.raises(ValueError, match="models list cannot be empty"):
            await async_extract_with_models(PersonModel, "text", models=[])

    @pytest.mark.asyncio
    async def test_async_empty_text_raises(self):
        from prompture.extraction.async_core import (
            extract_with_models as async_extract_with_models,
        )

        with pytest.raises(ValueError, match="Text input cannot be empty"):
            await async_extract_with_models(PersonModel, "", models=["openai/gpt-4"])
