"""Outcome reports group actual calls across validation retries and model fallback."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from prompture.drivers.base import Driver
from prompture.infra.tracker import UsageTracker


class Output(BaseModel):
    value: str = Field(min_length=3)


class ExtractDriver(Driver):
    supports_json_mode = True

    def __init__(self, values):
        self.values = iter(values)
        self.model = "openai/test"

    def generate(self, prompt, options):
        return {
            "text": next(self.values),
            "meta": {
                "cost": 1.0,
                "cost_status": "estimated",
                "usage_complete": True,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        }


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    tracker = UsageTracker(tmp_path / "usage.db")
    monkeypatch.setattr("prompture.infra.tracker.get_tracker", lambda: tracker)
    monkeypatch.setattr("prompture.extraction._usage.get_tracker", lambda: tracker)
    return tracker


def test_success_after_validation_retry(ledger, monkeypatch):
    from prompture.extraction.core import extract_with_model

    driver = ExtractDriver(['{"value":"x"}', '{"value":"good"}'])
    monkeypatch.setattr("prompture.extraction.core.get_driver_for_model", lambda *a, **kw: driver)
    result = extract_with_model(
        Output,
        "extract",
        model_name="openai/test",
        max_retries=2,
        cache=False,
        strategy="prompted_repair",
        ai_cleanup=False,
    )
    report = ledger.efficiency_report()
    assert result["model"].value == "good"
    assert report["total_calls"] == 2
    assert report["total_events"] == 3
    assert report["successful_extractions"] == 1
    assert report["retry_cost"] == 1
    assert report["cost_per_successful_extraction"] == 2


def test_fallback_model_is_one_validated_outcome(ledger, monkeypatch):
    from prompture.extraction.core import extract_with_models

    drivers = {"openai/bad": ExtractDriver(['{"value":"x"}']), "openai/good": ExtractDriver(['{"value":"good"}'])}
    monkeypatch.setattr("prompture.extraction.core.get_driver_for_model", lambda model, **kw: drivers[model])
    extract_with_models(
        Output,
        "extract",
        models=list(drivers),
        max_retries=1,
        cache=False,
        strategy="prompted_repair",
        ai_cleanup=False,
    )
    report = ledger.efficiency_report()
    assert report["total_calls"] == 2
    assert report["extraction_outcome_events"] == 1
    assert report["successful_extractions"] == 1
    assert report["fallback_cost"] == 1
    assert report["cost_per_successful_extraction"] == 2


def test_failed_extraction_retains_cost_and_default_is_not_success(ledger, monkeypatch):
    from prompture.extraction.core import extract_with_model

    driver = ExtractDriver(['{"value":"x"}', '{"value":"x"}', '{"value":"good"}'])
    monkeypatch.setattr("prompture.extraction.core.get_driver_for_model", lambda *a, **kw: driver)
    with pytest.raises(ValidationError):
        extract_with_model(
            Output,
            "extract",
            model_name="openai/test",
            max_retries=1,
            cache=False,
            strategy="prompted_repair",
            ai_cleanup=False,
        )
    extract_with_model(
        Output,
        "extract",
        model_name="openai/test",
        max_retries=1,
        cache=False,
        strategy="prompted_repair",
        ai_cleanup=False,
        fallback=Output(value="default"),
    )
    assert ledger.efficiency_report()["successful_extractions"] == 0
    extract_with_model(
        Output,
        "extract",
        model_name="openai/test",
        max_retries=1,
        cache=False,
        strategy="prompted_repair",
        ai_cleanup=False,
    )
    report = ledger.efficiency_report()
    assert report["successful_extractions"] == 1
    assert report["cost_per_successful_extraction"] == 3
    assert report["extraction_outcome_events"] == 3


async def test_async_retry_outcome_and_filtered_report(ledger, monkeypatch):
    from prompture.drivers.async_base import AsyncDriver
    from prompture.extraction.async_core import extract_with_model

    class AsyncExtractDriver(AsyncDriver):
        supports_json_mode = True
        model = "openai/test"
        values = iter(['{"value":"x"}', '{"value":"good"}'])

        async def generate(self, prompt, options):
            return {
                "text": next(self.values),
                "meta": {
                    "cost": 1.0,
                    "cost_status": "estimated",
                    "usage_complete": True,
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            }

    driver = AsyncExtractDriver()
    monkeypatch.setattr("prompture.extraction.async_core.get_async_driver_for_model", lambda *a, **kw: driver)
    await extract_with_model(
        Output,
        "extract",
        model_name="openai/test",
        max_retries=2,
        cache=False,
        strategy="prompted_repair",
        ai_cleanup=False,
    )
    report = ledger.efficiency_report(provider="openai")
    assert report["total_calls"] == 2
    assert report["extraction_outcome_events"] == 1
    assert report["successful_extractions"] == 1
    assert report["retry_cost"] == 1
    assert report["cost_per_successful_extraction"] == 2


@pytest.mark.parametrize("raises", [False, True])
def test_observation_failure_never_masks_result_or_leaks_context(monkeypatch, raises):
    from prompture.extraction._usage import tracked_extraction
    from prompture.infra.tracker import _ctx_extraction

    class BrokenTracker:
        def record(self, event):
            raise OSError("unavailable reporting backend")

    monkeypatch.setattr("prompture.extraction._usage.get_tracker", lambda: BrokenTracker())

    @tracked_extraction
    def extraction():
        if raises:
            raise ValueError("original extraction failure")
        return {"model": Output(value="valid"), "usage": {}}

    if raises:
        with pytest.raises(ValueError, match="original extraction failure"):
            extraction()
    else:
        assert extraction()["model"].value == "valid"
    assert _ctx_extraction.get() is None


@pytest.mark.parametrize(
    "result",
    [
        {"usage": {}, "error": "validation failed"},
        {"usage": {}},
        {
            "model": Output(value="default"),
            "usage": {},
            "field_results": {"value": {"status": "extraction_failed", "used_default": True}},
        },
        {"model": Output(value="valid"), "usage": {"validation_errors": ["failed field"]}},
    ],
)
def test_partial_or_default_stepwise_returns_are_not_success(ledger, result):
    from prompture.extraction._usage import tracked_extraction

    @tracked_extraction
    def stepwise_return():
        return result

    stepwise_return()
    assert ledger.efficiency_report()["successful_extractions"] == 0
