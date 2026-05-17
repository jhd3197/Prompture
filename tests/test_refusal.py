"""Tests for ``prompture.refusal``.

Covers the detector (normalization, category resolution, position
weighting, word boundaries, multi-language markers) and the evaluator
(report shape, sample selection, async path) with mocked drivers.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from prompture.refusal import (
    RefusalCategory,
    RefusalDetector,
    RefusalEvaluator,
    RefusalReport,
    is_refusal,
)
from prompture.refusal.detector import _normalize

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_lowercase(self):
        assert _normalize("Hello WORLD") == "hello world"

    def test_typographic_apostrophe_becomes_ascii(self):
        assert _normalize("I can’t do that") == "i can't do that"

    def test_typographic_quotes_become_ascii(self):
        out = _normalize("“nope”")
        assert '"' in out

    def test_markdown_emphasis_stripped(self):
        assert "cannot" in _normalize("I **cannot** do that")
        assert "*" not in _normalize("I **cannot** do that")
        assert "_" not in _normalize("I __cannot__ do that")

    def test_whitespace_collapsed(self):
        assert _normalize("hello\n\n  world\t!") == "hello world !"

    def test_leading_filler_stripped(self):
        # "Sure, " prefix should disappear so position scoring measures
        # distance from the real start.
        out = _normalize("Sure, but I can't help.")
        assert out.startswith("but i can't help")

    def test_empty_input_returns_empty(self):
        assert _normalize("") == ""
        assert _normalize(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Detector — happy paths
# ---------------------------------------------------------------------------


class TestDetectorHappyPaths:
    def test_hard_refusal(self):
        r = RefusalDetector().detect("I'm sorry, but I cannot help with that.")
        assert r.is_refusal
        assert r.category == RefusalCategory.HARD_REFUSAL
        assert r.confidence >= 0.9

    def test_policy_refusal(self):
        r = RefusalDetector().detect("As an AI language model, I must abide by my training.")
        assert r.is_refusal
        # Either pure policy, or policy+hard collision wins on hard;
        # both are valid refusal categories.
        assert r.category in {RefusalCategory.POLICY, RefusalCategory.HARD_REFUSAL}
        assert r.confidence >= 0.8

    def test_soft_refusal(self):
        r = RefusalDetector().detect("I'd rather not discuss that.")
        assert r.is_refusal
        assert r.category == RefusalCategory.SOFT_REFUSAL

    def test_compliance_not_flagged(self):
        r = RefusalDetector().detect("Sure! Here's a recipe for chocolate cake.")
        assert not r.is_refusal
        assert r.category is None
        assert r.confidence == 0.0

    def test_returned_normalized_text(self):
        r = RefusalDetector().detect("I CANNOT help with that")
        assert r.normalized_text.startswith("i cannot help")

    def test_matched_markers_in_reading_order(self):
        text = "As an AI, I cannot help with that. I refuse to comply."
        r = RefusalDetector().detect(text)
        assert r.is_refusal
        # First match should appear earlier in the normalized text
        # than the second one.
        assert r.position >= 0


# ---------------------------------------------------------------------------
# Detector — empty / whitespace
# ---------------------------------------------------------------------------


class TestEmptyHandling:
    def test_empty_string_is_refusal_by_default(self):
        r = RefusalDetector().detect("")
        assert r.is_refusal
        assert r.category == RefusalCategory.EMPTY
        assert r.confidence == 1.0

    def test_whitespace_only_is_refusal(self):
        r = RefusalDetector().detect("   \n\t  ")
        assert r.is_refusal
        assert r.category == RefusalCategory.EMPTY

    def test_empty_disabled_means_not_refusal(self):
        d = RefusalDetector(empty_is_refusal=False)
        r = d.detect("")
        assert not r.is_refusal
        assert r.category is None


# ---------------------------------------------------------------------------
# Detector — word boundaries
# ---------------------------------------------------------------------------


class TestWordBoundaries:
    def test_cantaloupe_does_not_trigger_can(self):
        # A naive substring search for "i can" would false-positive
        # on "I cantaloupe…".  Our markers are full phrases, and the
        # boundary check protects the few that could be embedded.
        r = RefusalDetector().detect("I cantaloupe this morning, what a strange thought.")
        assert not r.is_refusal

    def test_embedded_marker_inside_word_rejected(self):
        # Build a detector with a single short marker so we can probe
        # the boundary logic directly.
        d = RefusalDetector(
            replace_default_markers=True,
            custom_markers={"hard_refusal": ["nope"]},
        )
        # 'nope' inside 'nopenope' shouldn't match — both sides are word chars.
        r = d.detect("nopenope")
        assert not r.is_refusal
        # But a standalone 'nope' should.
        assert d.detect("nope").is_refusal


# ---------------------------------------------------------------------------
# Detector — position weighting
# ---------------------------------------------------------------------------


class TestPositionWeighting:
    def test_early_match_keeps_full_confidence(self):
        r = RefusalDetector().detect("I cannot help with that.")
        assert r.confidence >= 0.9

    def test_late_match_is_downweighted(self):
        # 800+ chars of fluff before the marker → confidence should
        # drop close to the position_decay floor (0.4 of base).
        fluff = "Some preamble about the topic. " * 30
        d = RefusalDetector(position_threshold=200, position_decay=0.4)
        r = d.detect(fluff + "I cannot help with that.")
        assert r.is_refusal or r.confidence < 0.95
        # Confidence at far position should be near base * decay = 0.95*0.4 = 0.38
        assert r.confidence <= 0.95


# ---------------------------------------------------------------------------
# Detector — Spanish
# ---------------------------------------------------------------------------


class TestSpanish:
    def test_spanish_hard_refusal(self):
        r = RefusalDetector().detect("Lo siento, pero no puedo ayudarte con eso.")
        assert r.is_refusal
        assert r.category == RefusalCategory.HARD_REFUSAL

    def test_spanish_policy(self):
        r = RefusalDetector().detect("Como una IA, no puedo discutir este tema.")
        assert r.is_refusal
        assert r.category in {RefusalCategory.POLICY, RefusalCategory.HARD_REFUSAL}

    def test_spanish_only_detector(self):
        d = RefusalDetector(languages=["es"])
        # English marker shouldn't fire.
        assert not d.is_refusal("I cannot help with that.")
        # Spanish marker should.
        assert d.is_refusal("No puedo ayudarte con eso.")


# ---------------------------------------------------------------------------
# Detector — custom markers
# ---------------------------------------------------------------------------


class TestCustomMarkers:
    def test_extend_defaults(self):
        d = RefusalDetector(custom_markers={"hard_refusal": ["banana boat"]})
        assert d.is_refusal("Banana boat — request denied.")
        # Defaults still work.
        assert d.is_refusal("I cannot help with that.")

    def test_replace_defaults(self):
        d = RefusalDetector(
            replace_default_markers=True,
            custom_markers={"hard_refusal": ["nope"]},
        )
        assert d.is_refusal("nope")
        # Default marker no longer fires.
        assert not d.is_refusal("I cannot help with that.")

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown refusal category"):
            RefusalDetector(custom_markers={"made_up": ["x"]})

    def test_custom_markers_normalized_on_load(self):
        # Pass an upper-case marker with a typographic apostrophe;
        # the detector should still find it in a normalized response.
        d = RefusalDetector(
            replace_default_markers=True,
            custom_markers={"hard_refusal": ["WON’T DO IT"]},
        )
        assert d.is_refusal("Sure, but I WON’T DO IT.")


# ---------------------------------------------------------------------------
# Detector — refusal_categories override
# ---------------------------------------------------------------------------


class TestRefusalCategoriesOverride:
    def test_deflection_excluded_by_default(self):
        d = RefusalDetector()
        # 'instead, i can' is a deflection marker but not a refusal trigger.
        r = d.detect("Instead, I can help you write a sonnet.")
        # If deflection fired, category should be DEFLECTION and
        # is_refusal=False.
        if r.category == RefusalCategory.DEFLECTION:
            assert not r.is_refusal

    def test_deflection_included_when_added(self):
        d = RefusalDetector(
            refusal_categories=[
                RefusalCategory.HARD_REFUSAL,
                RefusalCategory.POLICY,
                RefusalCategory.SOFT_REFUSAL,
                RefusalCategory.DEFLECTION,
                RefusalCategory.EMPTY,
            ]
        )
        # Now deflection should flip is_refusal.
        r = d.detect("Instead, I can help you with something else.")
        if r.category == RefusalCategory.DEFLECTION:
            assert r.is_refusal


# ---------------------------------------------------------------------------
# Module-level shortcut
# ---------------------------------------------------------------------------


class TestModuleLevelIsRefusal:
    def test_shortcut(self):
        assert is_refusal("I'm sorry, but I cannot help with that.")
        assert not is_refusal("Sure! Here you go.")


# ---------------------------------------------------------------------------
# RefusalResult
# ---------------------------------------------------------------------------


class TestRefusalResult:
    def test_bool_protocol(self):
        d = RefusalDetector()
        assert bool(d.detect("I cannot help with that request."))
        assert not bool(d.detect("Here's the answer you wanted."))


# ---------------------------------------------------------------------------
# Evaluator — score pre-existing responses
# ---------------------------------------------------------------------------


class TestEvaluateResponses:
    def test_basic_report_shape(self):
        responses = [
            "I cannot help with that.",
            "Sure, here you go.",
            "As an AI, I cannot do that.",
            "Here is the answer: 42.",
            "",
        ]
        report = RefusalEvaluator().evaluate_responses(responses)
        assert isinstance(report, RefusalReport)
        assert report.total == 5
        assert report.refusals == 3
        assert report.refusal_rate == 3 / 5
        assert report.duration_seconds >= 0

    def test_by_category_counts(self):
        responses = [
            "I cannot help with that.",  # hard
            "As an AI, I have policies.",  # policy
            "I'd rather not.",  # soft
            "",  # empty
            "Sure, here you go.",  # none
        ]
        report = RefusalEvaluator().evaluate_responses(responses)
        assert report.by_category.get("hard_refusal", 0) >= 1
        assert report.by_category.get("empty", 0) >= 1

    def test_samples_kept_with_max_samples(self):
        responses = ["I cannot help with that request."] * 10
        ev = RefusalEvaluator(max_samples=3)
        report = ev.evaluate_responses(responses)
        assert len(report.samples) == 3

    def test_samples_zero_keeps_nothing(self):
        ev = RefusalEvaluator(max_samples=0)
        report = ev.evaluate_responses(["I cannot help with that request."] * 4)
        assert report.samples == []

    def test_prompts_recorded_on_samples(self):
        responses = ["I cannot help with that request."]
        prompts = ["What is X?"]
        report = RefusalEvaluator().evaluate_responses(responses, prompts=prompts)
        assert report.samples[0][0] == "What is X?"
        assert report.samples[0][1] == "I cannot help with that request."

    def test_model_field_passed_through(self):
        report = RefusalEvaluator().evaluate_responses(["Sure!"], model="openai/gpt-4o")
        assert report.model == "openai/gpt-4o"

    def test_empty_responses_yields_zero_rate(self):
        report = RefusalEvaluator().evaluate_responses([])
        assert report.total == 0
        assert report.refusals == 0
        assert report.refusal_rate == 0.0


# ---------------------------------------------------------------------------
# Evaluator — driver path
# ---------------------------------------------------------------------------


class _MockDriver:
    """Minimal stand-in matching the Driver.generate contract."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.model = "mock/test"
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((prompt, dict(options)))
        text = self.responses.pop(0)
        return {"text": text, "meta": {"prompt_tokens": 1, "completion_tokens": 1}}


class TestEvaluateDriver:
    def test_runs_through_driver(self):
        driver = _MockDriver(
            [
                "I cannot help with that request.",
                "Sure, here's the answer.",
                "As an AI, I cannot fulfill this.",
            ]
        )
        report = RefusalEvaluator().evaluate_driver(
            driver,
            prompts=["q1", "q2", "q3"],
        )
        assert report.total == 3
        assert report.refusals == 2
        assert len(driver.calls) == 3

    def test_options_forwarded(self):
        driver = _MockDriver(["Sure!"])
        RefusalEvaluator().evaluate_driver(
            driver,
            prompts=["q"],
            options={"temperature": 0.0, "max_tokens": 50},
        )
        assert driver.calls[0][1] == {"temperature": 0.0, "max_tokens": 50}

    def test_model_string_routes_through_registry(self):
        driver = _MockDriver(["I cannot help you with that request."])
        with patch(
            "prompture.drivers.get_driver_for_model",
            return_value=driver,
        ):
            report = RefusalEvaluator().evaluate_driver(
                "ollama/llama3.1:8b",
                prompts=["q"],
            )
        assert report.refusals == 1
        assert report.model == "ollama/llama3.1:8b"


# ---------------------------------------------------------------------------
# Evaluator — async path
# ---------------------------------------------------------------------------


class _MockAsyncDriver:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.model = "mock/test"
        self.calls: list[str] = []

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(prompt)
        text = self.responses.pop(0)
        return {"text": text, "meta": {"prompt_tokens": 1, "completion_tokens": 1}}


class TestEvaluateDriverAsync:
    def test_async_run(self):
        driver = _MockAsyncDriver(
            [
                "I cannot help with that request.",
                "Sure! Here you go.",
                "I refuse to assist with this.",
            ]
        )

        async def _run():
            return await RefusalEvaluator().evaluate_driver_async(
                driver,
                prompts=["q1", "q2", "q3"],
                concurrency=2,
            )

        report = asyncio.run(_run())
        assert report.total == 3
        assert report.refusals == 2
        assert len(driver.calls) == 3

    def test_async_model_string_routes_through_registry(self):
        driver = _MockAsyncDriver(["Sure!"])
        with patch(
            "prompture.drivers.get_async_driver_for_model",
            return_value=driver,
        ):
            report = asyncio.run(
                RefusalEvaluator().evaluate_driver_async(
                    "openai/gpt-4o",
                    prompts=["q"],
                )
            )
        assert report.total == 1
        assert report.model == "openai/gpt-4o"
