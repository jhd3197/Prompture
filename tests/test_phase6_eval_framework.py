"""Phase 6: LLM-as-judge evaluation primitives.

Covers four orthogonal evaluators:

* :class:`LLMJudge` — single-output rubric scoring
* :class:`PairwiseJudge` — A/B preference with position-bias correction
* :class:`SelfConsistencyEvaluator` — N-sample majority vote
* :class:`FaithfulnessEvaluator` — claim-by-claim RAG grounding

Every test stubs the LLM driver so the suite never touches a provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from prompture.eval import (
    EvalError,
    FaithfulnessEvaluator,
    FaithfulnessResult,
    JudgeResult,
    LLMJudge,
    PairwiseJudge,
    PairwiseResult,
    SelfConsistencyEvaluator,
    SelfConsistencyResult,
)
from prompture.eval.judge import _criteria_block, _criterion_keys
from prompture.eval.pairwise import _parse_choice
from prompture.eval.self_consistency import _default_normalize

# ---------------------------------------------------------------------------
# Shared driver stubs
# ---------------------------------------------------------------------------


class _ScriptedDriver:
    """Returns canned texts in order; records every prompt."""

    def __init__(self, texts: list[str], *, with_meta: bool = True) -> None:
        self._texts = list(texts)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._with_meta = with_meta

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((prompt, options))
        text = self._texts.pop(0) if self._texts else ""
        meta = (
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001}
            if self._with_meta
            else {}
        )
        return {"text": text, "meta": meta}


class _RaisingDriver:
    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("driver-fail")


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class TestLLMJudgeHelpers:
    def test_criteria_block_handles_colon_split(self) -> None:
        block = _criteria_block(["clarity", "factuality: claims must match the reference"])
        assert "clarity" in block
        assert "factuality" in block
        assert "claims must match the reference" in block

    def test_criterion_keys_strip_descriptions(self) -> None:
        assert _criterion_keys(["clarity", "factuality: details"]) == ["clarity", "factuality"]

    def test_criteria_block_skips_blank(self) -> None:
        block = _criteria_block(["", "  ", "factuality"])
        # No empty bullets emitted.
        assert "- \n" not in block
        assert "factuality" in block


class TestLLMJudgeInit:
    def test_rejects_empty_criteria(self) -> None:
        with pytest.raises(EvalError, match="at least one criterion"):
            LLMJudge(driver=_ScriptedDriver([]), criteria=[])

    def test_rejects_small_scale(self) -> None:
        with pytest.raises(EvalError, match="scale"):
            LLMJudge(driver=_ScriptedDriver([]), scale=1)

    def test_rejects_prompt_missing_placeholder(self) -> None:
        with pytest.raises(EvalError, match="placeholder"):
            LLMJudge(driver=_ScriptedDriver([]), prompt="missing all placeholders")


class TestLLMJudgeEvaluate:
    def _judge(self, response_text: str, **kw: Any) -> tuple[LLMJudge, _ScriptedDriver]:
        driver = _ScriptedDriver([response_text])
        judge = LLMJudge(
            driver=driver,
            criteria=["factuality: claims must match the reference", "brevity"],
            scale=5,
            **kw,
        )
        return judge, driver

    def test_clean_json_response(self) -> None:
        judge, driver = self._judge(
            '{"score": 4, "reasoning": "Mostly factual, a bit long.", '
            '"per_criterion": {"factuality": 5, "brevity": 3}}'
        )
        result = judge.evaluate("What is X?", "X is Y", reference="X is Y")
        assert isinstance(result, JudgeResult)
        assert result.score == 4
        assert result.reasoning.startswith("Mostly factual")
        assert result.per_criterion == {"factuality": 5, "brevity": 3}
        # Reference block appears in the rendered prompt.
        assert "REFERENCE ANSWER" in driver.calls[0][0]
        # Meta was forwarded.
        assert result.meta.get("calls") == 1
        assert result.meta.get("total_tokens") == 15

    def test_no_reference_omits_reference_block(self) -> None:
        judge, driver = self._judge(
            '{"score": 3, "per_criterion": {"factuality": 3, "brevity": 3}, "reasoning": "OK"}'
        )
        judge.evaluate("input", "output")
        assert "REFERENCE ANSWER" not in driver.calls[0][0]

    def test_fenced_json_response(self) -> None:
        judge, _ = self._judge(
            'Here is the rubric scoring:\n```json\n'
            '{"score": 2, "per_criterion": {"factuality": 2, "brevity": 2}, "reasoning": "Weak"}\n```'
        )
        result = judge.evaluate("i", "o")
        assert result.score == 2

    def test_embedded_json_in_prose(self) -> None:
        judge, _ = self._judge(
            'Sure. {"score": 5, "per_criterion": {"factuality": 5, "brevity": 5}, "reasoning": "Great"} hope that helps.'
        )
        result = judge.evaluate("i", "o")
        assert result.score == 5

    def test_score_clamped_to_scale(self) -> None:
        judge, _ = self._judge(
            '{"score": 999, "per_criterion": {"factuality": 50, "brevity": -3}, "reasoning": "x"}'
        )
        result = judge.evaluate("i", "o")
        assert result.score == 5  # max(1, min(5, 999))
        assert result.per_criterion["factuality"] == 5
        # -3 clamps up to 1.
        assert result.per_criterion["brevity"] == 1

    def test_score_string_fraction(self) -> None:
        judge, _ = self._judge(
            '{"score": "4/5", "per_criterion": {"factuality": "5", "brevity": "3"}, "reasoning": "x"}'
        )
        result = judge.evaluate("i", "o")
        assert result.score == 4

    def test_case_insensitive_criterion_keys(self) -> None:
        judge, _ = self._judge(
            '{"score": 4, "per_criterion": {"FACTUALITY": 4, "Brevity": 3}, "reasoning": "x"}'
        )
        result = judge.evaluate("i", "o")
        assert result.per_criterion == {"factuality": 4, "brevity": 3}

    def test_pass_threshold(self) -> None:
        judge_passes, _ = self._judge(
            '{"score": 4, "per_criterion": {"factuality": 4, "brevity": 4}, "reasoning": "x"}',
            pass_threshold=3,
        )
        assert judge_passes.evaluate("i", "o").passed is True
        judge_fails, _ = self._judge(
            '{"score": 2, "per_criterion": {"factuality": 2, "brevity": 2}, "reasoning": "x"}',
            pass_threshold=3,
        )
        assert judge_fails.evaluate("i", "o").passed is False

    def test_missing_score_raises(self) -> None:
        judge, _ = self._judge('{"reasoning": "no score field"}')
        with pytest.raises(EvalError, match="score"):
            judge.evaluate("i", "o")

    def test_unparseable_response_raises(self) -> None:
        judge, _ = self._judge("the model said hello and produced no JSON")
        with pytest.raises(EvalError):
            judge.evaluate("i", "o")

    def test_driver_failure_raises_eval_error(self) -> None:
        judge = LLMJudge(driver=_RaisingDriver(), criteria=["clarity"])
        with pytest.raises(EvalError, match="driver call failed"):
            judge.evaluate("i", "o")


# ---------------------------------------------------------------------------
# PairwiseJudge
# ---------------------------------------------------------------------------


class TestParseChoice:
    def test_choice_a(self) -> None:
        assert _parse_choice("CHOICE: A\nREASON: A is more concise.") == ("A", "A is more concise.")

    def test_choice_b(self) -> None:
        assert _parse_choice("Choice = B\nReason: B has more detail.") == ("B", "B has more detail.")

    def test_response_a_fallback(self) -> None:
        choice, _ = _parse_choice("I think Response A is clearly better.")
        assert choice == "A"

    def test_single_letter_first_line(self) -> None:
        assert _parse_choice("A")[0] == "A"

    def test_empty_returns_none(self) -> None:
        assert _parse_choice("") == (None, "")


class TestPairwiseJudge:
    def test_both_passes_agree(self) -> None:
        # First pass picks A, second pass picks B (B presented first → remapped to A).
        driver = _ScriptedDriver(
            [
                "CHOICE: A\nREASON: A is better.",  # A first
                "CHOICE: B\nREASON: A is better.",  # B first, B==original A
            ]
        )
        judge = PairwiseJudge(driver=driver)
        result = judge.compare("input", "candidate A", "candidate B")
        assert isinstance(result, PairwiseResult)
        assert result.winner == "A"
        assert result.confidence == 1.0
        assert result.votes == {"A_first": "A", "B_first": "A"}

    def test_passes_disagree_yields_tie(self) -> None:
        # First pass: A. Second pass: B (B presented first → remapped → A).
        # Wait — for a TIE we need them to disagree at the original-frame level:
        # A_first: A and B_first: B (so the LLM kept picking the FIRST-presented one — pure position bias).
        driver = _ScriptedDriver(
            [
                "CHOICE: A\nREASON: position bias.",  # A first → A
                "CHOICE: A\nREASON: position bias.",  # B first → A→B in original frame
            ]
        )
        judge = PairwiseJudge(driver=driver)
        result = judge.compare("input", "x", "y")
        assert result.winner == "tie"
        assert result.confidence == 0.5
        assert result.votes == {"A_first": "A", "B_first": "B"}

    def test_no_swap_mode(self) -> None:
        driver = _ScriptedDriver(["CHOICE: B\nREASON: better."])
        judge = PairwiseJudge(driver=driver, swap_positions=False)
        result = judge.compare("input", "x", "y")
        assert result.winner == "B"
        assert result.confidence == 1.0
        assert len(driver.calls) == 1  # only one judge call

    def test_invalid_response_no_swap_raises(self) -> None:
        driver = _ScriptedDriver(["garbage output"])
        judge = PairwiseJudge(driver=driver, swap_positions=False)
        with pytest.raises(EvalError, match="no parseable choice"):
            judge.compare("i", "x", "y")

    def test_swap_one_invalid_one_valid_falls_through(self) -> None:
        # First pass invalid, second pass picks A (in B-first frame → B-first's A == original B).
        driver = _ScriptedDriver(
            [
                "no clear choice here",
                "CHOICE: A\nREASON: A is better.",
            ]
        )
        judge = PairwiseJudge(driver=driver)
        result = judge.compare("i", "x", "y")
        # Only the valid pass counts.
        assert result.winner == "B"
        assert result.confidence == 1.0

    def test_driver_failure(self) -> None:
        with pytest.raises(EvalError, match="driver call failed"):
            PairwiseJudge(driver=_RaisingDriver()).compare("i", "x", "y")

    def test_prompt_missing_placeholder(self) -> None:
        with pytest.raises(EvalError, match="placeholder"):
            PairwiseJudge(driver=_ScriptedDriver([]), prompt="no placeholders")


# ---------------------------------------------------------------------------
# SelfConsistencyEvaluator
# ---------------------------------------------------------------------------


class TestDefaultNormalize:
    def test_strips_and_lowercases(self) -> None:
        assert _default_normalize("  Hello World  ") == "hello world"

    def test_strips_trailing_punctuation(self) -> None:
        assert _default_normalize("42.") == "42"
        assert _default_normalize("Yes!") == "yes"

    def test_collapses_whitespace(self) -> None:
        assert _default_normalize("a  b\nc") == "a b c"


class TestSelfConsistency:
    def test_majority_consensus(self) -> None:
        driver = _ScriptedDriver(["42", "42", "42", "43", "42"])
        ev = SelfConsistencyEvaluator(driver=driver, n_samples=5)
        result = ev.evaluate("What is 6*7?")
        assert isinstance(result, SelfConsistencyResult)
        assert result.consensus == "42"
        assert result.agreement == 0.8
        assert result.counts == {"42": 4, "43": 1}
        # Meta accumulated across all 5 calls.
        assert result.meta["calls"] == 5

    def test_tied_majority_picks_one(self) -> None:
        driver = _ScriptedDriver(["A", "B", "A", "B"])
        ev = SelfConsistencyEvaluator(driver=driver, n_samples=4)
        result = ev.evaluate("question")
        # most_common returns insertion-order ties — either is acceptable.
        assert result.consensus in {"a", "b"}
        assert result.agreement == 0.5

    def test_empty_text_skipped(self) -> None:
        driver = _ScriptedDriver(["", "42", "  "])
        ev = SelfConsistencyEvaluator(driver=driver, n_samples=3)
        result = ev.evaluate("q")
        assert result.consensus == "42"
        # Only 1 non-empty sample.
        assert len(result.samples) == 1

    def test_all_empty_raises(self) -> None:
        driver = _ScriptedDriver(["", "", ""])
        ev = SelfConsistencyEvaluator(driver=driver, n_samples=3)
        with pytest.raises(EvalError, match="empty"):
            ev.evaluate("q")

    def test_custom_normalize(self) -> None:
        driver = _ScriptedDriver(["Yes.", "yes", "YES!"])
        ev = SelfConsistencyEvaluator(
            driver=driver,
            n_samples=3,
            normalize=lambda s: s.lower().rstrip(".!?"),
        )
        result = ev.evaluate("?")
        assert result.consensus == "yes"
        assert result.agreement == 1.0

    def test_custom_sampler(self) -> None:
        # No driver needed when sampler= is supplied.
        outputs = iter(["X", "X", "Y"])
        ev = SelfConsistencyEvaluator(sampler=lambda: next(outputs), n_samples=3)
        result = ev.evaluate("ignored")
        assert result.consensus == "x"
        assert result.agreement > 0.6

    def test_temperature_default_routes_into_options(self) -> None:
        driver = _ScriptedDriver(["x"])
        ev = SelfConsistencyEvaluator(driver=driver, n_samples=1, temperature=0.9)
        ev.evaluate("q")
        assert driver.calls[0][1].get("temperature") == 0.9

    def test_invalid_n_samples_raises(self) -> None:
        with pytest.raises(EvalError, match="n_samples"):
            SelfConsistencyEvaluator(driver=_ScriptedDriver([]), n_samples=0)

    def test_missing_driver_and_sampler(self) -> None:
        with pytest.raises(EvalError, match="driver"):
            SelfConsistencyEvaluator()  # no driver, no sampler

    def test_empty_prompt_for_driver_path(self) -> None:
        ev = SelfConsistencyEvaluator(driver=_ScriptedDriver(["x"]), n_samples=1)
        with pytest.raises(EvalError, match="prompt is required"):
            ev.evaluate("")


# ---------------------------------------------------------------------------
# FaithfulnessEvaluator
# ---------------------------------------------------------------------------


class TestFaithfulnessEvaluator:
    def _make(self, responses: list[str]) -> tuple[FaithfulnessEvaluator, _ScriptedDriver]:
        driver = _ScriptedDriver(responses)
        return FaithfulnessEvaluator(driver=driver), driver

    def test_all_supported_score_1(self) -> None:
        ev, _driver = self._make(
            [
                '["Paris is the capital of France.", "The Eiffel Tower is in Paris."]',
                "SUPPORTED",
                "SUPPORTED",
            ]
        )
        result = ev.evaluate(
            answer="Paris is the capital of France. The Eiffel Tower is in Paris.",
            evidence=[
                "Paris is the capital of France.",
                "The Eiffel Tower stands in central Paris.",
            ],
        )
        assert isinstance(result, FaithfulnessResult)
        assert result.score == 1.0
        assert len(result.supported) == 2
        assert result.unsupported == []
        assert result.contradicted == []

    def test_partial_support(self) -> None:
        ev, _ = self._make(
            [
                '["A", "B", "C"]',
                "SUPPORTED",
                "SILENT",
                "CONTRADICTED",
            ]
        )
        result = ev.evaluate(answer="A. B. C.", evidence=["e1", "e2"])
        # default treat_silent_as='unsupported': silent + contradicted both count against.
        assert pytest.approx(result.score) == 1 / 3
        assert result.supported == ["A"]
        assert set(result.unsupported) == {"B", "C"}
        assert result.contradicted == ["C"]

    def test_silent_as_neutral_excludes_silent(self) -> None:
        driver = _ScriptedDriver(
            [
                '["A", "B", "C"]',
                "SUPPORTED",
                "SILENT",
                "SUPPORTED",
            ]
        )
        ev = FaithfulnessEvaluator(driver=driver, treat_silent_as="neutral")
        result = ev.evaluate(answer="A. B. C.", evidence=["e"])
        # B is excluded from denominator entirely → 2/2 = 1.0
        assert result.score == 1.0

    def test_decomposition_returns_no_claims_raises(self) -> None:
        _ev, _ = self._make(["plain prose, no list"])
        # The bullet/numbered fallback will catch the prose as a single line,
        # so we need a truly empty decomposition.
        ev2, _ = self._make([""])
        with pytest.raises(EvalError, match="no claims"):
            ev2.evaluate(answer="x", evidence=["y"])

    def test_unparseable_verdict_treated_as_silent(self) -> None:
        ev, _ = self._make(
            [
                '["claim 1"]',
                "the model rambled and didn't pick a label",  # no verdict word
            ]
        )
        result = ev.evaluate(answer="claim 1.", evidence=["e"])
        # Defaults to SILENT → unsupported (under default treat_silent_as).
        assert result.score == 0.0
        assert result.unsupported == ["claim 1"]
        assert result.contradicted == []

    def test_fenced_json_decomposition(self) -> None:
        ev, _ = self._make(
            [
                '```json\n["claim a", "claim b"]\n```',
                "SUPPORTED",
                "SUPPORTED",
            ]
        )
        result = ev.evaluate("claim a. claim b.", evidence=["e"])
        assert result.score == 1.0
        assert len(result.supported) == 2

    def test_bullet_list_decomposition_fallback(self) -> None:
        ev, _ = self._make(
            [
                "- claim x\n- claim y\n- claim z",
                "SUPPORTED",
                "SUPPORTED",
                "SUPPORTED",
            ]
        )
        result = ev.evaluate("any answer", evidence=["e"])
        assert result.score == 1.0
        assert len(result.supported) == 3

    def test_evidence_text_assembled_with_indices(self) -> None:
        ev, driver = self._make(['["X"]', "SUPPORTED"])
        ev.evaluate("X.", evidence=["chunk one", "chunk two"])
        # The second call is the verification — evidence appears in its prompt.
        verify_prompt = driver.calls[1][0]
        assert "[1] chunk one" in verify_prompt
        assert "[2] chunk two" in verify_prompt

    def test_invalid_treat_silent_as(self) -> None:
        with pytest.raises(EvalError, match="treat_silent_as"):
            FaithfulnessEvaluator(driver=_ScriptedDriver([]), treat_silent_as="bogus")

    def test_prompt_missing_placeholder_raises(self) -> None:
        with pytest.raises(EvalError, match="answer"):
            FaithfulnessEvaluator(driver=_ScriptedDriver([]), decomposition_prompt="no placeholders")
        with pytest.raises(EvalError, match="claim"):
            FaithfulnessEvaluator(
                driver=_ScriptedDriver([]),
                verification_prompt="missing {evidence} but no claim",
            )

    def test_meta_accumulates_across_all_calls(self) -> None:
        ev, _ = self._make(
            [
                '["a", "b"]',
                "SUPPORTED",
                "CONTRADICTED",
            ]
        )
        result = ev.evaluate("a. b.", evidence=["e"])
        # 1 decomposition + 2 verifications = 3 calls.
        assert result.meta["calls"] == 3


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


class TestEvalExports:
    def test_all_evaluators_importable(self) -> None:
        from prompture.eval import (  # noqa: F401
            EvalDriver,
            EvalError,
            FaithfulnessEvaluator,
            FaithfulnessResult,
            JudgeResult,
            LLMJudge,
            PairwiseJudge,
            PairwiseResult,
            SelfConsistencyEvaluator,
            SelfConsistencyResult,
        )

    def test_result_to_dict_round_trip(self) -> None:
        r = JudgeResult(score=4, reasoning="ok", per_criterion={"a": 4}, passed=True, meta={"x": 1})
        d = r.to_dict()
        assert d["score"] == 4
        assert d["per_criterion"] == {"a": 4}
        assert d["passed"] is True
