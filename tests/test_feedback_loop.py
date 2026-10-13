"""Feedback → rules → tested candidate → promote or reject."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prompture.execution.improve import (
    FEEDBACK_HEADING,
    CandidateStatus,
    FeedbackLog,
    FeedbackLoop,
    PromotionLedger,
    apply_feedback_rules,
    feedback_rules,
    propose_from_feedback,
    score_instructions,
    summarize_feedback,
)
from prompture.execution.improve.feedback import feedback_judge

BASE = "You write LinkedIn posts.\n\n## Style\nPlain words."


class _RulesDriver:
    """Summarizes feedback into the canned rules."""

    def __init__(self, rules: list[str]) -> None:
        self.rules = rules
        self.prompts: list[str] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        return {"text": json.dumps(self.rules), "meta": {"cost": 0.01}}


class _Writer:
    """Writes short output when the instructions carry the rule, long output otherwise."""

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        system = messages[0]["content"]
        text = "short post" if "Keep posts under 50 words" in system else "a very long rambling post " * 20
        return {"text": text, "meta": {"cost": 0.001}}


class _LengthJudge:
    """Judge driver: 5 for short outputs, 2 for long ones."""

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        output = prompt.split("OUTPUT TO EVALUATE:", 1)[1].split("Respond with", 1)[0]
        score = 5 if len(output.split()) < 20 else 2
        return {"text": json.dumps({"score": score, "per_criterion": {}, "reasoning": "r"}), "meta": {"cost": 0.002}}


class TestRules:
    def test_append_section(self) -> None:
        revised = apply_feedback_rules(BASE, ["No emojis.", "Keep posts under 50 words"])
        assert revised.endswith(f"{FEEDBACK_HEADING}\n- No emojis.\n- Keep posts under 50 words\n")
        assert feedback_rules(revised) == ["No emojis.", "Keep posts under 50 words"]

    def test_merge_into_existing_section_in_place(self) -> None:
        text = f"Intro.\n\n{FEEDBACK_HEADING}\n- No emojis.\n\n## Style\nPlain words."
        revised = apply_feedback_rules(text, ["no emojis", "End with a question."])
        assert (
            revised == f"Intro.\n\n{FEEDBACK_HEADING}\n- No emojis.\n- End with a question.\n\n## Style\nPlain words."
        )

    def test_nothing_new_returns_unchanged(self) -> None:
        assert apply_feedback_rules(BASE, []) == BASE

    def test_summarize_skips_known_rules(self) -> None:
        driver = _RulesDriver(["No emojis", "Keep posts under 50 words"])
        rules, meta = summarize_feedback(["too long", "stop with the emojis"], driver, existing_rules=["No emojis."])
        assert rules == ["Keep posts under 50 words"]
        assert "- too long" in driver.prompts[0]
        assert "- No emojis." in driver.prompts[0]
        assert meta["cost"] == pytest.approx(0.01)

    def test_summarize_without_corrections_makes_no_call(self) -> None:
        driver = _RulesDriver(["x"])
        assert summarize_feedback([], driver) == ([], {})
        assert driver.prompts == []


class TestFeedbackLog:
    def test_persists_and_filters(self, tmp_path) -> None:
        path = tmp_path / "feedback.jsonl"
        log = FeedbackLog(path)
        log.add("Too long.", target="writer", prompt="p", output="o")
        log.add("Wrong tone.", target="support")
        reloaded = FeedbackLog(path)
        assert len(reloaded) == 2
        assert [e.correction for e in reloaded.entries("writer")] == ["Too long."]
        with pytest.raises(ValueError):
            log.add("   ")


class TestProposeAndScore:
    def test_candidate_is_inert_and_traceable(self) -> None:
        log = FeedbackLog()
        entry = log.add("Too long.", target="writer")
        candidate = propose_from_feedback("writer", BASE, log.entries(), _RulesDriver(["Keep posts under 50 words"]))
        assert candidate is not None
        assert candidate.origin == "feedback"
        assert candidate.status is CandidateStatus.PROPOSED
        assert candidate.motivating_task_ids == [entry.id]
        assert "+- Keep posts under 50 words" in candidate.diff()
        assert candidate.metadata["new_rules"] == ["Keep posts under 50 words"]

    def test_no_new_rules_no_candidate(self) -> None:
        log = FeedbackLog()
        log.add("Too long.")
        assert propose_from_feedback("writer", BASE, log.entries(), _RulesDriver([])) is None

    def test_score_instructions(self) -> None:
        from prompture.execution.improve.feedback import driver_runner

        judge = feedback_judge(_LengthJudge(), ["Keep posts under 50 words"])
        summary = score_instructions(
            BASE + "\nKeep posts under 50 words", ["a", "b"], run=driver_runner(_Writer()), judge=judge
        )
        assert summary.mean_score == 1.0
        assert summary.scored_cases == 2
        assert summary.total_calls == 4
        assert summary.cost_complete
        assert summary.total_cost == pytest.approx(0.006)

    def test_failed_runs_are_counted_not_scored(self) -> None:
        def boom(system: str, prompt: str) -> str:
            raise RuntimeError("provider down")

        summary = score_instructions(BASE, ["a"], run=boom, judge=feedback_judge(_LengthJudge(), []))
        assert summary.scored_cases == 0
        assert summary.termination_counts == {"RuntimeError": 1}
        assert summary.incomplete_measurements


class TestFeedbackLoop:
    def _loop(self, rules: list[str], tmp_path, **kwargs: Any) -> FeedbackLoop:
        from prompture.execution.improve.feedback import driver_runner

        return FeedbackLoop(
            "writer",
            base_instructions=BASE,
            driver=_RulesDriver(rules),
            judge_driver=_LengthJudge(),
            run=driver_runner(_Writer()),
            ledger=PromotionLedger(tmp_path / "promotions.json"),
            **kwargs,
        )

    def test_helpful_rule_is_promoted_and_can_roll_back(self, tmp_path) -> None:
        loop = self._loop(["Keep posts under 50 words"], tmp_path)
        log = FeedbackLog()
        log.add("Way too long.")
        outcome = loop.run(log.entries(), ["dev prompt"], ["held-out 1", "held-out 2"])

        assert outcome.promoted, outcome.reason
        assert outcome.scorecard.heldout.score_delta == pytest.approx(0.75)
        assert "Keep posts under 50 words" in loop.current_instructions()
        assert outcome.candidate.status is CandidateStatus.PROMOTED

        restored = loop.rollback("testing rollback")
        assert restored == BASE
        assert loop.current_instructions() == BASE

    def test_useless_rule_is_rejected(self, tmp_path) -> None:
        loop = self._loop(["End every post with a question."], tmp_path)
        log = FeedbackLog()
        log.add("Ask the reader something.")
        outcome = loop.run(log.entries(), [], ["held-out"])

        assert not outcome.promoted
        assert "below the required" in outcome.reason
        assert outcome.candidate.status is CandidateStatus.REJECTED
        assert loop.current_instructions() == BASE

    def test_nothing_to_propose(self, tmp_path) -> None:
        outcome = self._loop([], tmp_path).run([], ["d"], ["h"])
        assert outcome.candidate is None and not outcome.promoted

    def test_heldout_required(self, tmp_path) -> None:
        loop = self._loop(["Keep posts under 50 words"], tmp_path)
        log = FeedbackLog()
        log.add("Too long.")
        candidate = loop.propose(log.entries())
        with pytest.raises(ValueError, match="held-out"):
            loop.evaluate(candidate, ["d"], [])

    def test_assistant_applies_promoted_version(self, tmp_path) -> None:
        from prompture import Assistant, Persona

        assistant = Assistant(name="writer", persona=Persona(name="writer", system_prompt=BASE), model="openai/gpt-4o")
        loop = self._loop(["Keep posts under 50 words"], tmp_path)
        log = FeedbackLog()
        log.add("Too long.")
        loop.run(log.entries(), [], ["h"])
        updated = assistant.with_instructions(loop.ledger.active_content(assistant.name))
        assert "Keep posts under 50 words" in updated.instructions
