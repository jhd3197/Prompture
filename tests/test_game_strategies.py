"""Experimental strategies and image representation: bounds, failures, accounting and fallbacks."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from prompture.drivers.base import Driver
from prompture.games import (
    CancelToken,
    Connect4,
    Connect4Board,
    GameBudget,
    GameHarness,
    MemoryTraceSink,
    TurnCancelled,
    TurnFailed,
)
from prompture.games.evaluation import Fixture, evaluate, scorecard
from prompture.games.strategies import EXPERIMENTS, Proposals, Reconsider, experiment_status

WARNED = Connect4Board("4643663")  # Yellow: column 5 lets Red win next move (urgent safety warning)


def say(move: str, reason: str = "r", plan: str = "p") -> str:
    return json.dumps({"reason": reason, "plan": plan, "move": move})


class Script(Driver):
    """Replies in order; records prompts, options and any images it was sent."""

    supports_messages = True
    supports_vision = False
    model = "scripted/strategy"

    def __init__(self, replies: list[str], pause: float = 0.0):
        self.replies = list(replies)
        self.pause = pause
        self.prompts: list[str] = []
        self.options: list[dict[str, Any]] = []
        self.images: list[list[Any]] = []

    def _reply(self, prompt: str, options: dict[str, Any], images: list[Any]) -> dict[str, Any]:
        time.sleep(self.pause)
        self.prompts.append(prompt)
        self.options.append(dict(options))
        self.images.append(images)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {"text": text, "meta": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}}

    def generate(self, prompt, options):
        return self._reply(prompt, options, [])

    def generate_messages(self, messages, options):
        content = messages[-1]["content"]
        if isinstance(content, list):
            text = next(b["text"] for b in content if b["type"] == "text")
            return self._reply(text, options, [b["source"] for b in content if b["type"] == "image"])
        return self._reply(content, options, [])


def harness(driver: Script, **kwargs: Any) -> GameHarness:
    return GameHarness(Connect4(), driver=driver, level=1, concepts="all", **kwargs)


class TestReconsider:
    def test_no_warning_no_review(self):
        driver = Script([say("4")])
        decision = harness(driver, strategy=Reconsider()).choose(Connect4Board())
        assert len(driver.prompts) == 1 and decision.record.strategy["reviewed"] is False

    def test_one_review_with_the_witness_and_both_choices_kept(self):
        driver = Script([say("5"), say("1", reason="avoid the gift")])
        decision = harness(driver, strategy=Reconsider()).choose(WARNED)
        info = decision.record.strategy
        assert decision.name == "1" and decision.reason == "avoid the gift" and decision.model_selected
        assert (info["initial"], info["final"], info["revised"]) == ("5", "1", True)
        assert [a.purpose for a in decision.attempts] == ["choose", "review"]
        assert "You chose 5" in driver.prompts[1] and "warning: lets Red win next move in column 5" in driver.prompts[1]

    def test_review_is_never_reviewed_again(self):
        driver = Script([say("5"), say("6"), say("1")])
        decision = harness(driver, strategy=Reconsider()).choose(WARNED)
        assert len(driver.prompts) == 2 and decision.name == "6"

    def test_failed_review_keeps_or_stops_by_version(self):
        kept = harness(Script([say("5"), say("9")]), strategy=Reconsider()).choose(WARNED)
        assert kept.name == "5" and kept.record.strategy["review_failure"] == "invalid_output"
        assert kept.record.outcome == "accepted" and kept.record.failure is None
        with pytest.raises(TurnFailed, match="review failed"):
            harness(Script([say("5"), say("9")]), strategy=Reconsider(on_review_failure="stop")).choose(WARNED)

    def test_budget_refusing_the_review_keeps_the_original(self):
        driver = Script([say("5"), say("1")])
        decision = harness(driver, strategy=Reconsider(), budget=GameBudget(max_calls=1)).choose(WARNED)
        assert decision.name == "5" and decision.record.strategy["review_failure"] == "budget"
        assert len(driver.prompts) == 1

    def test_cancellation_during_review_plays_nothing(self):
        token = CancelToken()

        def cancel_on_review(event):
            if event.kind == "attempt_started" and event.data["purpose"] == "review":
                token.cancel("stop pressed")

        driver = Script([say("5"), say("1")], pause=0.3)
        h = harness(driver, strategy=Reconsider(), cancel=token, on_event=cancel_on_review, cleanup_timeout=0.0)
        with pytest.raises(TurnCancelled):
            h.choose(WARNED)


class TestProposals:
    def test_agreement_skips_the_selector(self):
        driver = Script([say("4"), say("4")])
        decision = harness(driver, strategy=Proposals()).choose(Connect4Board())
        info = decision.record.strategy
        assert len(driver.prompts) == 2 and info["agreement"] and info["selector"] is None
        assert [a.purpose for a in decision.attempts] == ["proposal", "proposal"]

    def test_disagreement_goes_to_a_constrained_selector(self):
        driver = Script([say("3", "left"), say("5", "right"), say("5", "picked")])
        decision = harness(driver, strategy=Proposals()).choose(Connect4Board())
        selector = decision.attempts[-1]
        assert decision.name == "5" and decision.reason == "picked" and decision.record.strategy["selector"] == "5"
        assert selector.purpose == "selector" and selector.schema["properties"]["move"]["enum"] == ["3", "5"]
        assert "- 4" not in driver.prompts[2].split("Your moves")[1] and "left" in driver.prompts[2]
        assert "agreement does not make a move correct" in driver.prompts[2]

    def test_selector_cannot_escape_the_proposals(self):
        decision = harness(Script([say("3"), say("5"), say("4")]), strategy=Proposals()).choose(Connect4Board())
        assert decision.name == "3" and decision.record.strategy["fallback"] == "first_valid_proposal"
        with pytest.raises(TurnFailed, match="selector failed"):
            harness(Script([say("3"), say("5"), say("4")]), strategy=Proposals(on_selector_failure="stop")).choose(
                Connect4Board()
            )

    def test_partial_and_total_worker_failure(self):
        partial = harness(Script([say("9"), say("4")]), strategy=Proposals()).choose(Connect4Board())
        assert partial.name == "4" and partial.record.strategy["partial"] and len(partial.attempts) == 2
        with pytest.raises(TurnFailed, match="only 1 of 2"):
            harness(Script([say("9"), say("4")]), strategy=Proposals(on_partial="stop")).choose(Connect4Board())
        rescued = harness(Script([say("9")]), strategy=Proposals(), seed=1).choose(Connect4Board())
        assert rescued.rescued and len(rescued.attempts) == 2

    def test_call_bound_options_accounting_and_trace_links(self):
        sink = MemoryTraceSink()
        driver = Script([say("3"), say("5"), say("6"), say("7")])
        strategy = Proposals(workers=3, worker_options=({"temperature": 0.2}, {"temperature": 0.8}))
        decision = harness(driver, strategy=strategy, trace=sink, options={"max_tokens": 50}).choose(Connect4Board())
        assert len(driver.prompts) == 4  # workers + one selector, never more
        assert [o.get("temperature") for o in driver.options[:3]] == [0.2, 0.8, None]
        assert all(o["max_tokens"] == 50 for o in driver.options)
        record = decision.record
        assert record.accounting.calls == 4 and record.accounting.total_tokens == 60
        traced = {r["attempt"]["attempt_id"] for r in sink.of_type("attempt")}
        assert {p["attempt_id"] for p in record.strategy["proposals"]} <= traced
        assert {a.attempt_id for a in record.attempts} == traced

    def test_budget_bounds_the_workers(self):
        driver = Script([say("3"), say("5"), say("5")])
        decision = harness(driver, strategy=Proposals(workers=3), budget=GameBudget(max_calls=1)).choose(
            Connect4Board()
        )
        assert len(driver.prompts) == 1 and decision.name == "3" and decision.record.strategy["budget_stopped"]

    def test_strategies_are_opt_in_and_fingerprinted(self):
        plain = harness(Script([say("4")]))
        assert plain.strategy is None and plain.representation == "text"
        digests = {
            plain.fingerprint().digest,
            harness(Script([say("4")]), strategy=Reconsider()).fingerprint().digest,
            harness(Script([say("4")]), strategy=Proposals()).fingerprint().digest,
            harness(Script([say("4")]), strategy=Proposals(workers=3)).fingerprint().digest,
        }
        assert len(digests) == 4
        with pytest.raises(ValueError):
            Proposals(workers=1)


class TestEvaluationOfExperiments:
    def test_fixes_and_harms_are_both_counted(self):
        fix = Fixture.from_dict(
            {
                "id": "fix",
                "game": "Connect Four",
                "split": "dev",
                "provenance": "t",
                "moves": list("4643663"),
                "acceptable": ["1"],
            }
        )
        harm = Fixture.from_dict(
            {
                "id": "harm",
                "game": "Connect Four",
                "split": "dev",
                "provenance": "t",
                "moves": list("4643663"),
                "acceptable": ["5"],
            }
        )
        make = lambda: harness(Script([say("5"), say("1")]), strategy=Reconsider())  # noqa: E731
        card = scorecard(evaluate(make, [fix, harm], game=Connect4()))
        stats = card["experiments"]["reconsider"]
        assert (stats["reviews"], stats["fixes"], stats["harms"], stats["revisions_judged"]) == (2, 1, 1, 2)

    def test_experiment_status_is_live_pending_without_live_evidence(self):
        assert experiment_status(None) == "live_pending"
        assert set(EXPERIMENTS) == {"reconsider", "proposals", "image"}
        assert all(e["baseline"] == e["rollback"] for e in EXPERIMENTS.values())


class TestImageRepresentation:
    def test_games_without_a_renderer_fall_back_to_text_and_say_so(self):
        driver = Script([say("4")])
        driver.supports_vision = True
        decision = harness(driver, representation="text+image").choose(Connect4Board())
        info = decision.record.context["representation"]
        assert info["effective"] == "text" and "no image renderer" in info["reason"] and driver.images == [[]]

    def test_unknown_representation_is_rejected(self):
        with pytest.raises(ValueError, match="representation"):
            harness(Script([say("4")]), representation="image-only")


chess = pytest.importorskip("chess")


class TestChessImages:
    def make(self, driver: Script, image_format: str = "svg"):
        from prompture.games.chess import ChessHarness

        h = ChessHarness(driver=driver, representation="text+image")
        h.game.image_format = image_format
        return h

    def test_svg_image_is_sent_with_the_full_text_and_a_stable_digest(self):
        board = chess.Board()
        board.push_san("e4")
        driver = Script([say("e5")])
        driver.supports_vision = True
        decision = self.make(driver).choose(board)
        info = decision.record.context["representation"]
        assert info["effective"] == "text+image" and info["media_type"] == "image/svg+xml"
        assert info["renderer"].startswith("python-chess-svg/1") and len(driver.images[0]) == 1
        assert "Position (FEN):" in driver.prompts[0] and "- e5" in driver.prompts[0]
        assert "picture of the board is attached" in driver.prompts[0]
        again = Script([say("e5")])
        again.supports_vision = True
        assert self.make(again).choose(board).record.context["representation"]["digest"] == info["digest"]
        other = chess.Board()
        other.push_san("d4")
        third = Script([say("d5")])
        third.supports_vision = True
        assert self.make(third).choose(other).record.context["representation"]["digest"] != info["digest"]

    def test_png_without_a_rasterizer_is_a_recorded_text_fallback(self):
        try:
            import cairosvg  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("cairosvg is installed; the fallback path is not reachable")
        driver = Script([say("e4")])
        driver.supports_vision = True
        info = self.make(driver, "png").choose(chess.Board()).record.context["representation"]
        assert info["effective"] == "text" and "cairosvg" in info["reason"] and driver.images == [[]]

    def test_driver_without_vision_gets_text(self):
        info = self.make(Script([say("e4")])).choose(chess.Board()).record.context["representation"]
        assert info == {"requested": "text+image", "effective": "text", "reason": "the driver does not accept images"}

    def test_renderer_marks_last_move_and_check(self):
        from prompture.games.chess.render import board_svg

        board = chess.Board()
        for san in ("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7#"):
            board.push_san(san)
        svg = board_svg(board)
        assert "lastmove" in svg and "check" in svg and board_svg(board) == svg
