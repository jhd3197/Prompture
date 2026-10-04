"""Tests for prompture.games.chess: motif tags, the move harness and puzzle runs."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("chess")
import chess
import chess.engine

from prompture.drivers.base import Driver
from prompture.games.chess import (
    CONCEPTS,
    ChessHarness,
    describe_move,
    play_puzzle,
    run_puzzles,
)


class ScriptedDriver(Driver):
    """Replies with canned texts, one per call, and records the prompts."""

    supports_messages = True

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.model = "scripted"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {
            "text": text,
            "meta": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "cost": 0.001,
                "raw_response": {},
            },
        }


class FakeEngine:
    """Scores moves from a {uci: centipawns} table; unknown moves score -900."""

    def __init__(self, scores: dict[str, int]):
        self.scores = scores

    def analyse(self, board, limit, multipv=1):
        moves = sorted(board.legal_moves, key=lambda m: -self.scores.get(m.uci(), -900))
        return [
            {"pv": [m], "score": chess.engine.PovScore(chess.engine.Cp(self.scores.get(m.uci(), -900)), board.turn)}
            for m in moves[:multipv]
        ]


def reply(move: str, reason: str = "because") -> str:
    return f'{{"reason": "{reason}", "move": "{move}"}}'


def tags(fen: str, san: str, concepts=CONCEPTS, **kwargs) -> list[str]:
    board = chess.Board(fen)
    return [m.text for m in describe_move(board, board.parse_san(san), concepts, **kwargs)]


def kinds(fen: str, san: str, concepts=CONCEPTS) -> set[str]:
    board = chess.Board(fen)
    return {m.kind for m in describe_move(board, board.parse_san(san), concepts)}


# -- motifs ------------------------------------------------------------------


class TestMotifs:
    def test_knight_fork_of_king_and_queen(self):
        assert "fork: attacks the king on g8 and the queen on c8" in tags("2q3k1/8/8/3N4/8/8/8/6K1 w - - 0 1", "Ne7+")

    def test_fork_by_a_piece_that_is_simply_lost_is_not_a_fork(self):
        # The king takes the undefended knight on f7.
        board_tags = kinds("3qk2r/8/8/6N1/8/8/8/4K3 w - - 0 1", "Nf7")
        assert "fork" not in board_tags
        assert "safety" in board_tags

    def test_absolute_pin(self):
        assert "pin: pins the knight on c6 to the king" in tags("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1", "Bb5")

    def test_pawn_pins_are_not_reported(self):
        assert "pin" not in kinds("4k3/4p3/8/8/8/8/8/R5K1 w - - 0 1", "Re1")

    def test_skewer_through_the_king(self):
        assert "skewer: attacks the king on d5, with the rook on h5 behind it" in tags(
            "8/8/8/3k3r/8/8/8/R5K1 w - - 0 1", "Ra5+"
        )

    def test_back_rank_mate(self):
        assert {"check", "mate", "back-rank-mate"} <= kinds("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8#")

    def test_wins_hanging_piece(self):
        assert "wins the knight on d5, which was undefended" in tags("4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1", "Rxd5")

    def test_wins_material_against_a_defended_piece(self):
        assert "wins material: the rook on e5 for a pawn" in tags("4r1k1/8/8/4r3/3P4/8/8/6K1 w - - 0 1", "dxe5")

    def test_safety_warning_for_a_queen_walking_into_a_pawn(self):
        assert "warning: leaves the queen on d5 to be won by the pawn on e6" in tags(
            "4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1", "Qd5"
        )

    def test_even_trade_is_not_a_safety_warning(self):
        assert "safety" not in kinds("4k3/8/4p3/3q4/8/8/8/3QK3 w - - 0 1", "Qxd5")

    def test_discovered_attack_and_check(self):
        assert "discovered attack: the rook on e1 now attacks the queen on e8" in tags(
            "4q1k1/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "Nf6+"
        )
        assert "discovered check from the rook on e1" in tags("4k3/8/8/8/8/8/4N3/4R1K1 w - - 0 1", "Nc3")

    def test_discovered_target_that_can_take_back_is_not_reported(self):
        assert "discovered-attack" not in kinds("6k1/8/8/8/3q4/8/1N6/B6K w - - 0 1", "Nc4")

    def test_capturing_a_pinned_piece(self):
        assert "pin: captures the pinned queen on f1" in tags("7k/8/8/8/8/8/6PP/4rQ1K b - - 0 1", "Rxf1#")

    def test_facts(self):
        assert tags("4k3/8/8/8/8/8/8/4K2R w K - 0 1", "O-O") == ["castles kingside"]
        assert tags("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b8=Q+") == ["promotes to a queen", "gives check"]

    def test_concepts_are_switches(self):
        fen, san = "2q3k1/8/8/3N4/8/8/8/6K1 w - - 0 1", "Ne7+"
        assert tags(fen, san, concepts=()) == ["gives check"]
        assert tags(fen, san, concepts=(), facts=False) == []

    def test_unknown_concept_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown chess concepts"):
            tags("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "Kd1", concepts={"windmill"})


# -- harness -----------------------------------------------------------------


class TestHarness:
    def test_level_0_offers_every_legal_move_without_tags(self):
        harness = ChessHarness(driver=ScriptedDriver([reply("e4")]), concepts=())
        candidates = harness.candidates(chess.Board())
        assert len(candidates) == 20
        assert all(not c.motifs for c in candidates)
        assert harness.schema(candidates)["properties"]["move"]["enum"] == [c.san for c in candidates]

    def test_choose_returns_the_models_move(self):
        driver = ScriptedDriver([reply("e4", "centre")])
        decision = ChessHarness(driver=driver).choose(chess.Board())
        assert decision.san == "e4"
        assert decision.reason == "centre"
        assert not decision.rescued
        assert decision.usage["total_tokens"] == 14

    def test_illegal_move_is_retried_with_feedback(self):
        driver = ScriptedDriver([reply("Qxh9"), reply("Nf3")])
        decision = ChessHarness(driver=driver).choose(chess.Board())
        assert decision.san == "Nf3"
        assert [a.error for a in decision.attempts] == ["Qxh9 is not a legal move in this position", None]
        assert "rejected: Qxh9 is not a legal move" in driver.prompts[1]
        assert decision.usage["total_tokens"] == 28

    def test_uci_and_decorated_san_are_accepted(self):
        assert ChessHarness(driver=ScriptedDriver([reply("g1f3")])).choose(chess.Board()).san == "Nf3"
        assert ChessHarness(driver=ScriptedDriver([reply("e4!?")])).choose(chess.Board()).san == "e4"

    def test_invalid_json_counts_as_a_failed_attempt(self):
        driver = ScriptedDriver(["I like the king's pawn", reply("e4")])
        decision = ChessHarness(driver=driver).choose(chess.Board())
        assert decision.san == "e4"
        assert "not valid JSON" in decision.attempts[0].error

    def test_rescue_after_retries_still_plays_a_legal_move(self):
        board = chess.Board()
        decision = ChessHarness(driver=ScriptedDriver([reply("Ke5")]), max_retries=2, seed=1).choose(board)
        assert decision.rescued
        assert len(decision.attempts) == 3
        assert decision.move in board.legal_moves

    def test_level_1_tags_facts_and_concepts_in_the_prompt(self):
        board = chess.Board("2q3k1/8/8/3N4/8/8/8/6K1 w - - 0 1")
        driver = ScriptedDriver([reply("Ne7+")])
        ChessHarness(driver=driver, level=1, concepts={"fork"}).choose(board)
        assert "- Ne7+: gives check; fork: attacks the king on g8 and the queen on c8" in driver.prompts[0]
        assert "knight d5" in driver.prompts[0]

    def test_engine_levels_need_an_engine(self):
        with pytest.raises(ValueError, match="needs an engine"):
            ChessHarness(driver=ScriptedDriver([reply("e4")]), level=2)

    def test_level_2_removes_blunders_and_rejects_hidden_moves(self):
        board = chess.Board("4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1")
        engine = FakeEngine({"d1d5": 500, "d1d2": 400})
        driver = ScriptedDriver([reply("Kf1"), reply("Rxd5")])
        harness = ChessHarness(driver=driver, level=2, engine=engine)
        assert [c.san for c in harness.candidates(board)] == ["Rd2", "Rxd5"]
        decision = harness.choose(board)
        assert decision.attempts[0].error == "Kf1 is legal but not one of your listed moves"
        assert decision.san == "Rxd5"

    def test_level_4_ranks_and_shows_evaluations(self):
        board = chess.Board("4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1")
        engine = FakeEngine({"d1d5": 500, "d1d2": 40, "e1f2": 20, "e1e2": 10})
        driver = ScriptedDriver([reply("Rxd5")])
        decision = ChessHarness(driver=driver, level=4, engine=engine).choose(board)
        assert [c.san for c in decision.candidates] == ["Rxd5", "Rd2", "Kf2"]
        assert "- Rxd5: engine: +5.00" in driver.prompts[0]

    def test_engine_fallback_plays_the_best_offered_move(self):
        board = chess.Board("4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1")
        engine = FakeEngine({"d1d5": 500, "d1d2": 40, "e1f2": 20})
        decision = ChessHarness(driver=ScriptedDriver([reply("??")]), level=3, engine=engine, max_retries=0).choose(
            board
        )
        assert decision.rescued
        assert decision.san == "Rxd5"

    def test_game_over_is_rejected(self):
        board = chess.Board("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1")
        with pytest.raises(ValueError, match="game is over"):
            ChessHarness(driver=ScriptedDriver([reply("e4")])).choose(board)


# -- puzzles -----------------------------------------------------------------

FORK_PUZZLE = {"id": "fork", "fen": "2q2k2/8/8/3N4/8/8/8/6K1 b - - 0 1", "moves": ["f8g8", "d5e7", "g8f7", "e7c8"]}
MATE_PUZZLE = {"id": "mate", "fen": "7k/5ppp/8/8/8/8/8/RR4K1 b - - 0 1", "moves": ["h8g8", "a1a8"]}


class TestPuzzles:
    def test_solved_line(self):
        result = play_puzzle(ChessHarness(driver=ScriptedDriver([reply("Ne7+"), reply("Nxc8")])), FORK_PUZZLE)
        assert result.solved
        assert result.played == result.expected == ["Ne7+", "Nxc8"]

    def test_wrong_move_stops_the_puzzle(self):
        result = play_puzzle(ChessHarness(driver=ScriptedDriver([reply("Nb6")])), FORK_PUZZLE)
        assert not result.solved
        assert result.played == ["Nb6"]

    def test_any_mate_counts_on_the_last_move(self):
        report = run_puzzles(ChessHarness(driver=ScriptedDriver([reply("Rb8#")])), [MATE_PUZZLE])
        assert report.solved == 1
        assert report.solve_rate == 1.0
        assert report.rescues == 0
