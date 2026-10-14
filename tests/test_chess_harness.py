"""Tests for prompture.games.chess: motif tags, the move harness and puzzle runs."""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("chess")
import chess
import chess.engine

from prompture.drivers.base import Driver
from prompture.games.chess import (
    CONCEPTS,
    ChessGame,
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


def reply(move: str, reason: str = "because", plan: str = "") -> str:
    return json.dumps({"reason": reason, "plan": plan, "move": move})


def tags(fen: str, san: str, concepts=CONCEPTS, **kwargs) -> list[str]:
    board = chess.Board(fen)
    return [m.text for m in describe_move(board, board.parse_san(san), concepts, **kwargs)]


def kinds(fen: str, san: str, concepts=CONCEPTS) -> set[str]:
    board = chess.Board(fen)
    return {m.kind for m in describe_move(board, board.parse_san(san), concepts)}


# -- motifs ------------------------------------------------------------------


class TestMotifs:
    @pytest.mark.parametrize(
        ("fen", "san", "expected"),
        [
            (
                "r1b2rk1/2q1bppp/5n2/p1pPp3/PpN1P3/5N1P/1PB2PP1/R2QR1K1 b - - 1 20",
                "Nd7",
                "allows d6: fork: attacks the queen on c7 and the bishop on e7",
            ),
            (
                "r1b2rk1/2qnbppp/3P4/p1p1p3/PpN1P3/5N1P/1PB2PP1/R2QR1K1 b - - 0 21",
                "Qb6",
                "warning: leaves the queen on b6 to be won by the knight on c4",
            ),
            (
                "7k/6p1/7p/6r1/PrP5/3R1P2/5P1P/5R1K w - - 1 34",
                "c5",
                "warning: leaves the pawn on c5 to be won by the rook on g5",
            ),
        ],
    )
    def test_saved_arena_positions(self, fen, san, expected):
        """The three actual mistakes from Arena, without an engine or model."""
        assert expected in tags(fen, san)

    def test_allows_finds_what_safety_misses(self):
        fen = "r1b2rk1/2q1bppp/5n2/p1pPp3/PpN1P3/5N1P/1PB2PP1/R2QR1K1 b - - 1 20"
        assert tags(fen, "Nd7", concepts={"safety"}) == []
        assert tags(fen, "Nd7", concepts={"allows"}, facts=False) == [
            "allows d6: fork: attacks the queen on c7 and the bishop on e7"
        ]

    def test_allows_winning_capture_and_even_trade(self):
        assert "allows exd5: winning capture of the queen on d5" in tags(
            "4k3/8/4p3/8/8/8/8/3QK3 w - - 0 1", "Qd5", concepts={"allows"}
        )
        assert "allows" not in kinds("4k3/8/4p3/3q4/8/8/8/3QK3 w - - 0 1", "Qxd5", concepts={"allows"})

    def test_allows_en_passant(self):
        assert "allows dxe3: winning capture of the pawn on e4" in tags(
            "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1", "e4", concepts={"allows"}
        )

    def test_allows_ignores_an_illegal_pinned_capture(self):
        assert not any(
            "exd5" in text for text in tags("4k3/8/4p3/8/8/8/8/3QR1K1 w - - 0 1", "Qd5", concepts={"allows"})
        )

    def test_allows_ignores_a_fork_that_can_be_taken(self):
        assert not any(
            text.startswith("allows Nf7:")
            for text in tags("3qk2r/p7/8/6N1/8/8/8/4K3 b - - 0 1", "a6", concepts={"allows"})
        )

    def test_allows_a_mate_in_one(self):
        # Fable vs Astra, 24.h4?? walked into ...Rc1#, a double check the hints did not mention.
        fen = "2r5/5pp1/p5k1/1p1pPbN1/1n6/8/PPr3PP/1K1R2R1 w - - 6 24"
        assert "allows Rc1#: checkmate" in tags(fen, "h4", concepts={"allows"})

    def test_a_fork_answered_by_an_even_trade_is_no_fork(self):
        # 19.Qh4+ "forked" the king on h7 and the queen on e7, which simply took it: ...Qxh4.
        fen = "2r2r2/3bqppk/p7/1p1pP3/1n3Q2/5N2/PPP3PP/1K1R3R w - - 0 19"
        assert "fork" not in kinds(fen, "Qh4+")

    def test_mate_has_no_allows_warnings(self):
        assert "allows" not in kinds("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "Ra8#", concepts={"allows"})

    def test_hints_do_not_mutate_the_position_or_history(self):
        board = chess.Board()
        board.push_san("e4")
        before, history = board.fen(), list(board.move_stack)
        describe_move(board, board.parse_san("d5"))
        assert (board.fen(), board.move_stack) == (before, history)

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


class TestPositionContext:
    """Readable positions and history-aware draw warnings."""

    def test_ascii_board_and_balanced_material(self):
        text = "\n".join(ChessGame().render(chess.Board()))
        assert "8 r n b q k b n r" in text
        assert "1 R N B Q K B N R" in text
        assert "  a b c d e f g h" in text
        assert "White 39, Black 39; equal" in text

    @pytest.mark.parametrize(
        ("fen", "balance"),
        [
            ("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1", "White 9, Black 0; White ahead by 9"),
            ("q3k3/8/8/8/8/8/8/4K3 w - - 0 1", "White 0, Black 9; Black ahead by 9"),
        ],
    )
    def test_material_counts_either_side_without_kings(self, fen, balance):
        assert balance in "\n".join(ChessGame().render(chess.Board(fen)))

    def test_second_and_third_occurrences_in_move_menu(self):
        board = chess.Board()
        harness = ChessHarness(driver=ScriptedDriver([reply("Ng8")]), level=1)
        for san in ("Nf3", "Nf6", "Ng1"):
            board.push_san(san)

        def notes():
            return [m.text for c in harness.candidates(board) if c.name == "Ng8" for m in c.motifs]

        assert "repeats the position: a third time allows a draw claim" in notes()
        board.push_san("Ng8")
        assert "occurred twice" in "\n".join(ChessGame().render(board))
        for san in ("Nf3", "Nf6", "Ng1"):
            board.push_san(san)
        assert "repeats the position a third time: a draw can be claimed" in notes()
        board.push_san("Ng8")
        assert "at least three times" in "\n".join(ChessGame().render(board))

    def test_fen_without_history_does_not_invent_a_repetition(self):
        board = chess.Board()
        for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
            board.push_san(san)
        assert "Repetition:" not in "\n".join(ChessGame().render(chess.Board(board.fen())))


class TestMoveMemory:
    """Only the last accepted choice for the same side and line is carried forward."""

    def test_own_reason_and_plan_survive_an_opponent_reply(self):
        driver = ScriptedDriver([reply("e4", "Claim the center", "Develop Nf3"), reply("Nf3")])
        harness, board = ChessHarness(driver=driver), chess.Board()
        board.push(harness.choose(board).move)
        board.push_san("e5")
        harness.choose(board)
        assert "Your previous move: e4" in driver.prompts[1]
        assert "Your previous reason: Claim the center" in driver.prompts[1]
        assert "Your previous plan: Develop Nf3" in driver.prompts[1]

    def test_sides_do_not_see_each_others_plans(self):
        driver = ScriptedDriver([reply("e4", plan="white plan"), reply("e5", plan="black plan"), reply("Nf3")])
        harness, board = ChessHarness(driver=driver), chess.Board()
        for _ in range(3):
            board.push(harness.choose(board).move)
        assert "white plan" not in driver.prompts[1]
        assert "white plan" in driver.prompts[2]
        assert "black plan" not in driver.prompts[2]

    @pytest.mark.parametrize("reset_kind", ["explicit", "new_game", "different_line", "fen_only"])
    def test_memory_is_not_used_outside_the_next_own_turn(self, reset_kind):
        driver = ScriptedDriver([reply("e4", plan="old plan"), reply("Nf3")])
        harness, board = ChessHarness(driver=driver), chess.Board()
        board.push(harness.choose(board).move)
        board.push_san("e5")
        if reset_kind == "explicit":
            harness.reset()
        elif reset_kind == "new_game":
            board = chess.Board()
        elif reset_kind == "different_line":
            board = chess.Board()
            board.push_san("d4")
            board.push_san("d5")
        else:
            board = chess.Board(board.fen())
        harness.choose(board)
        assert "old plan" not in driver.prompts[-1]

    def test_rejected_plan_is_not_saved(self):
        driver = ScriptedDriver([reply("Qh9", plan="rejected plan"), reply("e4", plan="accepted plan"), reply("Nf3")])
        harness, board = ChessHarness(driver=driver), chess.Board()
        board.push(harness.choose(board).move)
        board.push_san("e5")
        harness.choose(board)
        assert "accepted plan" in driver.prompts[-1]
        assert all("rejected plan" not in prompt for prompt in driver.prompts)

    def test_rescued_move_clears_old_memory(self):
        driver = ScriptedDriver([reply("e4", plan="old plan"), reply("Qh9", plan="bad plan")])
        harness, board = ChessHarness(driver=driver, max_retries=0), chess.Board()
        board.push(harness.choose(board).move)
        board.push_san("e5")
        assert harness.choose(board).rescued
        assert "old plan" not in harness.prompt(board, harness.candidates(board))


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


class TestRecordsAndIdentity:
    """Chess decisions carry records; line identity keeps full history."""

    def test_invalid_then_valid_chess_turn(self):
        board = chess.Board()
        board.push_san("e4")
        decision = ChessHarness(driver=ScriptedDriver([reply("Qxh9"), reply("e5")])).choose(board)
        record = decision.record
        assert record.outcome == "accepted" and record.move == "e5" and record.side == "Black" and record.ply == 1
        assert [a.valid for a in decision.attempts] == [False, True]
        assert record.accounting.calls == 2 and record.accounting.total_tokens == 28

    def test_history_line_and_state_round_trip(self):
        game = ChessGame()
        board = chess.Board()
        for san in ("e4", "e5", "Nf3"):
            board.push_san(san)
        line = game.history(board)
        assert line.root == chess.STARTING_FEN
        assert line.plies == (("White", "e4"), ("Black", "e5"), ("White", "Nf3"))
        restored = game.load_state(game.dump_state(board))
        assert restored.fen() == board.fen() and game.history(restored) == line
        assert game.history(chess.Board(board.fen())).plies == ()  # FEN alone has no history

    def test_illegal_history_fails_explicitly(self):
        with pytest.raises(ValueError, match="illegal chess move"):
            ChessGame().load_state({"initial": None, "moves": ["e2e4", "e2e4"]})

    def test_fingerprint_tracks_concepts_and_engine_but_not_model(self):
        plain = ChessHarness(driver=ScriptedDriver([reply("e4")])).fingerprint()
        assert ChessHarness(driver=ScriptedDriver([reply("e4")]), concepts={"fork"}).fingerprint() != plain
        assert ChessHarness("ollama/other", driver=ScriptedDriver([reply("e4")])).fingerprint() == plain
        assert plain.config["hints_version"] and plain.config["game"] == "chess"
        engine = FakeEngine({"e2e4": 30})
        engine.id = {"name": "FakeFish 1"}
        assisted = ChessHarness(driver=ScriptedDriver([reply("e4")]), level=2, engine=engine).fingerprint()
        assert assisted.config["engine"].startswith("uci:FakeFish 1") and assisted.config["assistance"] == "engine-menu"
