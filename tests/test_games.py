"""Tests for prompture.games: the generic harness, Connect Four, Othello and matches."""

from __future__ import annotations

from typing import Any

import pytest

from prompture.drivers.base import Driver
from prompture.games import (
    Connect4,
    Connect4Board,
    Connect4Engine,
    EnginePlayer,
    GameHarness,
    Othello,
    OthelloBoard,
    OthelloEngine,
    RandomPlayer,
    play_match,
)
from prompture.games.connect4 import describe_move as connect4_tags
from prompture.games.othello import PASS, parse_square
from prompture.games.othello import describe_move as othello_tags


class ScriptedDriver(Driver):
    """Replies with canned texts, one per call (the last one repeats), and records the prompts."""

    supports_messages = True

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.model = "scripted"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        text = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return {"text": text, "meta": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0}}


def reply(move: str) -> str:
    return f'{{"reason": "because", "move": "{move}"}}'


def c4_texts(moves: str, col: int, concepts=Connect4.concepts, **kwargs) -> list[str]:
    return [m.text for m in connect4_tags(Connect4Board(moves), col - 1, concepts, **kwargs)]


def othello_texts(board: OthelloBoard, square: str, concepts=Othello.concepts) -> list[str]:
    return [m.text for m in othello_tags(board, parse_square(square), concepts)]


def othello_from_rows(rows: list[str], turn: int = 1) -> OthelloBoard:
    """A board from 8 strings of '.', 'B' and 'W', top row first."""
    board = OthelloBoard()
    board.cells = [{".": 0, "B": 1, "W": 2}[ch] for row in rows for ch in row]
    board.turn = turn
    return board


# -- Connect Four ------------------------------------------------------------


class TestConnect4Rules:
    def test_discs_fall_and_four_wins(self):
        board = Connect4Board("4455667")
        assert board.won == 1
        assert board.legal_moves() == []
        assert Connect4().winner(board) == "Red"

    def test_full_column_is_illegal(self):
        board = Connect4Board("111111")
        assert 0 not in board.legal_moves()
        with pytest.raises(ValueError):
            board.play(0)

    def test_parse_move(self):
        game, board = Connect4(), Connect4Board("111111")
        assert game.parse_move(board, "column 4") == 3
        assert game.parse_move(board, "1") is None
        assert game.parse_move(board, "8") is None


class TestConnect4Tags:
    def test_win(self):
        assert "wins: connects four" in c4_texts("445566", 7)

    def test_block(self):
        assert "blocks Red's four in column 7" in c4_texts("4455661", 7)

    def test_double_threat(self):
        texts = c4_texts("4455", 3)
        assert "double threat: wins next move in columns 2 and 6; only one can be blocked" in texts
        assert not any(t.startswith("threat:") for t in texts)

    def test_threat_for_later(self):
        assert c4_texts("354551", 4, {"threat"}, facts=False) == ["threat: four would connect at column 6, row 4"]

    def test_safety_warns_about_filling_under_a_winning_cell(self):
        # Red's winning cell is column 5, row 3; Yellow filling row 2 under it hands it over.
        assert c4_texts("4643663", 5, {"safety"}, facts=False) == ["warning: lets Red win next move in column 5"]

    def test_facts_and_center(self):
        assert c4_texts("", 4) == ["lands on row 1", "takes the center column, part of the most lines"]
        assert c4_texts("44444", 4, ())[-1] == "fills column 4"
        assert c4_texts("", 4, (), facts=False) == []

    def test_unknown_concept(self):
        with pytest.raises(ValueError, match="Unknown Connect Four concepts"):
            c4_texts("", 4, {"fork"})


class TestConnect4Engine:
    def test_finds_the_immediate_win_and_the_double_threat(self):
        scores = Connect4Engine(4).analyse(Connect4Board("445566"), 7)
        assert scores[6].label == "wins in 1"
        scores = Connect4Engine(4).analyse(Connect4Board("4455"), 7)
        assert scores[2].label == scores[5].label == "wins in 2"

    def test_sees_a_forced_loss(self):
        # Yellow threatens column 3; every other move loses at once.
        scores = Connect4Engine(4).analyse(Connect4Board("14624615"), 7)
        assert scores[0].label == "loses in 1"
        assert max(scores, key=lambda c: scores[c].value) == 2


# -- Othello -----------------------------------------------------------------


class TestOthelloRules:
    def test_opening_moves_and_flips(self):
        board = OthelloBoard()
        assert sorted(Othello().move_name(board, m) for m in board.legal_moves()) == ["c4", "d3", "e6", "f5"]
        board.play(parse_square("d3"))
        assert board.count(1) == 4 and board.count(2) == 1

    def test_pass_and_game_end(self):
        board = othello_from_rows(["BW......"] + ["........"] * 7, turn=2)
        assert board.legal_moves() == [PASS]  # White cannot bracket; Black (c1) can
        board.play(PASS)
        board.play(parse_square("c1"))
        assert board.is_over()
        assert Othello().winner(board) == "Black"

    def test_illegal_moves_are_rejected(self):
        game, board = Othello(), OthelloBoard()
        assert game.parse_move(board, "a1") is None
        assert game.parse_move(board, "pass") is None
        assert game.parse_move(board, " D3 ") == parse_square("d3")


class TestOthelloTags:
    def test_corner(self):
        board = othello_from_rows([".WB....."] + ["........"] * 7)
        assert "takes the corner a1; it can never be flipped" in othello_texts(board, "a1")

    def test_x_square_warning(self):
        board = othello_from_rows(["........", "........", "..W.....", "...B...."] + ["........"] * 4)
        assert any(t.startswith("warning: X-square next to the empty corner a1") for t in othello_texts(board, "b2"))

    def test_safety_gives_away_a_corner(self):
        board = OthelloBoard(["d3", "e3", "f4", "g5", "e2", "e1", "f3", "c5", "d1", "f2"])
        assert othello_texts(board, "g2", {"safety"}) == ["flips 2 discs", "warning: gives White the corner h1"]

    def test_mobility_and_flips(self):
        texts = othello_texts(OthelloBoard(), "d3")
        assert texts == ["flips 1 disc", "leaves White 3 moves"]


class TestOthelloEngine:
    def test_prefers_a_corner(self):
        board = othello_from_rows([".WB....."] + ["........"] * 6 + ["......WB"])
        scores = OthelloEngine(2).analyse(board, 3)
        assert max(scores, key=lambda m: scores[m].value) == parse_square("a1")


# -- harness and matches -----------------------------------------------------


class TestGenericHarness:
    def test_connect4_menu_retry_and_prompt(self):
        driver = ScriptedDriver([reply("8"), reply("4")])
        decision = GameHarness(Connect4(), driver=driver, level=1, concepts={"center"}).choose(Connect4Board())
        assert decision.name == "4" and decision.move == 3
        assert decision.attempts[0].error == "8 is not a legal move in this position"
        prompt = driver.prompts[0]
        assert "You are playing Connect Four as Red." in prompt
        assert "- 4: lands on row 1; takes the center column" in prompt
        assert "Column 1 (bottom up): empty" in prompt

    def test_builtin_engine_is_used_for_engine_levels(self):
        board = Connect4Board("4455")
        harness = GameHarness(Connect4(), driver=ScriptedDriver([reply("3")]), level=4)
        assert isinstance(harness.engine, Connect4Engine)
        assert [c.name for c in harness.candidates(board)][:2] in (["3", "6"], ["6", "3"])
        assert "engine: wins in 2" in harness.prompt(board, harness.candidates(board))

    def test_level_2_hides_losing_moves(self):
        board = Connect4Board("14624615")  # Red must block column 3
        harness = GameHarness(
            Connect4(), driver=ScriptedDriver([reply("1"), reply("3")]), level=2, engine=Connect4Engine(4)
        )
        assert [c.name for c in harness.candidates(board)] == ["3"]
        decision = harness.choose(board)
        assert decision.attempts[0].error == "1 is legal but not one of your listed moves"
        assert decision.name == "3"

    def test_othello_pass_is_offered_and_named(self):
        board = othello_from_rows(["BW......"] + ["........"] * 7, turn=2)
        decision = GameHarness(Othello(), driver=ScriptedDriver([reply("pass")]), level=1).choose(board)
        assert decision.move == PASS and decision.name == "pass"
        assert decision.candidates[0].motifs[0].text == "passes: no other move is possible"

    def test_rescue_plays_the_engine_move(self):
        board = Connect4Board("445566")
        harness = GameHarness(Connect4(), driver=ScriptedDriver([reply("x")]), level=3, max_retries=0)
        decision = harness.choose(board)
        assert decision.rescued and decision.name in ("3", "7")  # both connect four

    def test_failed_model_call_is_a_failed_attempt(self):
        class DeadDriver(ScriptedDriver):
            def generate(self, prompt, options):
                raise RuntimeError("Ollama chat request failed: read timed out")

        decision = GameHarness(Connect4(), driver=DeadDriver([""]), max_retries=1, seed=0).choose(Connect4Board())
        assert decision.rescued
        assert decision.attempts[0].error == "the model call failed (Ollama chat request failed: read timed out)"

    def test_all_concepts_and_unknown_ones(self):
        harness = GameHarness(Othello(), driver=ScriptedDriver([reply("d3")]), concepts="all")
        assert harness.concepts == frozenset(Othello.concepts)
        with pytest.raises(ValueError, match="Unknown Othello concepts"):
            GameHarness(Othello(), driver=ScriptedDriver([reply("d3")]), concepts={"fork"})


class TestMatches:
    def test_engine_beats_random_at_connect4(self):
        game = Connect4()
        result = play_match(game, {"Red": EnginePlayer(game, Connect4Engine(4)), "Yellow": RandomPlayer(game, seed=3)})
        assert result.finished and result.winner == "Red"
        assert game.is_over(result.state)

    def test_model_versus_random_records_rescues(self):
        game = Connect4()
        model = GameHarness(game, driver=ScriptedDriver([reply("nonsense")]), max_retries=0, seed=1)
        result = play_match(game, {"Red": model, "Yellow": RandomPlayer(game, seed=2)})
        assert result.rescues["Red"] == sum(1 for side, _ in result.decisions if side == "Red")

    def test_othello_game_runs_to_the_end(self):
        game = Othello()
        result = play_match(game, {"Black": RandomPlayer(game, seed=5), "White": RandomPlayer(game, seed=6)})
        assert result.finished and game.is_over(result.state)

    def test_missing_player_and_move_cap(self):
        game = Connect4()
        with pytest.raises(ValueError, match="No player for Yellow"):
            play_match(game, {"Red": RandomPlayer(game)})
        capped = play_match(game, {"Red": RandomPlayer(game), "Yellow": RandomPlayer(game)}, max_moves=3)
        assert not capped.finished and len(capped.moves) == 3
