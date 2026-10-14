"""Bounded own-decision memory, checkpoints and context budgets across games."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from prompture.drivers.base import Driver
from prompture.games import Connect4, Connect4Board, FailureKind, GameHarness, Othello, OthelloBoard, TurnFailed
from prompture.games.memory import CheckpointError, DecisionMemory

PASS_LINE = [
    "d3",
    "c5",
    "b6",
    "b5",
    "b4",
    "e3",
    "f5",
    "c3",
    "c2",
    "f4",
    "b3",
    "d2",
    "e2",
    "b2",
    "f3",
    "g3",
    "a1",
    "d1",
    "h2",
    "f6",
    "g6",
    "e6",
    "g2",
    "g5",
    "f7",
    "a2",
    "h4",
    "e7",
    "e8",
    "d6",
    "c4",
    "a7",
    "d7",
    "a3",
    "f1",
    "g4",
    "h3",
    "a6",
    "b1",
    "c1",
    "e1",
    "g7",
    "h7",
    "h1",
    "a5",
    "h8",
    "a4",
    "c6",
    "c7",
    "h6",
    "a8",
    "b7",
    "h5",
    "f8",
    "d8",
    "c8",
    "f2",
    "g1",
]


class Planner(Driver):
    """Plays the first listed move (or a scripted one) and labels each plan with the call number."""

    supports_messages = True
    model = "scripted/planner"

    def __init__(self, moves: list[str] | None = None):
        self.moves = list(moves or [])
        self.prompts: list[str] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        n = len(self.prompts)
        side = re.search(r"as (\w+)\.", prompt).group(1)
        move = self.moves.pop(0) if self.moves else re.search(r"^- ([^:\n]+)", prompt, re.M).group(1).strip()
        text = json.dumps({"reason": f"{side} reason {n}", "plan": f"{side} plan {n}", "move": move})
        return {"text": text, "meta": {"total_tokens": 1}}


def play(harness: GameHarness, state: Any, turns: int) -> Any:
    for _ in range(turns):
        decision = harness.choose(state)
        state = harness.game.apply(state, decision.move)
    return state


class TestPolicies:
    def test_generic_harness_defaults_to_no_memory(self):
        driver = Planner()
        play(GameHarness(Connect4(), driver=driver), Connect4Board(), 3)
        assert "reason 1" not in driver.prompts[2]

    def test_window_shows_own_accepted_decisions_oldest_first(self):
        driver = Planner()
        harness = GameHarness(Connect4(), driver=driver, memory=4)
        play(harness, Connect4Board(), 7)
        last = driver.prompts[6]  # Red's fourth turn
        shown = re.findall(r"- ply (\d+), \d: (Red reason \d+); plan: (Red plan \d+)", last)
        assert [ply for ply, _, _ in shown] == ["1", "3", "5"]
        assert "Yellow" not in last.split("Your recent decisions")[1]
        assert "not orders" in last

    def test_window_is_bounded(self):
        driver = Planner()
        play(GameHarness(Connect4(), driver=driver, memory=1), Connect4Board(), 7)
        assert re.findall(r"- ply (\d+),", driver.prompts[6]) == ["5"]

    def test_previous_is_only_the_next_own_turn(self):
        driver = Planner()
        harness = GameHarness(Othello(), driver=driver, memory="previous")
        state = play(harness, OthelloBoard(), 2)
        harness.choose(state)
        assert "Your previous move:" in driver.prompts[2] and "Black plan 1" in driver.prompts[2]

    def test_memory_needs_line_identity(self):
        class Bare(Connect4):
            history = None  # type: ignore[assignment]

        with pytest.raises(ValueError, match="no history"):
            GameHarness(Bare(), driver=Planner(), memory=4)
        assert GameHarness(Bare(), driver=Planner()).choose(Connect4Board()).name

    def test_policy_is_part_of_the_fingerprint(self):
        digests = {
            GameHarness(Connect4(), driver=Planner(), memory=m).fingerprint().digest for m in ("none", "previous", 4, 8)
        }
        assert len(digests) == 4
        with pytest.raises(ValueError, match="memory must be"):
            DecisionMemory(Connect4(), "always")


class TestCommitSemantics:
    def test_repeated_queries_do_not_duplicate_history(self):
        driver = Planner()
        harness = GameHarness(Connect4(), driver=driver, memory=8)
        board = Connect4Board()
        for _ in range(3):
            decision = harness.choose(board)  # asked three times, applied once
        board = Connect4Board([decision.move])
        board = play(harness, board, 1)  # Yellow
        harness.choose(board)
        assert len(re.findall(r"- ply \d+,", driver.prompts[-1])) == 1
        assert "Red reason 3" in driver.prompts[-1]

    def test_a_move_the_host_did_not_play_is_never_remembered(self):
        driver = Planner()
        harness = GameHarness(Connect4(), driver=driver, memory=4)
        harness.choose(Connect4Board())  # model suggests column 1, host plays 4 instead
        board = play(harness, Connect4Board([3]), 1)
        harness.choose(board)
        assert "Your recent decisions" not in driver.prompts[-1]

    def test_explicit_commit(self):
        driver = Planner()
        harness = GameHarness(Connect4(), driver=driver, memory=4)
        board = Connect4Board()
        decision = harness.choose(board)
        harness.commit(board, decision)
        checkpoint = harness.memory.checkpoint(Connect4Board([decision.move, 0]))
        assert [e["move"] for e in checkpoint["entries"]] == [decision.name]

    def test_rejected_attempts_and_rescues_leave_no_invented_plans(self):
        driver = Planner(moves=["9", "9", "9"])
        harness = GameHarness(Connect4(), driver=driver, memory=4, max_retries=2, seed=1)
        board = Connect4Board()
        rescued = harness.choose(board)
        assert rescued.rescued
        board = play(harness, harness.game.apply(board, rescued.move), 1)
        harness.choose(board)
        assert "Your recent decisions" not in driver.prompts[-1]
        red = [e for e in harness.memory.checkpoint(board)["entries"] if e["side"] == "Red"]
        assert red and all(e["reason"] == e["plan"] == "" and e["rescued"] for e in red)


class TestIsolation:
    def test_othello_pass_keeps_the_movers_previous_plan(self):
        driver = Planner(moves=["b8", "g8"])
        harness = GameHarness(Othello(), driver=driver, memory="previous")
        game = Othello()
        before = game.load_state({"initial": None, "moves": PASS_LINE})
        assert game.player(before) == "Black"
        state = game.apply(before, harness.choose(before).move)
        assert game.legal_moves(state) == [-1]  # White must pass
        state = game.apply(state, -1)
        assert game.player(state) == "Black"
        harness.choose(state)
        assert "Your previous move: b8" in driver.prompts[-1] and "Black plan 1" in driver.prompts[-1]

    def test_no_memory_crosses_games_or_branches(self):
        driver = Planner()
        harness = GameHarness(Connect4(), driver=driver, memory=8)
        play(harness, Connect4Board(), 4)
        harness.choose(Connect4Board())  # a new game from the start
        assert "Your recent decisions" not in driver.prompts[-1]
        harness.choose(Connect4Board([6, 6]))  # another branch
        assert "Your recent decisions" not in driver.prompts[-1]


def _resume_matches(game: Any, harness_factory, opening: list[Any], turns: int) -> None:
    driver_a = Planner()
    first = harness_factory(driver_a)
    state = play(first, opening, turns)
    checkpoint = json.loads(json.dumps(first.memory.checkpoint(state)))
    first.choose(state)
    restored = game.load_state(game.dump_state(state))
    driver_b = Planner()
    second = harness_factory(driver_b)
    second.memory.restore(checkpoint, restored)
    second.choose(restored)
    assert driver_a.prompts[-1] == driver_b.prompts[-1]
    assert "reason" in driver_b.prompts[-1].split("Give a short reason")[1]


class TestCheckpoints:
    def test_resumed_games_get_the_same_next_context(self):
        _resume_matches(Connect4(), lambda d: GameHarness(Connect4(), driver=d, memory=4), Connect4Board(), 5)
        _resume_matches(Othello(), lambda d: GameHarness(Othello(), driver=d, memory="previous"), OthelloBoard(), 5)

    def test_checkpoint_validation(self):
        harness = GameHarness(Connect4(), driver=Planner(), memory=4)
        state = play(harness, Connect4Board(), 3)
        checkpoint = harness.memory.checkpoint(state)
        other = DecisionMemory(Connect4(), 4)
        with pytest.raises(CheckpointError, match="diverged"):
            other.restore(checkpoint, Connect4Board([5, 5, 5]))
        with pytest.raises(CheckpointError, match="version"):
            other.restore({**checkpoint, "memory_version": "0"}, state)
        with pytest.raises(CheckpointError, match="is for"):
            DecisionMemory(Othello(), 4).restore(checkpoint, OthelloBoard())
        with pytest.raises(CheckpointError, match="initial position"):
            other.restore({**checkpoint, "line": {"root": "elsewhere", "plies": []}}, state)
        later = Connect4Board([*state.moves, 0])
        other.restore(checkpoint, later)  # resuming after more moves is fine
        assert len(other.checkpoint(later)["entries"]) == 3

    def test_legacy_games_resume_without_memory_explicitly(self):
        memory = DecisionMemory(Connect4(), 4)
        memory.restore(None, Connect4Board([3, 3]))
        assert memory.notes == ["resumed without saved memory (legacy game)"]
        assert memory.checkpoint(Connect4Board([3, 3]))["entries"] == []


class TestContextBudget:
    def harness(self, budget: int | None, memory: Any = "none") -> GameHarness:
        return GameHarness(Connect4(), driver=Planner(), level=1, concepts="all", context_budget=budget, memory=memory)

    def test_compact_prompt_keeps_every_move_and_urgent_facts(self):
        board = Connect4Board("4455661")  # Yellow: block at 7 is urgent, other moves lose
        full = self.harness(None).prompt(board, self.harness(None).candidates(board))
        harness = self.harness(len(full) // 4 - 15)  # room for every urgent note, not every fact
        candidates = harness.candidates(board)
        compact, info = harness._compose(board, candidates)
        assert len(compact) < len(full) and info["omitted_notes"] > 0
        for candidate in candidates:
            assert f"- {candidate.name}" in compact
            for motif in candidate.motifs:
                if motif.severity == "urgent":
                    assert motif.text in compact
        assert "left out to fit the context budget" in compact
        assert compact.rstrip().endswith("then the move you choose.")
        assert info["estimated_tokens"] <= harness.context_budget and info["estimate"] == "chars/4"

    def test_required_content_that_cannot_fit_fails_clearly(self):
        with pytest.raises(TurnFailed) as caught:
            self.harness(20).choose(Connect4Board())
        assert caught.value.kind is FailureKind.CONFIGURATION and "does not fit" in str(caught.value)

    def test_memory_is_dropped_before_the_menu(self):
        driver = Planner()
        roomy = GameHarness(Connect4(), driver=driver, memory=8)
        state = play(roomy, Connect4Board(), 4)
        full = roomy.prompt(state, roomy.candidates(state))
        assert "Your recent decisions" in full
        tight = GameHarness(Connect4(), driver=Planner(), memory=8, context_budget=(len(full) - 60) // 4)
        tight.memory.restore(roomy.memory.checkpoint(state), state)
        text, info = tight._compose(state, tight.candidates(state))
        assert info["memory_omitted"] and "Your recent decisions" not in text
        assert all(f"- {c}" in text for c in "1234567")

    def test_record_keeps_prompt_sizing(self):
        decision = self.harness(5000).choose(Connect4Board())
        assert decision.record.context["prompt"]["budget"] == 5000
        assert decision.record.context["prompt"]["omitted_notes"] == 0


chess = pytest.importorskip("chess")


class TestChessMemory:
    def test_default_previous_and_window_policies(self):
        from prompture.games.chess import ChessHarness

        driver = Planner()
        harness = ChessHarness(driver=driver)
        assert harness.memory.policy == "previous"
        play(harness, chess.Board(), 3)
        assert "Your previous move:" in driver.prompts[2] and "White plan 1" in driver.prompts[2]
        windowed = Planner()
        play(ChessHarness(driver=windowed, memory=4), chess.Board(), 5)
        assert re.findall(r"- ply (\d+),", windowed.prompts[4]) == ["1", "3"]

    def test_no_memory_crosses_into_a_puzzle(self):
        from prompture.games.chess import ChessHarness, play_puzzle

        driver = Planner(moves=["e4", "e5", "Ne7+", "Nxc8"])
        harness = ChessHarness(driver=driver, memory=8)
        play(harness, chess.Board(), 2)
        puzzle = {"id": "fork", "fen": "2q2k2/8/8/3N4/8/8/8/6K1 b - - 0 1", "moves": ["f8g8", "d5e7", "g8f7", "e7c8"]}
        play_puzzle(harness, puzzle)
        assert all("White plan 1" not in p for p in driver.prompts[2:])

    def test_chess_checkpoint_resume(self):
        from prompture.games.chess import ChessGame, ChessHarness

        _resume_matches(ChessGame(), lambda d: ChessHarness(driver=d, memory=4), chess.Board(), 6)
