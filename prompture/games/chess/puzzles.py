"""Puzzle runs: measure a harness setting on themed puzzles.

Puzzles use the Lichess / Chess Studio format: ``fen`` is the position before
the opponent's last move and ``moves`` is the UCI line from there. The first
move is the opponent's; then the solver and the replies alternate, ending on
the solver's move. On the final move any checkmate counts, as on Lichess.

Running one theme's pack with a concept off and then on shows whether the
concept helps a model::

    pack = load_puzzles("puzzles/fork.json")
    plain = run_puzzles(ChessHarness(model), pack)
    taught = run_puzzles(ChessHarness(model, concepts={"fork"}), pack)
    print(plain.solve_rate, taught.solve_rate)
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

from .harness import ChessHarness, MoveDecision


@dataclass
class PuzzleResult:
    """How the model did on one puzzle."""

    id: str
    solved: bool
    played: list[str]
    expected: list[str]
    rescues: int
    decisions: list[MoveDecision] = field(default_factory=list)


@dataclass
class PuzzleReport:
    """Totals over a puzzle run."""

    results: list[PuzzleResult]

    @property
    def solved(self) -> int:
        return sum(r.solved for r in self.results)

    @property
    def solve_rate(self) -> float:
        return self.solved / len(self.results) if self.results else 0.0

    @property
    def rescues(self) -> int:
        return sum(r.rescues for r in self.results)


def load_puzzles(path: str | Path) -> list[dict[str, Any]]:
    """Puzzles from a pack file: ``{"puzzles": [...]}`` or a bare list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    puzzles: list[dict[str, Any]] = data["puzzles"] if isinstance(data, dict) else data
    return puzzles


def play_puzzle(harness: ChessHarness, puzzle: dict[str, Any]) -> PuzzleResult:
    """Play the solver's side of one puzzle; stop at the first wrong move."""
    board = chess.Board(puzzle["fen"])
    line = [chess.Move.from_uci(uci) for uci in puzzle["moves"]]
    board.push(line[0])
    played: list[str] = []
    expected: list[str] = []
    decisions: list[MoveDecision] = []
    solved = True
    for index in range(1, len(line), 2):
        wanted = line[index]
        expected.append(board.san(wanted))
        decision = harness.choose(board)
        decisions.append(decision)
        played.append(decision.san)
        board.push(decision.move)
        last = index == len(line) - 1
        if decision.move != wanted and not (last and board.is_checkmate()):
            solved = False
            break
        if not last:
            board.push(line[index + 1])
    return PuzzleResult(
        str(puzzle.get("id", "")),
        solved,
        played,
        expected,
        sum(d.rescued for d in decisions),
        decisions,
    )


def run_puzzles(harness: ChessHarness, puzzles: Iterable[dict[str, Any]]) -> PuzzleReport:
    """Play every puzzle and total the results."""
    return PuzzleReport([play_puzzle(harness, puzzle) for puzzle in puzzles])
