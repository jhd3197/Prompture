"""Othello (Reversi) for the game harness: rules, move tags and a built-in engine.

The board is 8x8 with columns ``a``-``h`` left to right and rows ``1``-``8``
top to bottom, as in Othello notation. Black moves first from the standard
start (d4, e5 white; d5, e4 black). A move places a disc that brackets one or
more opponent discs in a straight line; those flip. A player with no such
move must pass (the move ``"pass"``); when neither can move, the side with
more discs wins.

Tags:

* **Facts**: how many discs a move flips, and passing.
* **Concepts** (:data:`CONCEPTS`): ``corner`` (corners can never be flipped),
  ``x-square`` (the square diagonally inside an empty corner, which usually
  gives that corner away), ``edge``, ``mobility`` (how many moves the move
  leaves the opponent; leaving none forces a pass) and ``safety`` (the move
  hands the opponent a corner they could not take before).

:class:`OthelloEngine` is a depth-limited alpha-beta search on a positional
weight table plus mobility, so every harness level works without installing
anything.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from .base import Engine, Motif, Score, check_concepts

SIZE = 8
BLACK, WHITE = 1, 2
NAMES = {BLACK: "Black", WHITE: "White"}
SYMBOLS = {0: ".", BLACK: "B", WHITE: "W"}
PASS = -1
CONCEPTS = ("corner", "x-square", "edge", "mobility", "safety")
CORNERS = (0, 7, 56, 63)
X_SQUARES = {9: 0, 14: 7, 49: 56, 54: 63}  # X-square -> its corner
_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_SQUARE = re.compile(r"^([a-h])([1-8])$")
_WEIGHTS = (
    (100, -20, 10, 5, 5, 10, -20, 100),
    (-20, -50, -2, -2, -2, -2, -50, -20),
    (10, -2, -1, -1, -1, -1, -2, 10),
    (5, -2, -1, -1, -1, -1, -2, 5),
    (5, -2, -1, -1, -1, -1, -2, 5),
    (10, -2, -1, -1, -1, -1, -2, 10),
    (-20, -50, -2, -2, -2, -2, -50, -20),
    (100, -20, 10, 5, 5, 10, -20, 100),
)
WEIGHTS = [w for row in _WEIGHTS for w in row]


def square_name(index: int) -> str:
    return "pass" if index == PASS else "abcdefgh"[index % SIZE] + str(index // SIZE + 1)


class OthelloBoard:
    """An Othello position; ``moves`` replays a game from the start (names like ``"d3"`` or indices)."""

    def __init__(self, moves: Iterable[str | int] = ()) -> None:
        self.cells = [0] * (SIZE * SIZE)  # index = row * 8 + col, row 0 is rank 1 (top)
        self.cells[27] = self.cells[36] = WHITE  # d4, e5
        self.cells[28] = self.cells[35] = BLACK  # e4, d5
        self.turn = BLACK
        self.moves: list[int] = []
        for move in moves:
            self.play(parse_square(move) if isinstance(move, str) else move)

    def copy(self) -> OthelloBoard:
        board = OthelloBoard.__new__(OthelloBoard)
        board.cells, board.turn, board.moves = self.cells[:], self.turn, self.moves[:]
        return board

    def flips(self, index: int, who: int) -> list[int]:
        """Discs a ``who`` disc on ``index`` would flip (empty if the move is illegal)."""
        if self.cells[index]:
            return []
        row, col = divmod(index, SIZE)
        opponent = 3 - who
        flipped: list[int] = []
        for dr, dc in _DIRECTIONS:
            r, c = row + dr, col + dc
            line = []
            while 0 <= r < SIZE and 0 <= c < SIZE and self.cells[r * SIZE + c] == opponent:
                line.append(r * SIZE + c)
                r, c = r + dr, c + dc
            if line and 0 <= r < SIZE and 0 <= c < SIZE and self.cells[r * SIZE + c] == who:
                flipped += line
        return flipped

    def placements(self, who: int) -> list[int]:
        """Squares where ``who`` may place a disc."""
        return [i for i in range(SIZE * SIZE) if not self.cells[i] and self.flips(i, who)]

    def legal_moves(self) -> list[int]:
        """Placements for the side to move; ``[PASS]`` if it has none but the game goes on; ``[]`` at the end."""
        mine = self.placements(self.turn)
        if mine:
            return mine
        return [PASS] if self.placements(3 - self.turn) else []

    def play(self, index: int) -> None:
        if index == PASS:
            if self.legal_moves() != [PASS]:
                raise ValueError("Passing is only allowed with no other move")
        else:
            flipped = self.flips(index, self.turn) if 0 <= index < SIZE * SIZE else []
            if not flipped:
                raise ValueError(f"{square_name(index)} is not a legal move")
            for square in [index, *flipped]:
                self.cells[square] = self.turn
        self.moves.append(index)
        self.turn = 3 - self.turn

    def count(self, who: int) -> int:
        return self.cells.count(who)

    def is_over(self) -> bool:
        return not self.legal_moves()


def parse_square(text: str) -> int:
    text = text.strip().lower()
    if text == "pass":
        return PASS
    match = _SQUARE.match(text)
    if not match:
        raise ValueError(f"Not an Othello square: {text!r}")
    return (int(match.group(2)) - 1) * SIZE + "abcdefgh".index(match.group(1))


def _is_edge(index: int) -> bool:
    row, col = divmod(index, SIZE)
    return row in (0, SIZE - 1) or col in (0, SIZE - 1)


def describe_move(
    board: OthelloBoard, index: int, concepts: Iterable[str] = CONCEPTS, *, facts: bool = True
) -> list[Motif]:
    """Tags for playing ``index`` (or :data:`PASS`): facts (if ``facts``) then the enabled concepts."""
    wanted = check_concepts(Othello(), concepts)
    me, opponent = board.turn, 3 - board.turn
    if index == PASS:
        return [Motif("pass", "passes: no other move is possible")] if facts else []
    flipped = board.flips(index, me)
    after = board.copy()
    after.play(index)
    motifs = []
    if facts:
        motifs.append(Motif("flips", f"flips {len(flipped)} disc{'s' if len(flipped) != 1 else ''}"))
    name = square_name(index)
    if "corner" in wanted and index in CORNERS:
        motifs.append(Motif("corner", f"takes the corner {name}; it can never be flipped", (name,)))
    if "x-square" in wanted and index in X_SQUARES and not board.cells[X_SQUARES[index]]:
        corner = square_name(X_SQUARES[index])
        motifs.append(
            Motif("x-square", f"warning: X-square next to the empty corner {corner}, which it can give away", (corner,))
        )
    if "edge" in wanted and _is_edge(index) and index not in CORNERS:
        motifs.append(Motif("edge", "takes an edge square"))
    replies = after.placements(opponent)
    if "mobility" in wanted:
        if replies:
            motifs.append(
                Motif("mobility", f"leaves {NAMES[opponent]} {len(replies)} move{'s' if len(replies) != 1 else ''}")
            )
        elif after.legal_moves():
            motifs.append(Motif("mobility", f"leaves {NAMES[opponent]} no move: they must pass"))
    if "safety" in wanted:
        before = set(board.placements(opponent))
        for gift in sorted(set(replies) & set(CORNERS) - before):
            motifs.append(
                Motif(
                    "safety",
                    f"warning: gives {NAMES[opponent]} the corner {square_name(gift)}",
                    (square_name(gift),),
                )
            )
    return motifs


class Othello:
    """Othello rules and notation for the harness."""

    title = "Othello"
    concepts: tuple[str, ...] = CONCEPTS
    rules = (
        "Rules: place a disc so that it brackets one or more of the opponent's discs in a straight line "
        "(across, down or diagonal) between it and another of your discs; the bracketed discs flip to your color. "
        "If you have no such move you must pass. When neither side can move, most discs wins."
    )

    def new_state(self) -> OthelloBoard:
        return OthelloBoard()

    def player(self, board: OthelloBoard) -> str:
        return NAMES[board.turn]

    def legal_moves(self, board: OthelloBoard) -> list[int]:
        return board.legal_moves()

    def move_name(self, board: OthelloBoard, index: int) -> str:
        return square_name(index)

    def parse_move(self, board: OthelloBoard, text: str) -> int | None:
        try:
            index = parse_square(text)
        except ValueError:
            return None
        return index if index in board.legal_moves() else None

    def describe_move(
        self, board: OthelloBoard, index: int, concepts: Iterable[str], *, facts: bool = True
    ) -> list[Motif]:
        return describe_move(board, index, concepts, facts=facts)

    def render(self, board: OthelloBoard) -> list[str]:
        lines = []
        if board.moves:
            lines.append("Moves so far: " + " ".join(square_name(m) for m in board.moves))
        lines.append("Board (B = Black, W = White, . = empty):")
        lines.append("  " + " ".join("abcdefgh"))
        for row in range(SIZE):
            lines.append(f"{row + 1} " + " ".join(SYMBOLS[board.cells[row * SIZE + c]] for c in range(SIZE)))
        lines.append(f"Discs: Black {board.count(BLACK)}, White {board.count(WHITE)}")
        if board.legal_moves() == [PASS]:
            lines.append(f"{NAMES[board.turn]} has no legal placement and must pass.")
        else:
            lines.append(f"{NAMES[board.turn]} to move.")
        return lines

    def apply(self, board: OthelloBoard, index: int) -> OthelloBoard:
        after = board.copy()
        after.play(index)
        return after

    def is_over(self, board: OthelloBoard) -> bool:
        return board.is_over()

    def winner(self, board: OthelloBoard) -> str | None:
        black, white = board.count(BLACK), board.count(WHITE)
        if not board.is_over() or black == white:
            return None
        return NAMES[BLACK] if black > white else NAMES[WHITE]

    def default_engine(self) -> Engine:
        return OthelloEngine()


class OthelloEngine:
    """Alpha-beta search over Othello. ``depth`` is in plies; passes count as plies."""

    def __init__(self, depth: int = 4) -> None:
        self.depth = depth

    def analyse(self, board: OthelloBoard, count: int) -> dict[Any, Score]:
        scores = {}
        for move in board.legal_moves():
            child = board.copy()
            child.play(move)
            value = -self._search(child, self.depth - 1, -math.inf, math.inf)
            scores[move] = _score(value)
        return scores

    def _search(self, board: OthelloBoard, depth: int, alpha: float, beta: float) -> float:
        moves = board.legal_moves()
        if not moves:
            diff = board.count(board.turn) - board.count(3 - board.turn)
            return (10_000 + abs(diff)) * (1 if diff > 0 else -1) if diff else 0
        if depth == 0:
            return evaluate(board)
        moves.sort(key=lambda m: -WEIGHTS[m] if m != PASS else 0)
        best = -math.inf
        for move in moves:
            child = board.copy()
            child.play(move)
            value = -self._search(child, depth - 1, -beta, -alpha)
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best


def evaluate(board: OthelloBoard) -> int:
    """Heuristic score for the side to move: square weights plus mobility."""
    me, opponent = board.turn, 3 - board.turn
    score = 0
    for index, value in enumerate(board.cells):
        if value == me:
            score += WEIGHTS[index]
        elif value == opponent:
            score -= WEIGHTS[index]
    return score + 5 * (len(board.placements(me)) - len(board.placements(opponent)))


def _score(value: float) -> Score:
    """Search value (for the mover) as winning chances and a label."""
    if abs(value) >= 10_000:
        margin = int(abs(value) - 10_000)
        if value > 0:
            return Score(1.0, f"wins by {margin} discs")
        return Score(-1.0, f"loses by {margin} discs")
    return Score(math.tanh(value / 60), f"{int(value):+d}")
