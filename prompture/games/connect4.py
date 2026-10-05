"""Connect Four for the game harness: rules, move tags and a built-in engine.

The board is 7 columns by 6 rows. Red moves first, then Yellow. A move is a
column, named ``"1"`` to ``"7"`` from the left; the disc falls to the lowest
empty row (rows count ``1`` to ``6`` from the bottom). Four in a row in any
direction wins; a full board is a draw.

Tags:

* **Facts**: the row the disc lands on, and whether it fills the column.
* **Concepts** (:data:`CONCEPTS`): ``win`` (connects four), ``block`` (stops
  the opponent's four), ``threat`` (makes a new three with the fourth cell
  empty), ``double-threat`` (two ways to win next move: only one can be
  blocked), ``safety`` (the move lets the opponent win next move, often by
  filling the cell under their winning cell) and ``center`` (the middle
  column, part of the most lines).

:class:`Connect4Engine` is a depth-limited alpha-beta search, so every harness
level works without installing anything.

Example::

    from prompture.games import GameHarness
    from prompture.games.connect4 import Connect4

    game = Connect4()
    harness = GameHarness(game, "ollama/llama3.2:3b", level=2, concepts={"block", "safety"})
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .base import Engine, Motif, Score, check_concepts

ROWS, COLS = 6, 7
RED, YELLOW = 1, 2
NAMES = {RED: "Red", YELLOW: "Yellow"}
SYMBOLS = {0: ".", RED: "R", YELLOW: "Y"}
CONCEPTS = ("win", "block", "threat", "double-threat", "safety", "center")
CENTER = COLS // 2
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
# Every line of four cells, as flat indices (row * COLS + col).
WINDOWS = [
    tuple((r + i * dr) * COLS + (c + i * dc) for i in range(4))
    for r in range(ROWS)
    for c in range(COLS)
    for dr, dc in _DIRECTIONS
    if 0 <= r + 3 * dr < ROWS and 0 <= c + 3 * dc < COLS
]
_ORDER = sorted(range(COLS), key=lambda c: abs(c - CENTER))  # centre-first search order
_WIN = 1_000_000


class Connect4Board:
    """A Connect Four position. ``moves`` replays a game: column indices or a string of 1-7 digits."""

    def __init__(self, moves: Iterable[int] | str = ()) -> None:
        self.cells = [0] * (ROWS * COLS)  # row 0 is the bottom
        self.heights = [0] * COLS
        self.moves: list[int] = []
        self.won = 0
        for col in [int(ch) - 1 for ch in moves] if isinstance(moves, str) else moves:
            self.play(col)

    @property
    def turn(self) -> int:
        return RED if len(self.moves) % 2 == 0 else YELLOW

    def copy(self) -> Connect4Board:
        board = Connect4Board.__new__(Connect4Board)
        board.cells, board.heights, board.moves, board.won = self.cells[:], self.heights[:], self.moves[:], self.won
        return board

    def legal_moves(self) -> list[int]:
        if self.won:
            return []
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    def play(self, col: int) -> None:
        if not 0 <= col < COLS or self.heights[col] >= ROWS or self.won:
            raise ValueError(f"Column {col + 1} cannot be played")
        row = self.heights[col]
        who = self.turn
        self.cells[row * COLS + col] = who
        self.heights[col] += 1
        self.moves.append(col)
        if self.connects(row, col, who):
            self.won = who

    def undo(self) -> None:
        col = self.moves.pop()
        self.heights[col] -= 1
        self.cells[self.heights[col] * COLS + col] = 0
        self.won = 0

    def connects(self, row: int, col: int, who: int) -> bool:
        """Whether a ``who`` disc on (row, col) is part of four in a row."""
        for dr, dc in _DIRECTIONS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.cells[r * COLS + c] == who:
                    count += 1
                    r, c = r + sign * dr, c + sign * dc
            if count >= 4:
                return True
        return False

    def winning_cells(self, who: int) -> set[tuple[int, int]]:
        """Empty cells where a ``who`` disc would connect four, playable now or not."""
        found = set()
        for index, value in enumerate(self.cells):
            if value:
                continue
            row, col = divmod(index, COLS)
            self.cells[index] = who
            if self.connects(row, col, who):
                found.add((row, col))
            self.cells[index] = 0
        return found

    def immediate_wins(self, who: int) -> set[int]:
        """Columns where ``who`` would connect four by playing now."""
        return {c for r, c in self.winning_cells(who) if self.heights[c] == r}

    def is_full(self) -> bool:
        return all(h == ROWS for h in self.heights)


def cell_name(row: int, col: int) -> str:
    return f"column {col + 1}, row {row + 1}"


def describe_move(
    board: Connect4Board, col: int, concepts: Iterable[str] = CONCEPTS, *, facts: bool = True
) -> list[Motif]:
    """Tags for dropping a disc into ``col``: facts (if ``facts``) then the enabled concepts."""
    wanted = check_concepts(Connect4(), concepts)
    me, opponent = board.turn, 3 - board.turn
    row = board.heights[col]
    after = board.copy()
    after.play(col)
    motifs = []
    if facts:
        motifs.append(Motif("row", f"lands on row {row + 1}"))
        if row == ROWS - 1:
            motifs.append(Motif("full", f"fills column {col + 1}"))
    if after.won:
        if "win" in wanted:
            motifs.append(Motif("win", "wins: connects four"))
        return motifs
    if "block" in wanted and col in board.immediate_wins(opponent):
        motifs.append(Motif("block", f"blocks {NAMES[opponent]}'s four in column {col + 1}"))
    opponent_wins = after.immediate_wins(opponent)
    mine_after = after.winning_cells(me)
    playable = sorted(c for r, c in mine_after if after.heights[c] == r)
    double = "double-threat" in wanted and len(playable) >= 2 and not opponent_wins
    if double:
        columns = " and ".join(str(c + 1) for c in playable)
        motifs.append(
            Motif("double-threat", f"double threat: wins next move in columns {columns}; only one can be blocked")
        )
    if "threat" in wanted and not double:
        for r, c in sorted(mine_after - board.winning_cells(me)):
            now = " (playable now)" if after.heights[c] == r else ""
            motifs.append(Motif("threat", f"threat: four would connect at {cell_name(r, c)}{now}"))
    if "safety" in wanted and opponent_wins:
        columns = ", ".join(str(c + 1) for c in sorted(opponent_wins))
        motifs.append(Motif("safety", f"warning: lets {NAMES[opponent]} win next move in column {columns}"))
    if "center" in wanted and col == CENTER:
        motifs.append(Motif("center", "takes the center column, part of the most lines"))
    return motifs


class Connect4:
    """Connect Four rules and notation for the harness."""

    title = "Connect Four"
    concepts: tuple[str, ...] = CONCEPTS
    rules = (
        "Rules: drop a disc into a column (1-7, left to right); it falls to the lowest empty row "
        "(rows 1-6, bottom to top). Four of your discs in a row, column or diagonal wins."
    )

    def new_state(self) -> Connect4Board:
        return Connect4Board()

    def player(self, board: Connect4Board) -> str:
        return NAMES[board.turn]

    def legal_moves(self, board: Connect4Board) -> list[int]:
        return board.legal_moves()

    def move_name(self, board: Connect4Board, col: int) -> str:
        return str(col + 1)

    def parse_move(self, board: Connect4Board, text: str) -> int | None:
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) != 1:
            return None
        col = int(digits) - 1
        return col if col in board.legal_moves() else None

    def describe_move(
        self, board: Connect4Board, col: int, concepts: Iterable[str], *, facts: bool = True
    ) -> list[Motif]:
        return describe_move(board, col, concepts, facts=facts)

    def render(self, board: Connect4Board) -> list[str]:
        lines = []
        if board.moves:
            lines.append("Columns played so far: " + " ".join(str(c + 1) for c in board.moves))
        lines.append("Board (R = Red, Y = Yellow, . = empty; top row first):")
        lines.append("    " + " ".join(str(c + 1) for c in range(COLS)))
        for row in reversed(range(ROWS)):
            lines.append(f"{row + 1} | " + " ".join(SYMBOLS[board.cells[row * COLS + c]] for c in range(COLS)))
        for col in range(COLS):
            stack = [NAMES[board.cells[r * COLS + col]] for r in range(board.heights[col])]
            lines.append(f"Column {col + 1} (bottom up): {', '.join(stack) if stack else 'empty'}")
        lines.append(f"{NAMES[board.turn]} to move.")
        return lines

    def apply(self, board: Connect4Board, col: int) -> Connect4Board:
        after = board.copy()
        after.play(col)
        return after

    def is_over(self, board: Connect4Board) -> bool:
        return bool(board.won) or board.is_full()

    def winner(self, board: Connect4Board) -> str | None:
        return NAMES[board.won] if board.won else None

    def default_engine(self) -> Engine:
        return Connect4Engine()


class Connect4Engine:
    """Alpha-beta search over Connect Four. ``depth`` is in plies (single moves)."""

    def __init__(self, depth: int = 6) -> None:
        self.depth = depth

    def analyse(self, board: Connect4Board, count: int) -> dict[Any, Score]:
        work = board.copy()
        scores = {}
        for col in work.legal_moves():
            work.play(col)
            value = -self._search(work, self.depth - 1, -math.inf, math.inf, 1)
            work.undo()
            scores[col] = _score(value)
        return scores

    def _search(self, board: Connect4Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        if board.won:
            return -(_WIN - ply)  # the previous move connected four
        moves = [c for c in _ORDER if board.heights[c] < ROWS]
        if not moves:
            return 0
        if depth == 0:
            return evaluate(board)
        best = -math.inf
        for col in moves:
            board.play(col)
            value = -self._search(board, depth - 1, -beta, -alpha, ply + 1)
            board.undo()
            best = max(best, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best


def evaluate(board: Connect4Board) -> int:
    """Heuristic score for the side to move: open lines weighted by how full they are."""
    me, opponent = board.turn, 3 - board.turn
    cells = board.cells
    score = 0
    for window in WINDOWS:
        mine = theirs = 0
        for index in window:
            if cells[index] == me:
                mine += 1
            elif cells[index] == opponent:
                theirs += 1
        if mine and not theirs:
            score += (0, 1, 5, 40)[mine]
        elif theirs and not mine:
            score -= (0, 1, 5, 40)[theirs]
    for row in range(ROWS):
        value = cells[row * COLS + CENTER]
        score += 3 if value == me else -3 if value == opponent else 0
    return score


def _score(value: float) -> Score:
    """Search value (for the mover) as winning chances and a label."""
    if abs(value) >= _WIN - 100:
        plies = int(_WIN - abs(value))
        if value > 0:
            return Score(1 - plies / 1000, f"wins in {(plies + 1) // 2}")
        return Score(-1 + plies / 1000, f"loses in {plies // 2}")
    return Score(math.tanh(value / 100), f"{int(value):+d}")
