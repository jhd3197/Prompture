"""Static exchange counting on one square, with legal recaptures only.

:func:`exchange` plays a capture and then the cheapest *legal* recapture for
each side in turn (so pinned defenders cannot recapture, and pieces behind
the first attacker join in as x-rays once it has moved), letting either side
stop when continuing would lose. Promotions count the promoted piece. The
sequence is cut after ``max_plies`` captures; a cut count is marked
``complete=False`` and must be described as uncertain.

This is a local material count on one square, not a search: it ignores
checks, threats elsewhere and zwischenzugs. Hints built on it say "local
exchange", never that a line is forced.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
MAX_EXCHANGE_PLIES = 12


@dataclass(frozen=True)
class Exchange:
    """Net material for the side making the first capture, and the line counted."""

    gain: int
    line: tuple[str, ...]
    complete: bool = True

    @property
    def evidence(self) -> str:
        text = " ".join(self.line)
        return text if self.complete else f"{text} ... (count stopped after {len(self.line)} captures)"


def captured_value(board: chess.Board, move: chess.Move) -> int:
    """Material a move captures (a pawn for en passant), plus any promotion gain."""
    if board.is_en_passant(move):
        value = VALUES[chess.PAWN]
    else:
        piece = board.piece_at(move.to_square)
        value = VALUES[piece.piece_type] if piece else 0
    if move.promotion:
        value += VALUES[move.promotion] - VALUES[chess.PAWN]
    return value


def _cheapest_recapture(board: chess.Board, square: chess.Square) -> chess.Move | None:
    best: tuple[int, int, str] | None = None
    choice = None
    for move in board.legal_moves:
        if move.to_square != square or board.is_en_passant(move):
            continue
        if move.promotion not in (None, chess.QUEEN):
            continue  # a queen promotion dominates underpromotions in a material count
        piece = board.piece_at(move.from_square)
        key = (VALUES[piece.piece_type] if piece else 100, 0 if move.promotion else 1, move.uci())
        if best is None or key < best:
            best, choice = key, move
    return choice


def exchange(board: chess.Board, move: chess.Move, max_plies: int = MAX_EXCHANGE_PLIES) -> Exchange:
    """The local exchange started by ``move`` (any legal move; non-captures start with no gain).

    ``board`` is not modified.
    """
    work = board.copy(stack=False)
    square = move.to_square
    gains = [captured_value(work, move)]
    line = [work.san(move)]
    mover = work.piece_at(move.from_square)
    on_square = VALUES[move.promotion] if move.promotion else (VALUES[mover.piece_type] if mover else 0)
    work.push(move)
    complete = True
    while True:
        reply = _cheapest_recapture(work, square)
        if reply is None:
            break
        if len(line) >= max_plies:
            complete = False
            break
        gain = on_square + (VALUES[reply.promotion] - VALUES[chess.PAWN] if reply.promotion else 0)
        gains.append(gain - gains[-1])
        piece = work.piece_at(reply.from_square)
        on_square = VALUES[reply.promotion] if reply.promotion else (VALUES[piece.piece_type] if piece else 0)
        line.append(work.san(reply))
        work.push(reply)
    # Each side may decline to continue: back the speculative gains up the line.
    for index in range(len(gains) - 1, 0, -1):
        gains[index - 1] = -max(-gains[index - 1], gains[index])
    return Exchange(gains[0], tuple(line), complete)


def best_capture(board: chess.Board, square: chess.Square, max_plies: int = MAX_EXCHANGE_PLIES) -> Exchange | None:
    """The side to move's most profitable capture on ``square`` (by local exchange), if any is profitable."""
    best = None
    for move in board.legal_moves:
        if move.to_square != square or board.is_en_passant(move):
            continue
        if move.promotion not in (None, chess.QUEEN):
            continue
        result = exchange(board, move, max_plies)
        if result.gain > 0 and (best is None or result.gain > best.gain):
            best = result
    return best
