"""Chess on the shared game harness: rules from python-chess, tags from :mod:`.motifs`.

:class:`ChessHarness` is a :class:`~prompture.games.base.GameHarness` for chess
(see that module for the ``level`` and ``concepts`` dials). Engine levels
(2 and up) need a UCI engine such as Stockfish: pass a path or a running
``chess.engine.SimpleEngine``.

Example::

    from prompture.games.chess import ChessHarness
    import chess

    board = chess.Board()
    with ChessHarness("ollama/llama3.2:3b", level=1, concepts={"fork", "safety"}) as harness:
        decision = harness.choose(board)
        board.push(decision.move)
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import chess
import chess.engine

from ..base import BLUNDER_DROP, Engine, GameHarness, Motif, Score
from .motifs import CONCEPTS, describe_move

_PIECE_ORDER = (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)


def winning_chances(cp: int | None, mate: int | None) -> float:
    """Lichess's -1..1 winning chances for a centipawn score or a mate count."""
    if mate is not None:
        cp = 1000 if mate > 0 else -1000
    cp = max(-1000, min(1000, cp or 0))
    return 2 / (1 + math.exp(-0.00368208 * cp)) - 1


class ChessGame:
    """Chess rules and notation (SAN) for the harness."""

    title = "chess"
    concepts: tuple[str, ...] = CONCEPTS
    rules = ""

    def new_state(self) -> chess.Board:
        return chess.Board()

    def player(self, board: chess.Board) -> str:
        return "White" if board.turn == chess.WHITE else "Black"

    def legal_moves(self, board: chess.Board) -> list[chess.Move]:
        return list(board.legal_moves)

    def move_name(self, board: chess.Board, move: chess.Move) -> str:
        return board.san(move)

    def parse_move(self, board: chess.Board, text: str) -> chess.Move | None:
        try:
            return board.parse_san(text.rstrip("!?"))
        except ValueError:
            try:
                return board.parse_uci(text.lower())
            except ValueError:
                return None

    def describe_move(
        self, board: chess.Board, move: chess.Move, concepts: Iterable[str], *, facts: bool = True
    ) -> list[Motif]:
        return describe_move(board, move, concepts, facts=facts)

    def render(self, board: chess.Board) -> list[str]:
        lines = []
        history = _history(board)
        if history:
            lines.append(f"Moves so far: {history}")
        lines.append(f"Position (FEN): {board.fen()}")
        for side in (chess.WHITE, chess.BLACK):
            lines.append(f"{'White' if side else 'Black'} pieces: {_piece_list(board, side)}")
        lines.append(f"{self.player(board)} to move." + (" You are in check." if board.is_check() else ""))
        return lines

    def apply(self, board: chess.Board, move: chess.Move) -> chess.Board:
        after = board.copy()
        after.push(move)
        return after

    def is_over(self, board: chess.Board) -> bool:
        return board.is_game_over(claim_draw=True)

    def winner(self, board: chess.Board) -> str | None:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return None
        return "White" if outcome.winner == chess.WHITE else "Black"

    def default_engine(self) -> Engine | None:
        return None


class ChessEngine:
    """A UCI engine (Stockfish ...) as a harness :class:`~prompture.games.base.Engine`."""

    def __init__(
        self,
        engine: chess.engine.SimpleEngine | str | Path,
        limit: chess.engine.Limit | None = None,
    ) -> None:
        self._owned = isinstance(engine, (str, Path))
        self.engine: Any = chess.engine.SimpleEngine.popen_uci(str(engine)) if self._owned else engine
        self.limit = limit or chess.engine.Limit(depth=12)

    def analyse(self, board: chess.Board, count: int) -> dict[Any, Score]:
        infos = self.engine.analyse(board, self.limit, multipv=max(1, count))
        scores = {}
        for info in infos:
            if not info.get("pv"):
                continue
            score = info["score"].pov(board.turn)
            cp, mate = score.score(), score.mate()
            if mate is not None:
                label = f"mate in {mate}" if mate > 0 else f"gets mated in {-mate}"
            else:
                label = f"{(cp or 0) / 100:+.2f}"
            scores[info["pv"][0]] = Score(winning_chances(cp, mate), label)
        return scores

    def close(self) -> None:
        """Quit the engine process if this object started it."""
        if self._owned and self.engine is not None:
            self.engine.quit()
            self.engine = None


class ChessHarness(GameHarness):
    """A :class:`~prompture.games.base.GameHarness` for chess."""

    def __init__(
        self,
        model: str | None = None,
        *,
        driver: Any = None,
        level: int = 0,
        concepts: Iterable[str] | str = (),
        engine: chess.engine.SimpleEngine | str | Path | None = None,
        engine_limit: chess.engine.Limit | None = None,
        top_n: int | None = None,
        blunder_drop: float = BLUNDER_DROP,
        max_retries: int = 2,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if level >= 2 and engine is None:
            raise ValueError(f"level {level} needs an engine, e.g. engine='stockfish'")
        super().__init__(
            ChessGame(),
            model,
            driver=driver,
            level=level,
            concepts=concepts,
            engine=ChessEngine(engine, engine_limit) if engine is not None else None,
            top_n=top_n,
            blunder_drop=blunder_drop,
            max_retries=max_retries,
            system_prompt=system_prompt,
            options=options,
            seed=seed,
        )


def _history(board: chess.Board, plies: int = 40) -> str:
    """The last ``plies`` moves of the game in SAN, with move numbers."""
    stack = board.move_stack[-plies:]
    replay = board.copy()
    for _ in stack:
        replay.pop()
    return replay.variation_san(stack) if stack else ""


def _piece_list(board: chess.Board, color: chess.Color) -> str:
    names = []
    for piece_type in _PIECE_ORDER:
        for square in board.pieces(piece_type, color):
            names.append(f"{chess.piece_name(piece_type)} {chess.square_name(square)}")
    return ", ".join(names)
