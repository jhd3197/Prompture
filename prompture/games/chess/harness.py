"""Chess on the shared game harness: rules from python-chess, tags from :mod:`.motifs`.

:class:`ChessHarness` is a :class:`~prompture.games.base.GameHarness` for chess
(see that module for the ``level`` and ``concepts`` dials). Engine levels
(2 and up) need a UCI engine such as Stockfish: pass a path or a running
``chess.engine.SimpleEngine``.

Example::

    from prompture.games.chess import ChessHarness
    import chess

    board = chess.Board()
    with ChessHarness("ollama/llama3.2:3b", level=1, concepts={"fork", "safety", "allows"}) as harness:
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

from ..base import BLUNDER_DROP, Candidate, Engine, GameHarness, GameLine, Motif, Score
from .motifs import CONCEPTS, HINTS_VERSION, VALUES, describe_move

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
    sides = ("White", "Black")
    concepts: tuple[str, ...] = CONCEPTS
    rules = ""
    version = "1"
    hints_version = HINTS_VERSION

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
        lines.append("Board (White uppercase, Black lowercase; . = empty):")
        lines += [f"{8 - rank} {row}" for rank, row in enumerate(str(board).splitlines())]
        lines.append("  a b c d e f g h")
        for side in (chess.WHITE, chess.BLACK):
            lines.append(f"{'White' if side else 'Black'} pieces: {_piece_list(board, side)}")
        material = {
            side: sum(len(board.pieces(piece, side)) * VALUES[piece] for piece in VALUES if piece != chess.KING)
            for side in (chess.WHITE, chess.BLACK)
        }
        balance = material[chess.WHITE] - material[chess.BLACK]
        label = "equal" if not balance else f"{'White' if balance > 0 else 'Black'} ahead by {abs(balance)}"
        lines.append(
            f"Material (P=1, N=B=3, R=5, Q=9; kings excluded): "
            f"White {material[chess.WHITE]}, Black {material[chess.BLACK]}; {label}."
        )
        if board.is_repetition(3):
            lines.append("Repetition: this position has occurred at least three times; a draw can be claimed.")
        elif board.is_repetition(2):
            lines.append("Repetition: this position has occurred twice; a third time allows a draw claim.")
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

    def history(self, board: chess.Board) -> GameLine:
        """The line from the board's root position (FEN) through its full move stack, in SAN."""
        root = board.root()
        replay = root.copy(stack=False)
        plies = []
        for move in board.move_stack:
            plies.append(("White" if replay.turn == chess.WHITE else "Black", replay.san(move)))
            replay.push(move)
        return GameLine(root.fen(), tuple(plies))

    def dump_state(self, board: chess.Board) -> dict[str, Any]:
        return {"game": self.title, "initial": board.root().fen(), "moves": [m.uci() for m in board.move_stack]}

    def load_state(self, data: dict[str, Any]) -> chess.Board:
        """Replay ``{"initial": FEN or None, "moves": [UCI or SAN]}``; illegal moves raise ``ValueError``."""
        board = chess.Board(data["initial"]) if data.get("initial") else chess.Board()
        for name in data.get("moves", []):
            move = self.parse_move(board, str(name))
            if move is None or move not in board.legal_moves:
                raise ValueError(f"illegal chess move {name!r} at ply {len(board.move_stack)} from {board.fen()}")
            board.push(move)
        return board


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

    def identity(self) -> str:
        """Engine name and search limit, for configuration fingerprints."""
        info = getattr(self.engine, "id", None)
        name = info.get("name") if isinstance(info, dict) else type(self.engine).__name__
        limit = {k: v for k, v in vars(self.limit).items() if v is not None} if self.limit else {}
        return f"uci:{name}:{sorted(limit.items())}"

    def close(self) -> None:
        """Quit the engine process if this object started it."""
        if self._owned and self.engine is not None:
            self.engine.quit()
            self.engine = None


class ChessHarness(GameHarness):
    """A :class:`~prompture.games.base.GameHarness` for chess.

    Prompts include a board diagram and material totals. At level 1 and above,
    the move menu also identifies repetitions using the board's move stack.
    ``memory="previous"`` (the default) shows the last accepted reason/plan on
    that side's next turn on the same line; ``"none"`` or a window size (``4``,
    ``8``) change that (see :mod:`prompture.games.memory`). Call :meth:`reset`
    when starting a new game. Keep the board's history (rather than rebuilding
    from FEN) for repetition and move memory. Calls stream by default where
    supported; pass ``stream=False`` to opt out.
    """

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
        stream: bool = True,
        memory: str | int | None = "previous",
        **harness_options: Any,
    ) -> None:
        """``harness_options`` are passed to :class:`~prompture.games.base.GameHarness`
        (``failure_policy``, ``trace``, ``deadline``, ``cancel``, ``retry``, ``budget``,
        ``on_event`` ...)."""
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
            stream=stream,
            memory=memory,
            **harness_options,
        )

    def reset(self) -> None:
        """Forget both sides' previous reasons/plans before starting another game."""
        self.memory.clear()

    def candidates(self, board: chess.Board) -> list[Candidate]:
        candidates = super().candidates(board)
        if self.level < 1:
            return candidates
        for candidate in candidates:
            after = self.game.apply(board, candidate.move)
            if after.is_repetition(3):
                candidate.motifs.append(Motif("repetition", "repeats the position a third time: a draw can be claimed"))
            elif after.is_repetition(2):
                candidate.motifs.append(Motif("repetition", "repeats the position: a third time allows a draw claim"))
        return candidates


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
