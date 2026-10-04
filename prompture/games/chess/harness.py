"""ChessHarness: let any model play chess without ever making an illegal move.

The model never writes a move freely. Each turn it is shown a menu of
candidate moves and must answer with one of them; the answer's JSON schema
lists the candidates as an ``enum``, so drivers with native structured output
(Ollama, llama.cpp, OpenAI, Claude ...) cannot produce anything else. Every
reply is still checked with python-chess. A rejected reply is sent back with
the reason; after ``max_retries`` the harness plays a fallback move and marks
the decision ``rescued``, so a model's rescue count measures how much it leans
on the harness.

Two independent dials shape the help a model gets:

``level`` — which moves it is shown (the safety rail):

    0  every legal move, untagged
    1  every legal move, tagged with facts (captures, checks, promotions)
    2  legal moves minus blunders (needs an engine)
    3  the engine's top moves, unranked and without scores (top 5 by default)
    4  the engine's top moves, ranked, with evaluations (top 3 by default)
    5  only the engine's best move

``concepts`` — which ideas it knows (see :mod:`.motifs`): ``"fork"``,
``"pin"``, ``"safety"`` ... Each enabled concept tags the candidate moves
that show it, so a model "knows" forks when forking moves say so.

At level 3 and above most of the strength is the engine's. To model a player
of a given strength, keep ``level`` low (0-2) and vary ``concepts``.

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
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine

from ...exceptions import ExtractionError
from .motifs import CONCEPTS, Motif, describe_move

LEVELS = {
    0: "every legal move",
    1: "every legal move, tagged with facts",
    2: "legal moves minus blunders",
    3: "the engine's top moves, unranked",
    4: "the engine's top moves, ranked with evaluations",
    5: "only the engine's best move",
}
DEFAULT_TOP_N = {3: 5, 4: 3, 5: 1}
BLUNDER_DROP = 0.3  # drop in winning chances (-1..1) that Lichess calls a blunder
_PIECE_ORDER = (chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
Scores = dict[chess.Move, tuple[int | None, int | None]]  # move -> (centipawns, mate) for the mover


def winning_chances(cp: int | None, mate: int | None) -> float:
    """Lichess's -1..1 winning chances for a centipawn score or a mate count."""
    if mate is not None:
        cp = 1000 if mate > 0 else -1000
    cp = max(-1000, min(1000, cp or 0))
    return 2 / (1 + math.exp(-0.00368208 * cp)) - 1


@dataclass
class Candidate:
    """A move offered to the model, with its tags and (if analysed) its score for the mover."""

    move: chess.Move
    san: str
    motifs: list[Motif] = field(default_factory=list)
    cp: int | None = None
    mate: int | None = None

    @property
    def scored(self) -> bool:
        return self.cp is not None or self.mate is not None

    def evaluation(self) -> str:
        if self.mate is not None:
            return f"mate in {self.mate}" if self.mate > 0 else f"gets mated in {-self.mate}"
        return f"{(self.cp or 0) / 100:+.2f}"


@dataclass
class Attempt:
    """One model reply and, if it was rejected, why."""

    reply: Any
    error: str | None = None


@dataclass
class MoveDecision:
    """The move the harness settled on and how it got there."""

    move: chess.Move
    san: str
    reason: str
    candidates: list[Candidate]
    attempts: list[Attempt]
    rescued: bool
    usage: dict[str, float]


class ChessHarness:
    """Chooses moves with a model, constrained to a menu of candidates."""

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
        if level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}, got {level!r}")
        if level >= 2 and engine is None:
            raise ValueError(f"level {level} ({LEVELS[level]}) needs an engine, e.g. engine='stockfish'")
        self.concepts = frozenset(CONCEPTS if concepts == "all" else concepts)
        unknown = self.concepts - set(CONCEPTS)
        if unknown:
            raise ValueError(f"Unknown chess concepts: {sorted(unknown)}. Known: {list(CONCEPTS)}")
        if driver is None:
            if model is None:
                raise ValueError("Pass a model string like 'ollama/llama3.2:3b' or a driver")
            from ...drivers import get_driver_for_model

            driver = get_driver_for_model(model)
        self.driver = driver
        self.model: str = model or str(getattr(driver, "model", ""))
        self.level = level
        self.top_n = top_n or DEFAULT_TOP_N.get(level)
        self.blunder_drop = blunder_drop
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.options = options or {}
        self.engine_limit = engine_limit or chess.engine.Limit(depth=12)
        self._rng = random.Random(seed)
        self._owns_engine = isinstance(engine, (str, Path))
        self.engine: chess.engine.SimpleEngine | None = (
            chess.engine.SimpleEngine.popen_uci(str(engine)) if isinstance(engine, (str, Path)) else engine
        )

    def __enter__(self) -> ChessHarness:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Quit the engine if the harness started it."""
        if self._owns_engine and self.engine is not None:
            self.engine.quit()
            self.engine = None

    # -- candidates ---------------------------------------------------------

    def candidates(self, board: chess.Board) -> list[Candidate]:
        """The moves this level offers in ``board``, tagged with facts and concepts."""
        moves = list(board.legal_moves)
        scores: Scores = {}
        if self.level >= 2:
            scores = self._analyse(board, len(moves) if self.level == 2 else self.top_n or 1)
            moves = self._filter(moves, scores)
        offered = []
        for move in moves:
            cp, mate = scores.get(move, (None, None))
            motifs = describe_move(board, move, self.concepts, facts=self.level >= 1)
            offered.append(Candidate(move, board.san(move), motifs, cp, mate))
        if self.level == 4 or self.level == 5:
            offered.sort(key=lambda c: -winning_chances(c.cp, c.mate))
        else:
            offered.sort(key=lambda c: c.san)  # no order hint below level 4
        return offered

    def _analyse(self, board: chess.Board, count: int) -> Scores:
        assert self.engine is not None  # levels >= 2 require one at construction
        infos = self.engine.analyse(board, self.engine_limit, multipv=max(1, count))
        scores: Scores = {}
        for info in infos:
            if not info.get("pv"):
                continue
            score = info["score"].pov(board.turn)
            scores[info["pv"][0]] = (score.score(), score.mate())
        return scores

    def _filter(self, moves: list[chess.Move], scores: Scores) -> list[chess.Move]:
        scored = [m for m in moves if m in scores]
        if self.level >= 3:
            return scored
        best = max(winning_chances(*scores[m]) for m in scored)
        return [m for m in scored if best - winning_chances(*scores[m]) < self.blunder_drop]

    # -- prompt -------------------------------------------------------------

    def prompt(self, board: chess.Board, candidates: list[Candidate], feedback: str | None = None) -> str:
        """The turn's prompt: game so far, the position as piece lists, and the menu."""
        color = "White" if board.turn == chess.WHITE else "Black"
        lines = [f"You are playing chess as {color}."]
        history = _history(board)
        if history:
            lines.append(f"Moves so far: {history}")
        lines.append(f"Position (FEN): {board.fen()}")
        for side in (chess.WHITE, chess.BLACK):
            lines.append(f"{'White' if side else 'Black'} pieces: {_piece_list(board, side)}")
        lines.append(f"{color} to move." + (" You are in check." if board.is_check() else ""))
        lines.append("")
        lines.append("Your moves (pick exactly one, written as shown):")
        for candidate in candidates:
            notes = [m.text for m in candidate.motifs]
            if self.level >= 4 and candidate.scored:
                notes.insert(0, f"engine: {candidate.evaluation()}")
            lines.append(f"- {candidate.san}" + (f": {'; '.join(notes)}" if notes else ""))
        if feedback:
            lines.append("")
            lines.append(f"Your previous answer was rejected: {feedback}")
        lines.append("")
        lines.append("Give a short reason, then the move you choose.")
        return "\n".join(lines)

    @staticmethod
    def schema(candidates: list[Candidate]) -> dict[str, Any]:
        """JSON schema for the reply; the move must be one of the candidates' SAN."""
        return {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "move": {"type": "string", "enum": [c.san for c in candidates]},
            },
            "required": ["reason", "move"],
        }

    # -- choosing -----------------------------------------------------------

    def choose(self, board: chess.Board) -> MoveDecision:
        """Ask the model for a move in ``board`` and return a legal one, always."""
        if board.is_game_over():
            raise ValueError("The game is over; there is no move to choose.")
        candidates = self.candidates(board)
        schema = self.schema(candidates)
        attempts: list[Attempt] = []
        usage = dict.fromkeys(_USAGE_KEYS, 0.0)
        feedback = None
        for _ in range(self.max_retries + 1):
            reply, reason, error = self._ask(board, candidates, schema, feedback, usage)
            if error is None:
                chosen, error = self._match(board, candidates, reply)
                if chosen is not None:
                    attempts.append(Attempt(reply))
                    return MoveDecision(chosen.move, chosen.san, reason, candidates, attempts, False, usage)
            attempts.append(Attempt(reply, error))
            feedback = error
        fallback = self._fallback(candidates)
        return MoveDecision(fallback.move, fallback.san, "", candidates, attempts, True, usage)

    def _ask(
        self,
        board: chess.Board,
        candidates: list[Candidate],
        schema: dict[str, Any],
        feedback: str | None,
        usage: dict[str, float],
    ) -> tuple[Any, str, str | None]:
        from ...extraction.core import ask_for_json

        try:
            result = ask_for_json(
                self.driver,
                self.prompt(board, candidates, feedback),
                schema,
                ai_cleanup=False,
                model_name=self.model,
                options=self.options,
                system_prompt=self.system_prompt,
            )
        except ExtractionError as exc:
            return None, "", f"the reply was not valid JSON ({exc})"
        for key in _USAGE_KEYS:
            usage[key] += float(result.get("usage", {}).get(key) or 0)
        answer = result.get("json_object")
        if not isinstance(answer, dict):
            return answer, "", 'the reply must be a JSON object with "reason" and "move"'
        return answer.get("move"), str(answer.get("reason") or ""), None

    @staticmethod
    def _match(board: chess.Board, candidates: list[Candidate], reply: Any) -> tuple[Candidate | None, str | None]:
        if not isinstance(reply, str) or not reply.strip():
            return None, "no move was given"
        text = reply.strip()
        by_san = {c.san: c for c in candidates}
        if text in by_san:
            return by_san[text], None
        try:
            move = board.parse_san(text.rstrip("!?"))
        except ValueError:
            try:
                move = board.parse_uci(text.lower())
            except ValueError:
                return None, f"{text} is not a legal move in this position"
        for candidate in candidates:
            if candidate.move == move:
                return candidate, None
        return None, f"{text} is legal but not one of your listed moves"

    def _fallback(self, candidates: list[Candidate]) -> Candidate:
        scored = [c for c in candidates if c.scored]
        if scored:
            return max(scored, key=lambda c: winning_chances(c.cp, c.mate))
        return self._rng.choice(candidates)


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
