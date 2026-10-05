"""GameHarness: let any model play a turn-based game without ever making an illegal move.

The model never writes a move freely. Each turn it is shown a menu of
candidate moves and must answer with one of them; the answer's JSON schema
lists the candidates as an ``enum``, so drivers with native structured output
(Ollama, llama.cpp, OpenAI, Claude ...) cannot produce anything else. Every
reply is still checked against the game's rules. A rejected reply is sent back
with the reason (a failed model call counts as a rejected reply); after
``max_retries`` the harness plays a fallback move and
marks the decision ``rescued``, so a model's rescue count measures how much it
leans on the harness.

Two independent dials shape the help a model gets:

``level`` — which moves it is shown (the safety rail):

    0  every legal move, untagged
    1  every legal move, tagged with facts (captures, checks, flips ...)
    2  legal moves minus blunders (needs an engine)
    3  the engine's top moves, unranked and without scores (top 5 by default)
    4  the engine's top moves, ranked, with evaluations (top 3 by default)
    5  only the engine's best move

``concepts`` — which ideas it knows: each game names its own (``"fork"`` in
chess, ``"double-threat"`` in Connect Four, ``"corner"`` in Othello). Each
enabled concept tags the candidate moves that show it, so a model "knows"
forks when forking moves say so.

At level 3 and above most of the strength is the engine's. To model a player
of a given strength, keep ``level`` low (0-2) and vary ``concepts``.

A game plugs in by implementing :class:`Game`; an engine by implementing
:class:`Engine` (``analyse(state, count)`` returning a :class:`Score` per move).
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..exceptions import ExtractionError

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
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")


@dataclass(frozen=True)
class Motif:
    """One tag on a move: its kind (a fact or concept name) and a sentence."""

    kind: str
    text: str
    squares: tuple[str, ...] = ()

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Score:
    """An engine's verdict on a move: winning chances for the mover (-1..1) and a label."""

    value: float
    label: str


class Engine(Protocol):
    """Scores moves. ``count`` is how many of the best moves are needed; more is fine."""

    def analyse(self, state: Any, count: int) -> dict[Any, Score]: ...


class Game(Protocol):
    """The rules, notation and move tags of one game. States may be mutable; ``apply`` must not mutate."""

    title: str
    concepts: tuple[str, ...]
    rules: str

    def new_state(self) -> Any: ...

    def player(self, state: Any) -> str: ...

    def legal_moves(self, state: Any) -> list[Any]: ...

    def move_name(self, state: Any, move: Any) -> str: ...

    def parse_move(self, state: Any, text: str) -> Any | None: ...

    def describe_move(self, state: Any, move: Any, concepts: Iterable[str], *, facts: bool = True) -> list[Motif]: ...

    def render(self, state: Any) -> list[str]: ...

    def apply(self, state: Any, move: Any) -> Any: ...

    def is_over(self, state: Any) -> bool: ...

    def winner(self, state: Any) -> str | None: ...

    def default_engine(self) -> Engine | None: ...


def check_concepts(game: Game, concepts: Iterable[str]) -> set[str]:
    """``concepts`` as a set, rejecting names ``game`` does not know."""
    wanted = set(concepts)
    unknown = wanted - set(game.concepts)
    if unknown:
        raise ValueError(f"Unknown {game.title} concepts: {sorted(unknown)}. Known: {list(game.concepts)}")
    return wanted


@dataclass
class Candidate:
    """A move offered to the model, with its tags and (if analysed) its score."""

    move: Any
    name: str
    motifs: list[Motif] = field(default_factory=list)
    score: Score | None = None

    @property
    def san(self) -> str:
        """The move's name; chess calls it SAN."""
        return self.name

    @property
    def scored(self) -> bool:
        return self.score is not None

    def evaluation(self) -> str:
        return self.score.label if self.score else ""


@dataclass
class Attempt:
    """One model reply and, if it was rejected, why."""

    reply: Any
    error: str | None = None


@dataclass
class MoveDecision:
    """The move a player settled on and how it got there."""

    move: Any
    name: str
    reason: str
    candidates: list[Candidate]
    attempts: list[Attempt]
    rescued: bool
    usage: dict[str, float]

    @property
    def san(self) -> str:
        return self.name


class GameHarness:
    """Chooses moves in ``game`` with a model, constrained to a menu of candidates."""

    def __init__(
        self,
        game: Game,
        model: str | None = None,
        *,
        driver: Any = None,
        level: int = 0,
        concepts: Iterable[str] | str = (),
        engine: Engine | None = None,
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
            engine = game.default_engine()
            if engine is None:
                raise ValueError(f"level {level} ({LEVELS[level]}) needs an engine")
        self.game = game
        self.concepts = frozenset(check_concepts(game, game.concepts if concepts == "all" else concepts))
        if driver is None:
            if model is None:
                raise ValueError("Pass a model string like 'ollama/llama3.2:3b' or a driver")
            from ..drivers import get_driver_for_model

            driver = get_driver_for_model(model)
        self.driver = driver
        self.model: str = model or str(getattr(driver, "model", ""))
        self.level = level
        self.engine = engine
        self.top_n = top_n or DEFAULT_TOP_N.get(level)
        self.blunder_drop = blunder_drop
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self.options = options or {}
        self._rng = random.Random(seed)

    def __enter__(self) -> GameHarness:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the engine, if it holds anything (a chess engine process)."""
        close = getattr(self.engine, "close", None)
        if close is not None:
            close()

    # -- candidates ---------------------------------------------------------

    def candidates(self, state: Any) -> list[Candidate]:
        """The moves this level offers in ``state``, tagged with facts and concepts."""
        moves = self.game.legal_moves(state)
        scores: dict[Any, Score] = {}
        if self.level >= 2:
            assert self.engine is not None  # levels >= 2 get one at construction
            scores = self.engine.analyse(state, len(moves) if self.level == 2 else self.top_n or 1)
            moves = self._filter(moves, scores)
        offered = [
            Candidate(
                move,
                self.game.move_name(state, move),
                self.game.describe_move(state, move, self.concepts, facts=self.level >= 1),
                scores.get(move),
            )
            for move in moves
        ]
        if self.level >= 4:
            offered.sort(key=lambda c: -c.score.value if c.score else 0)
        else:
            offered.sort(key=lambda c: c.name)  # no order hint below level 4
        return offered

    def _filter(self, moves: list[Any], scores: dict[Any, Score]) -> list[Any]:
        scored = [m for m in moves if m in scores]
        if self.level >= 3:
            return sorted(scored, key=lambda m: -scores[m].value)[: self.top_n]
        best = max(scores[m].value for m in scored)
        return [m for m in scored if best - scores[m].value < self.blunder_drop]

    # -- prompt -------------------------------------------------------------

    def prompt(self, state: Any, candidates: list[Candidate], feedback: str | None = None) -> str:
        """The turn's prompt: rules, the position as the game renders it, and the menu."""
        lines = [f"You are playing {self.game.title} as {self.game.player(state)}."]
        if self.game.rules:
            lines.append(self.game.rules)
        lines += self.game.render(state)
        lines.append("")
        lines.append("Your moves (pick exactly one, written as shown):")
        for candidate in candidates:
            notes = [m.text for m in candidate.motifs]
            if self.level >= 4 and candidate.score:
                notes.insert(0, f"engine: {candidate.score.label}")
            lines.append(f"- {candidate.name}" + (f": {'; '.join(notes)}" if notes else ""))
        if feedback:
            lines.append("")
            lines.append(f"Your previous answer was rejected: {feedback}")
        lines.append("")
        lines.append("Give a short reason, then the move you choose.")
        return "\n".join(lines)

    @staticmethod
    def schema(candidates: list[Candidate]) -> dict[str, Any]:
        """JSON schema for the reply; the move must be one of the candidates' names."""
        return {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "move": {"type": "string", "enum": [c.name for c in candidates]},
            },
            "required": ["reason", "move"],
        }

    # -- choosing -----------------------------------------------------------

    def choose(self, state: Any) -> MoveDecision:
        """Ask the model for a move in ``state`` and return a legal one, always."""
        if self.game.is_over(state):
            raise ValueError("The game is over; there is no move to choose.")
        candidates = self.candidates(state)
        schema = self.schema(candidates)
        attempts: list[Attempt] = []
        usage = dict.fromkeys(_USAGE_KEYS, 0.0)
        feedback = None
        for _ in range(self.max_retries + 1):
            reply, reason, error = self._ask(state, candidates, schema, feedback, usage)
            if error is None:
                chosen, error = self._match(state, candidates, reply)
                if chosen is not None:
                    attempts.append(Attempt(reply))
                    return MoveDecision(chosen.move, chosen.name, reason, candidates, attempts, False, usage)
            attempts.append(Attempt(reply, error))
            feedback = error
        fallback = self._fallback(candidates)
        return MoveDecision(fallback.move, fallback.name, "", candidates, attempts, True, usage)

    def _ask(
        self,
        state: Any,
        candidates: list[Candidate],
        schema: dict[str, Any],
        feedback: str | None,
        usage: dict[str, float],
    ) -> tuple[Any, str, str | None]:
        from ..extraction.core import ask_for_json

        try:
            result = ask_for_json(
                self.driver,
                self.prompt(state, candidates, feedback),
                schema,
                ai_cleanup=False,
                model_name=self.model,
                options=self.options,
                system_prompt=self.system_prompt,
            )
        except ExtractionError as exc:
            return None, "", f"the reply was not valid JSON ({exc})"
        except Exception as exc:  # a timeout or a dead backend must not abort a whole game
            return None, "", f"the model call failed ({exc})"
        for key in _USAGE_KEYS:
            usage[key] += float(result.get("usage", {}).get(key) or 0)
        answer = result.get("json_object")
        if not isinstance(answer, dict):
            return answer, "", 'the reply must be a JSON object with "reason" and "move"'
        return answer.get("move"), str(answer.get("reason") or ""), None

    def _match(self, state: Any, candidates: list[Candidate], reply: Any) -> tuple[Candidate | None, str | None]:
        if not isinstance(reply, (str, int)) or not str(reply).strip():
            return None, "no move was given"
        text = str(reply).strip()
        by_name = {c.name: c for c in candidates}
        if text in by_name:
            return by_name[text], None
        move = self.game.parse_move(state, text)
        if move is None:
            return None, f"{text} is not a legal move in this position"
        for candidate in candidates:
            if candidate.move == move:
                return candidate, None
        return None, f"{text} is legal but not one of your listed moves"

    def _fallback(self, candidates: list[Candidate]) -> Candidate:
        scored = [c for c in candidates if c.score]
        if scored:
            return max(scored, key=lambda c: c.score.value if c.score else 0)
        return self._rng.choice(candidates)
