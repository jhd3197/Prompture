"""Full games between players: models (harnesses), engines or random movers.

Anything with ``choose(state) -> MoveDecision`` is a player, so a
:class:`~prompture.games.base.GameHarness` plays an :class:`EnginePlayer`
the same way two models play each other::

    from prompture.games import GameHarness, EnginePlayer, play_match
    from prompture.games.connect4 import Connect4

    game = Connect4()
    result = play_match(game, {
        "Red": GameHarness(game, "ollama/qwen3:8b", concepts="all"),
        "Yellow": EnginePlayer(game),
    })
    print(result.winner, result.moves, result.rescues)
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .base import Engine, Game, MoveDecision
from .control import CancelToken
from .records import TurnCancelled, TurnError, TurnFailed


class Player(Protocol):
    def choose(self, state: Any) -> MoveDecision: ...


class EnginePlayer:
    """Plays the engine's best move (the game's built-in engine unless one is given)."""

    def __init__(self, game: Game, engine: Engine | None = None) -> None:
        engine = engine or game.default_engine()
        if engine is None:
            raise ValueError(f"{game.title} has no built-in engine; pass one")
        self.game = game
        self.engine = engine

    def choose(self, state: Any) -> MoveDecision:
        scores = self.engine.analyse(state, 1)
        move = max(scores, key=lambda m: scores[m].value)
        name = self.game.move_name(state, move)
        return MoveDecision(move, name, scores[move].label, [], [], False, {}, assistance="engine")


class RandomPlayer:
    """Plays a uniformly random legal move: the floor any model should beat."""

    def __init__(self, game: Game, seed: int | None = None) -> None:
        self.game = game
        self._rng = random.Random(seed)  # nosec B311 - random opponent for games, not crypto

    def choose(self, state: Any) -> MoveDecision:
        move = self._rng.choice(self.game.legal_moves(state))
        return MoveDecision(move, self.game.move_name(state, move), "", [], [], False, {}, assistance="random")


@dataclass
class MatchResult:
    """A finished, move-capped, forfeited, stopped or cancelled game.

    ``terminal_reason`` is ``"game_over"``, ``"move_cap"``, ``"forfeit"``
    (``forfeited_by`` lost under a strict policy), ``"stopped"`` (a turn failed
    under the ``stop`` policy or a fatal error) or ``"cancelled"``. Only
    ``game_over`` and ``forfeit`` are ``finished``; a stopped or cancelled game
    has no winner and is not a draw. ``error`` keeps the turn error, whose
    ``record`` holds the failed turn's attempts.
    """

    winner: str | None
    moves: list[str]
    state: Any
    decisions: list[tuple[str, MoveDecision]] = field(default_factory=list)
    finished: bool = True
    terminal_reason: str = "game_over"
    forfeited_by: str | None = None
    error: Exception | None = None
    game_id: str | None = None

    @property
    def rescues(self) -> dict[str, int]:
        """Rescued (fallback) moves per player."""
        counts: dict[str, int] = {}
        for player, decision in self.decisions:
            counts[player] = counts.get(player, 0) + decision.rescued
        return counts

    @property
    def draw(self) -> bool:
        """A completed game with no winner (never a stopped, cancelled or capped one)."""
        return self.terminal_reason == "game_over" and self.winner is None


def play_match(
    game: Game,
    players: dict[str, Player],
    *,
    state: Any = None,
    max_moves: int = 400,
    game_id: str | None = None,
    cancel: CancelToken | None = None,
) -> MatchResult:
    """Play ``game`` from ``state`` (the start by default) with ``players`` keyed by side name.

    Players with ``start_game(game_id)`` get the game's identity first; players
    with ``commit(state, decision)`` are told when their move is applied. A
    :class:`~prompture.games.records.TurnError` ends the game without playing a
    move: a forfeit loses (when there is exactly one other side), anything else
    stops or cancels the game with no winner. ``cancel`` is checked before every
    move (pass the same token to model players to stop a turn in progress).
    """
    state = game.new_state() if state is None else state
    game_id = game_id or uuid.uuid4().hex[:12]
    for player in _unique(players.values()):
        start = getattr(player, "start_game", None)
        if callable(start):
            start(game_id)
    moves: list[str] = []
    decisions: list[tuple[str, MoveDecision]] = []
    while not game.is_over(state):
        if len(moves) >= max_moves:
            return MatchResult(None, moves, state, decisions, False, "move_cap", game_id=game_id)
        if cancel is not None and cancel.cancelled:
            stopped = TurnCancelled(f"the match was cancelled ({cancel.reason})")
            return MatchResult(None, moves, state, decisions, False, "cancelled", error=stopped, game_id=game_id)
        side = game.player(state)
        if side not in players:
            raise ValueError(f"No player for {side}; got {sorted(players)}")
        try:
            decision = players[side].choose(state)
        except TurnError as exc:
            return _ended_by(exc, side, players, moves, state, decisions, game_id)
        _commit(players[side], state, decision)
        moves.append(decision.name)
        decisions.append((side, decision))
        state = game.apply(state, decision.move)
    return MatchResult(game.winner(state), moves, state, decisions, game_id=game_id)


def _unique(players: Iterable[Player]) -> list[Player]:
    seen: list[Player] = []
    for player in players:
        if not any(player is other for other in seen):
            seen.append(player)
    return seen


def _commit(player: Player, state: Any, decision: MoveDecision) -> None:
    commit = getattr(player, "commit", None)
    if callable(commit):
        commit(state, decision)


def _ended_by(
    exc: TurnError,
    side: str,
    players: dict[str, Player],
    moves: list[str],
    state: Any,
    decisions: list[tuple[str, MoveDecision]],
    game_id: str,
) -> MatchResult:
    if isinstance(exc, TurnCancelled):
        return MatchResult(None, moves, state, decisions, False, "cancelled", error=exc, game_id=game_id)
    if isinstance(exc, TurnFailed) and exc.forfeit:
        others = [name for name in players if name != side]
        winner = others[0] if len(others) == 1 else None
        return MatchResult(winner, moves, state, decisions, True, "forfeit", side, exc, game_id)
    return MatchResult(None, moves, state, decisions, False, "stopped", error=exc, game_id=game_id)
