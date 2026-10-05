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
from dataclasses import dataclass, field
from typing import Any, Protocol

from .base import Engine, Game, MoveDecision


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
        return MoveDecision(move, name, scores[move].label, [], [], False, {})


class RandomPlayer:
    """Plays a uniformly random legal move: the floor any model should beat."""

    def __init__(self, game: Game, seed: int | None = None) -> None:
        self.game = game
        self._rng = random.Random(seed)  # nosec B311 - random opponent for games, not crypto

    def choose(self, state: Any) -> MoveDecision:
        move = self._rng.choice(self.game.legal_moves(state))
        return MoveDecision(move, self.game.move_name(state, move), "", [], [], False, {})


@dataclass
class MatchResult:
    """A finished (or move-capped) game."""

    winner: str | None
    moves: list[str]
    state: Any
    decisions: list[tuple[str, MoveDecision]] = field(default_factory=list)
    finished: bool = True

    @property
    def rescues(self) -> dict[str, int]:
        """Rescued (fallback) moves per player."""
        counts: dict[str, int] = {}
        for player, decision in self.decisions:
            counts[player] = counts.get(player, 0) + decision.rescued
        return counts


def play_match(
    game: Game,
    players: dict[str, Player],
    *,
    state: Any = None,
    max_moves: int = 400,
) -> MatchResult:
    """Play ``game`` from ``state`` (the start by default) with ``players`` keyed by side name."""
    state = game.new_state() if state is None else state
    moves: list[str] = []
    decisions: list[tuple[str, MoveDecision]] = []
    while not game.is_over(state):
        if len(moves) >= max_moves:
            return MatchResult(None, moves, state, decisions, finished=False)
        side = game.player(state)
        if side not in players:
            raise ValueError(f"No player for {side}; got {sorted(players)}")
        decision = players[side].choose(state)
        moves.append(decision.name)
        decisions.append((side, decision))
        state = game.apply(state, decision.move)
    return MatchResult(game.winner(state), moves, state, decisions)
