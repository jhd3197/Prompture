"""Game harnesses that let language models play rules-bound games without illegal moves.

* :mod:`.base` — :class:`GameHarness` (menu of legal moves, enum-constrained
  replies, retries and rescues, the ``level`` and ``concepts`` dials) and the
  :class:`Game` / :class:`Engine` interfaces a game implements.
* :mod:`.match` — :func:`play_match` between models, engines and random movers.
* :mod:`.connect4` and :mod:`.othello` — pure-Python games with built-in engines.
* :mod:`.chess` — chess with motif detectors and puzzle runs (needs python-chess,
  ``pip install "prompture[chess]"``; imported on its own).
"""

from .base import (
    LEVELS,
    Attempt,
    Candidate,
    Engine,
    Game,
    GameHarness,
    Motif,
    MoveDecision,
    Score,
)
from .connect4 import Connect4, Connect4Board, Connect4Engine
from .match import EnginePlayer, MatchResult, Player, RandomPlayer, play_match
from .othello import Othello, OthelloBoard, OthelloEngine

__all__ = [
    "LEVELS",
    "Attempt",
    "Candidate",
    "Connect4",
    "Connect4Board",
    "Connect4Engine",
    "Engine",
    "EnginePlayer",
    "Game",
    "GameHarness",
    "MatchResult",
    "Motif",
    "MoveDecision",
    "Othello",
    "OthelloBoard",
    "OthelloEngine",
    "Player",
    "RandomPlayer",
    "Score",
    "play_match",
]
