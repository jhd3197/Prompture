"""Game harnesses that let language models play rules-bound games without illegal moves.

* :mod:`.base` — :class:`GameHarness` (menu of legal moves, enum-constrained
  replies, retries and rescues, the ``level`` and ``concepts`` dials) and the
  :class:`Game` / :class:`Engine` interfaces a game implements.
* :mod:`.records` / :mod:`.trace` — turn records, failure kinds, honest usage
  accounting, configuration fingerprints and trace sinks (JSONL, in-memory).
* :mod:`.control` — turn deadlines, cancellation, transient retry policy,
  shared budgets and bounded progress events.
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
    GameHistory,
    GameLine,
    Motif,
    MoveDecision,
    Score,
    game_line,
)
from .connect4 import Connect4, Connect4Board, Connect4Engine
from .control import CancelToken, EventBuffer, GameBudget, RetryPolicy, TurnEvent
from .match import EnginePlayer, MatchResult, Player, RandomPlayer, play_match
from .othello import Othello, OthelloBoard, OthelloEngine
from .records import (
    Accounting,
    BudgetExhausted,
    FailureInfo,
    FailureKind,
    Fingerprint,
    TurnCancelled,
    TurnError,
    TurnFailed,
    TurnRecord,
    classify_failure,
    fingerprint,
)
from .trace import JsonlTraceSink, MemoryTraceSink, TraceSink, TraceWriteError, read_jsonl_trace

__all__ = [
    "LEVELS",
    "Accounting",
    "Attempt",
    "BudgetExhausted",
    "CancelToken",
    "Candidate",
    "Connect4",
    "Connect4Board",
    "Connect4Engine",
    "Engine",
    "EnginePlayer",
    "EventBuffer",
    "FailureInfo",
    "FailureKind",
    "Fingerprint",
    "Game",
    "GameBudget",
    "GameHarness",
    "GameHistory",
    "GameLine",
    "JsonlTraceSink",
    "MatchResult",
    "MemoryTraceSink",
    "Motif",
    "MoveDecision",
    "Othello",
    "OthelloBoard",
    "OthelloEngine",
    "Player",
    "RandomPlayer",
    "RetryPolicy",
    "Score",
    "TraceSink",
    "TraceWriteError",
    "TurnCancelled",
    "TurnError",
    "TurnEvent",
    "TurnFailed",
    "TurnRecord",
    "classify_failure",
    "fingerprint",
    "game_line",
    "play_match",
    "read_jsonl_trace",
]
