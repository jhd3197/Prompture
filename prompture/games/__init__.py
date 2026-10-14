"""Game harnesses that let language models play rules-bound games without illegal moves.

* :mod:`.base` — :class:`GameHarness` (menu of legal moves, enum-constrained
  replies, retries and rescues, the ``level`` and ``concepts`` dials) and the
  :class:`Game` / :class:`Engine` interfaces a game implements.
* :mod:`.records` / :mod:`.trace` — turn records, failure kinds, honest usage
  accounting, configuration fingerprints and trace sinks (JSONL, in-memory).
* :mod:`.control` — turn deadlines, cancellation, transient retry policy,
  shared budgets and bounded progress events.
* :mod:`.evaluation` / :mod:`.fixtures` — fixtures, deterministic hint checks,
  seeded evaluation runs, scorecards and preregistered paired comparisons.
* :mod:`.memory` — bounded memory of a side's own accepted decisions, tied to
  the game line, with validated checkpoints.
* :mod:`.match` — :func:`play_match` between models, engines and random movers.
* :mod:`.strategies` — opt-in experiments (one targeted reconsideration,
  proposals plus a constrained selector); none is known to improve play.
* :mod:`.tournament` — side-balanced schedules, resumable batches with bounded
  concurrency and shared budgets, and exploratory opponent calibration.
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
    RenderedImage,
    Score,
    game_line,
)
from .connect4 import Connect4, Connect4Board, Connect4Engine
from .control import CancelToken, EventBuffer, GameBudget, RetryPolicy, TurnEvent
from .evaluation import (
    EvaluationRun,
    Fixture,
    Preregistration,
    check_hints,
    compare,
    evaluate,
    scorecard,
    summary_text,
)
from .match import EnginePlayer, MatchResult, Player, RandomPlayer, play_match
from .memory import CheckpointError, DecisionMemory, MemoryEntry
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
from .strategies import Proposals, Reconsider, experiment_status
from .tournament import (
    BatchReport,
    BatchStore,
    Entrant,
    calibrate,
    double_round_robin,
    pair_schedule,
    run_batch,
    standings,
)
from .trace import JsonlTraceSink, MemoryTraceSink, TraceSink, TraceWriteError, read_jsonl_trace

__all__ = [
    "LEVELS",
    "Accounting",
    "Attempt",
    "BatchReport",
    "BatchStore",
    "BudgetExhausted",
    "CancelToken",
    "Candidate",
    "CheckpointError",
    "Connect4",
    "Connect4Board",
    "Connect4Engine",
    "DecisionMemory",
    "Engine",
    "EnginePlayer",
    "Entrant",
    "EvaluationRun",
    "EventBuffer",
    "FailureInfo",
    "FailureKind",
    "Fingerprint",
    "Fixture",
    "Game",
    "GameBudget",
    "GameHarness",
    "GameHistory",
    "GameLine",
    "JsonlTraceSink",
    "MatchResult",
    "MemoryEntry",
    "MemoryTraceSink",
    "Motif",
    "MoveDecision",
    "Othello",
    "OthelloBoard",
    "OthelloEngine",
    "Player",
    "Preregistration",
    "Proposals",
    "RandomPlayer",
    "Reconsider",
    "RenderedImage",
    "RetryPolicy",
    "Score",
    "TraceSink",
    "TraceWriteError",
    "TurnCancelled",
    "TurnError",
    "TurnEvent",
    "TurnFailed",
    "TurnRecord",
    "calibrate",
    "check_hints",
    "classify_failure",
    "compare",
    "double_round_robin",
    "evaluate",
    "experiment_status",
    "fingerprint",
    "game_line",
    "pair_schedule",
    "play_match",
    "read_jsonl_trace",
    "run_batch",
    "scorecard",
    "standings",
    "summary_text",
]
