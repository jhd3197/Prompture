"""Chess for language models: legal-only move choice with adjustable help.

Requires python-chess: ``pip install "prompture[chess]"``.

* :mod:`.motifs` tags moves with facts and chess concepts (forks, pins ...).
* :class:`ChessHarness` asks a model for a move from a constrained menu,
  with a ``level`` dial for which moves are offered and a ``concepts`` dial
  for which ideas the model is told about.
* :mod:`.puzzles` measures a harness setting on themed puzzle packs.
* :mod:`.fixtures` holds validated evaluation positions (hint checks and
  scorable decisions) for :mod:`prompture.games.evaluation`.
"""

try:
    import chess
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError('prompture.games.chess needs python-chess: pip install "prompture[chess]"') from exc

from ..base import LEVELS, Attempt, Candidate, Motif, MoveDecision
from .fixtures import CHESS_FIXTURES, chess_fixtures, chess_hint_report
from .harness import ChessEngine, ChessGame, ChessHarness, winning_chances
from .motifs import CONCEPTS, FACTS, describe_move, losing_capture
from .puzzles import PuzzleReport, PuzzleResult, load_puzzles, play_puzzle, run_puzzles

__all__ = [
    "CHESS_FIXTURES",
    "CONCEPTS",
    "FACTS",
    "LEVELS",
    "Attempt",
    "Candidate",
    "ChessEngine",
    "ChessGame",
    "ChessHarness",
    "Motif",
    "MoveDecision",
    "PuzzleReport",
    "PuzzleResult",
    "chess_fixtures",
    "chess_hint_report",
    "describe_move",
    "load_puzzles",
    "losing_capture",
    "play_puzzle",
    "run_puzzles",
    "winning_chances",
]
