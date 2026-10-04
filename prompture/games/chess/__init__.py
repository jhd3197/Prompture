"""Chess for language models: legal-only move choice with adjustable help.

Requires python-chess: ``pip install "prompture[chess]"``.

* :mod:`.motifs` tags moves with facts and chess concepts (forks, pins ...).
* :class:`ChessHarness` asks a model for a move from a constrained menu,
  with a ``level`` dial for which moves are offered and a ``concepts`` dial
  for which ideas the model is told about.
* :mod:`.puzzles` measures a harness setting on themed puzzle packs.
"""

try:
    import chess
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError('prompture.games.chess needs python-chess: pip install "prompture[chess]"') from exc

from .harness import LEVELS, Attempt, Candidate, ChessHarness, MoveDecision, winning_chances
from .motifs import CONCEPTS, FACTS, Motif, describe_move, losing_capture
from .puzzles import PuzzleReport, PuzzleResult, load_puzzles, play_puzzle, run_puzzles

__all__ = [
    "CONCEPTS",
    "FACTS",
    "LEVELS",
    "Attempt",
    "Candidate",
    "ChessHarness",
    "Motif",
    "MoveDecision",
    "PuzzleReport",
    "PuzzleResult",
    "describe_move",
    "load_puzzles",
    "losing_capture",
    "play_puzzle",
    "run_puzzles",
    "winning_chances",
]
