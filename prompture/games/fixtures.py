"""Built-in Connect Four and Othello fixtures: the shared evaluator is not chess-only.

Positions are move sequences replayed through each game's own rules, so an
illegal history fails loudly. They need no optional dependency.
"""

from __future__ import annotations

from typing import Any

from .evaluation import Fixture

HAND = "hand-built for the Prompture harness"

CONNECT4_FIXTURES: list[dict[str, Any]] = [
    {
        "id": "c4-win-now",
        "split": "dev",
        "provenance": HAND,
        "moves": list("445566"),
        "acceptable": ["3", "7"],
        "expect": [{"move": "7", "present": ["win"]}],
    },
    {
        "id": "c4-double-threat",
        "split": "dev",
        "provenance": HAND,
        "moves": list("4455"),
        "acceptable": ["3", "6"],
        "expect": [{"move": "3", "present": ["double-threat"], "absent": ["threat"]}],
    },
    {
        "id": "c4-block-but-lost",
        "split": "validation",
        "provenance": HAND,
        "moves": list("4455661"),
        "expect": [{"move": "7", "present": ["block: blocks Red's four in column 7", "safety"]}],
        "notes": "Red has two winning cells; blocking one still loses",
    },
    {
        "id": "c4-filling-under-a-win",
        "split": "dev",
        "provenance": HAND,
        "moves": list("4643663"),
        "expect": [
            {"move": "5", "concepts": ["safety"], "present": ["safety: warning: lets Red win next move in column 5"]},
            {"move": "1", "concepts": ["safety"], "absent": ["safety"]},
        ],
    },
    {
        "id": "c4-quiet-opening",
        "split": "test",
        "provenance": HAND,
        "moves": [],
        "expect": [{"move": "1", "absent": ["safety", "threat", "win", "block"]}],
    },
]

OTHELLO_FIXTURES: list[dict[str, Any]] = [
    {
        "id": "othello-gives-corner",
        "split": "dev",
        "provenance": HAND,
        "moves": ["d3", "e3", "f4", "g5", "e2", "e1", "f3", "c5", "d1", "f2"],
        "acceptable": ["f1", "g3", "c4", "g4", "c6", "d6", "e6"],
        "expect": [
            {"move": "g2", "present": ["safety: warning: gives White the corner h1", "x-square"]},
            {"move": "f1", "absent": ["safety", "x-square"]},
        ],
    },
    {
        "id": "othello-opening-mobility",
        "split": "validation",
        "provenance": HAND,
        "moves": [],
        "acceptable": ["c4", "d3", "e6", "f5"],
        "expect": [{"move": "d3", "present": ["mobility: leaves White 3 moves"]}],
    },
]


def builtin_fixtures(game_title: str) -> list[Fixture]:
    """The built-in fixtures for ``"Connect Four"`` or ``"Othello"``."""
    table = {"Connect Four": CONNECT4_FIXTURES, "Othello": OTHELLO_FIXTURES}[game_title]
    return [Fixture.from_dict({"game": game_title, "initial": None, **item}) for item in table]
