"""Chess evaluation fixtures: positive and negative hint checks plus scorable decisions.

Every position and move is validated through python-chess when loaded (see
:func:`chess_fixtures`). Fixtures cover mates, mate threats and quiet
defenses, forks (including false forks), exchanges (even, losing, x-ray and
pinned defenders), promotions (lost, check, allowed, underpromotion), en
passant, history-aware repetition and misleading sacrifices. The three Arena
mistakes are kept as regressions; they are positions where a warning must
appear, not unique-best-move puzzles.

``gap`` marks an expectation the current detectors are known to miss. The
hint report lists those apart from regressions, and reports a gap that starts
passing so its marker can be removed.

The ``checkmate`` evaluator accepts any mating move (as Lichess does on a
puzzle's last move).
"""

from __future__ import annotations

from typing import Any

import chess

from ..evaluation import Fixture, HintReport, check_hints, register_evaluator

ARENA = "Arena game (Prompture consumer), saved position"
HAND = "hand-built for the Prompture harness"


def _checkmate(game: Any, board: chess.Board, move: chess.Move) -> bool:
    after = board.copy(stack=False)
    after.push(move)
    return after.is_checkmate()


register_evaluator("chess", "checkmate", _checkmate)

CHESS_FIXTURES: list[dict[str, Any]] = [
    # -- mates and mate threats ---------------------------------------------------
    {
        "id": "mate-back-rank",
        "split": "dev",
        "provenance": HAND,
        "initial": "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
        "evaluator": "checkmate",
        "expect": [{"move": "Ra8#", "present": ["mate", "back-rank-mate"]}],
        "tags": ["mate"],
    },
    {
        "id": "mate-not-with-luft",
        "split": "dev",
        "provenance": HAND,
        "initial": "6k1/5pp1/7p/8/8/8/8/R5K1 w - - 0 1",
        "expect": [{"move": "Ra8+", "present": ["check"], "absent": ["mate", "back-rank-mate"]}],
        "tags": ["mate", "negative"],
    },
    {
        "id": "allows-mate-in-one",
        "split": "validation",
        "provenance": "Arena game Fable vs Astra, 24.h4?? allowed ...Rc1#",
        "initial": "2r5/5pp1/p5k1/1p1pPbN1/1n6/8/PPr3PP/1K1R2R1 w - - 6 24",
        "acceptable": ["Ka1", "Ne4", "Rd3", "b3"],
        "expect": [
            {"move": "h4", "concepts": ["allows"], "present": ["allows: allows Rc1#: checkmate"]},
            {"move": "Ka1", "concepts": ["allows"], "absent": ["allows: allows Rc1#: checkmate"]},
        ],
        "tags": ["mate", "threat"],
        "notes": "acceptable = every move that leaves no mate in one (python-chess)",
    },
    {
        "id": "quiet-defense-luft",
        "split": "dev",
        "provenance": HAND,
        "initial": "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",
        "acceptable": ["Kf8", "f5", "f6", "g5", "g6", "h5", "h6"],
        "expect": [
            {"move": "Kh8", "concepts": ["allows"], "present": ["allows: allows Ra8#: checkmate"]},
            {"move": "h6", "concepts": ["allows"], "absent": ["allows"]},
        ],
        "tags": ["quiet-defense", "mate"],
    },
    {
        "id": "mate-capturing-a-pinned-queen",
        "split": "test",
        "provenance": HAND,
        "initial": "7k/8/8/8/8/8/6PP/4rQ1K b - - 0 1",
        "evaluator": "checkmate",
        "expect": [{"move": "Rxf1#", "present": ["mate", "pin: pin: captures the pinned queen on f1"]}],
        "tags": ["mate", "pin"],
    },
    # -- forks ------------------------------------------------------------------------
    {
        "id": "fork-knight-king-queen",
        "split": "dev",
        "provenance": HAND,
        "initial": "2q3k1/8/8/3N4/8/8/8/6K1 w - - 0 1",
        "acceptable": ["Ne7+"],
        "expect": [
            {"move": "Ne7+", "present": ["fork: fork: attacks the king on g8 and the queen on c8"]},
            {"move": "Nb6", "absent": ["fork: fork: attacks the king on g8 and the queen on c8"]},
        ],
        "tags": ["fork"],
    },
    {
        "id": "fork-met-by-even-trade",
        "split": "validation",
        "provenance": "Arena game, 19.Qh4+ was answered by ...Qxh4",
        "initial": "2r2r2/3bqppk/p7/1p1pP3/1n3Q2/5N2/PPP3PP/1K1R3R w - - 0 19",
        "expect": [{"move": "Qh4+", "absent": ["fork"]}],
        "tags": ["fork", "negative"],
    },
    {
        "id": "fork-by-a-lost-piece",
        "split": "dev",
        "provenance": HAND,
        "initial": "3qk2r/8/8/6N1/8/8/8/4K3 w - - 0 1",
        "expect": [{"move": "Nf7", "present": ["safety"], "absent": ["fork"]}],
        "tags": ["fork", "negative"],
    },
    {
        "id": "regression-20-Nd7-allows-d6-fork",
        "split": "dev",
        "provenance": ARENA + ", 20...Nd7",
        "initial": "r1b2rk1/2q1bppp/5n2/p1pPp3/PpN1P3/5N1P/1PB2PP1/R2QR1K1 b - - 1 20",
        "expect": [
            {"move": "Nd7", "present": ["allows: allows d6: fork: attacks the queen on c7 and the bishop on e7"]}
        ],
        "tags": ["fork", "regression"],
    },
    # -- exchanges ---------------------------------------------------------------------
    {
        "id": "regression-21-Qb6-loses-queen",
        "split": "dev",
        "provenance": ARENA + ", 21...Qb6",
        "initial": "r1b2rk1/2qnbppp/3P4/p1p1p3/PpN1P3/5N1P/1PB2PP1/R2QR1K1 b - - 0 21",
        "expect": [
            {
                "move": "Qb6",
                "present": [
                    "safety: warning: leaves the queen on b6 to be won by the knight on c4",
                    "allows: allows Nxb6: winning capture of the queen on b6",
                ],
            }
        ],
        "tags": ["exchange", "regression"],
    },
    {
        "id": "regression-34-c5-loses-pawn",
        "split": "dev",
        "provenance": ARENA + ", 34.c5",
        "initial": "7k/6p1/7p/6r1/PrP5/3R1P2/5P1P/5R1K w - - 1 34",
        "expect": [
            {
                "move": "c5",
                "present": [
                    "safety: warning: leaves the pawn on c5 to be won by the rook on g5",
                    "allows: allows Rxc5: winning capture of the pawn on c5",
                ],
            }
        ],
        "tags": ["exchange", "regression"],
    },
    {
        "id": "exchange-wins-defended-rook",
        "split": "dev",
        "provenance": HAND,
        "initial": "4r1k1/8/8/4r3/3P4/8/8/6K1 w - - 0 1",
        "acceptable": ["dxe5"],
        "expect": [{"move": "dxe5", "present": ["hanging-piece: wins material: the rook on e5 for a pawn"]}],
        "tags": ["exchange"],
    },
    {
        "id": "exchange-even-queen-trade",
        "split": "dev",
        "provenance": HAND,
        "initial": "4k3/8/4p3/3q4/8/8/8/3QK3 w - - 0 1",
        "expect": [{"move": "Qxd5", "absent": ["safety", "allows", "hanging-piece"]}],
        "tags": ["exchange", "negative"],
    },
    {
        "id": "exchange-rook-for-defended-knight",
        "split": "dev",
        "provenance": HAND,
        "initial": "4k3/8/5p2/4n3/8/8/4R3/4R1K1 w - - 0 1",
        "expect": [
            {
                "move": "Rxe5+",
                "present": ["safety: warning: leaves the rook on e5 to be won by the pawn on f6"],
                "absent": ["hanging-piece"],
            }
        ],
        "tags": ["exchange", "negative"],
    },
    {
        "id": "exchange-xray-doubled-rooks",
        "split": "validation",
        "provenance": HAND,
        "initial": "4r1k1/8/8/4r3/8/8/4R3/4R1K1 w - - 0 1",
        "acceptable": ["Rxe5"],
        "expect": [{"move": "Rxe5", "concepts": ["hanging-piece"], "present": ["hanging-piece"]}],
        "tags": ["exchange", "x-ray"],
        "notes": "Rxe5 Rxe5 Rxe5 nets a rook: the second rook recaptures through the first",
    },
    {
        "id": "exchange-attacked-twice-defended-once",
        "split": "dev",
        "provenance": HAND,
        "initial": "3rk3/8/1b6/8/8/2P2N2/8/4K3 w - - 0 1",
        "expect": [
            {
                "move": "Nd4",
                "concepts": ["safety"],
                "present": ["safety: warning: leaves the knight on d4 to be won by the bishop on b6"],
            },
            {"move": "Ne5", "concepts": ["safety"], "absent": ["safety"]},
        ],
        "tags": ["exchange", "x-ray"],
        "notes": "Bxd4 cxd4 Rxd4 nets a pawn; a one-recapture count missed it",
    },
    {
        "id": "pinned-defender",
        "split": "dev",
        "provenance": HAND,
        "initial": "3k4/8/3b4/4n3/8/8/8/3RR1K1 w - - 0 1",
        "acceptable": ["Rxe5"],
        "expect": [{"move": "Rxe5", "present": ["hanging-piece: wins the knight on e5, which was undefended"]}],
        "tags": ["exchange", "pin"],
        "notes": "the bishop on d6 is pinned to the king by the rook on d1, so it cannot recapture",
    },
    {
        "id": "hanging-knight",
        "split": "dev",
        "provenance": HAND,
        "initial": "4k3/8/8/3n4/8/8/8/3RK3 w - - 0 1",
        "acceptable": ["Rxd5"],
        "expect": [
            {"move": "Rxd5", "present": ["hanging-piece: wins the knight on d5, which was undefended"]},
            {"move": "Rd2", "absent": ["hanging-piece"]},
        ],
        "tags": ["exchange"],
    },
    # -- promotions ------------------------------------------------------------------------
    {
        "id": "promotion-with-check",
        "split": "dev",
        "provenance": HAND,
        "initial": "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        "acceptable": ["b8=Q+", "b8=R+"],
        "expect": [{"move": "b8=Q+", "present": ["promotion: promotes to a queen", "check"]}],
        "tags": ["promotion"],
    },
    {
        "id": "promotion-into-capture",
        "split": "validation",
        "provenance": HAND,
        "initial": "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "acceptable": ["axb8=Q+", "axb8=R+"],
        "expect": [
            {"move": "a8=Q", "present": ["safety: warning: leaves the queen on a8 to be won by the rook on b8"]},
            {"move": "axb8=Q+", "present": ["hanging-piece"], "absent": ["safety"]},
        ],
        "tags": ["promotion", "exchange"],
    },
    {
        "id": "allows-promotion",
        "split": "validation",
        "provenance": HAND,
        "initial": "4k3/8/8/8/8/8/1p1K4/8 w - - 0 1",
        "acceptable": ["Kc2"],
        "expect": [
            {"move": "Ke2", "concepts": ["allows"], "present": ["allows"]},
            {"move": "Kc2", "concepts": ["allows"], "absent": ["allows"]},
        ],
        "tags": ["promotion", "negative"],
        "notes": "only Kc2 meets b1=Q (with Kxb1); every other move lets the pawn promote safely",
    },
    {
        "id": "underpromotion-knight-check",
        "split": "test",
        "provenance": HAND,
        "initial": "8/4P1k1/8/8/8/8/8/4K3 w - - 0 1",
        "expect": [
            {"move": "e8=Q", "present": ["promotion: promotes to a queen"]},
            {
                "move": "e8=N+",
                "present": ["promotion: promotes to a knight (an underpromotion)"],
            },
        ],
        "tags": ["promotion"],
    },
    # -- en passant ------------------------------------------------------------------------
    {
        "id": "allows-en-passant",
        "split": "dev",
        "provenance": HAND,
        "initial": "4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1",
        "expect": [
            {
                "move": "e4",
                "concepts": ["allows"],
                "present": ["allows: allows dxe3: winning capture of the pawn on e4"],
            },
            {"move": "e3", "present": ["safety: warning: leaves the pawn on e3 to be won by the pawn on d4"]},
        ],
        "tags": ["en-passant"],
    },
    {
        "id": "en-passant-capture",
        "split": "dev",
        "provenance": HAND,
        "initial": "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 2",
        "expect": [{"move": "dxe6", "present": ["capture: captures a pawn en passant"]}],
        "tags": ["en-passant"],
    },
    # -- repetition (history-aware) --------------------------------------------------------
    {
        "id": "repetition-second",
        "split": "validation",
        "provenance": HAND,
        "initial": None,
        "moves": ["Nf3", "Nf6", "Ng1"],
        "expect": [{"move": "Ng8", "present": ["repetition: repeats the position: a third time allows a draw claim"]}],
        "tags": ["repetition"],
    },
    {
        "id": "repetition-third",
        "split": "dev",
        "provenance": HAND,
        "initial": None,
        "moves": ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"],
        "expect": [
            {"move": "Ng8", "present": ["repetition: repeats the position a third time: a draw can be claimed"]}
        ],
        "tags": ["repetition"],
    },
    {
        "id": "repetition-not-from-fen-alone",
        "split": "dev",
        "provenance": HAND,
        "initial": "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
        "moves": ["Nf6", "Ng1"],
        "expect": [{"move": "Ng8", "absent": ["repetition"]}],
        "tags": ["repetition", "negative"],
        "notes": "the start position occurred before this FEN; without that history it is not a repetition",
    },
    # -- misleading sacrifices and grabs ------------------------------------------------------
    {
        "id": "misleading-bishop-sacrifice",
        "split": "dev",
        "provenance": HAND,
        "initial": "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
        "expect": [
            {
                "move": "Bxf7+",
                "present": ["safety: warning: leaves the bishop on f7 to be won by the king on e8"],
                "absent": ["hanging-piece"],
            }
        ],
        "tags": ["sacrifice", "negative"],
    },
    {
        "id": "misleading-queen-grab",
        "split": "validation",
        "provenance": HAND,
        "initial": "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1",
        "expect": [
            {
                "move": "Qxd5",
                "present": ["safety: warning: leaves the queen on d5 to be won by the pawn on e6"],
                "absent": ["hanging-piece"],
            }
        ],
        "tags": ["sacrifice", "negative"],
    },
    {
        "id": "pawn-grab-loses-knight",
        "split": "test",
        "provenance": HAND,
        "initial": "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "expect": [
            {"move": "Nxe5", "present": ["safety: warning: leaves the knight on e5 to be won by the knight on c6"]}
        ],
        "tags": ["sacrifice", "negative"],
    },
    # -- other motifs -------------------------------------------------------------------------
    {
        "id": "discovered-attack-and-fork",
        "split": "dev",
        "provenance": HAND,
        "initial": "4q1k1/8/8/8/4N3/8/8/4R1K1 w - - 0 1",
        "acceptable": ["Nf6+"],
        "expect": [
            {
                "move": "Nf6+",
                "present": ["discovered-attack: discovered attack: the rook on e1 now attacks the queen on e8", "fork"],
            },
            {"move": "Nc3", "present": ["safety: warning: leaves the rook on e1 to be won by the queen on e8"]},
        ],
        "tags": ["discovered-attack", "fork"],
    },
    {
        "id": "absolute-pin",
        "split": "dev",
        "provenance": HAND,
        "initial": "4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1",
        "expect": [
            {"move": "Bb5", "present": ["pin: pin: pins the knight on c6 to the king"]},
            {"move": "Bc4", "absent": ["pin"]},
        ],
        "tags": ["pin"],
    },
    {
        "id": "skewer-through-king",
        "split": "validation",
        "provenance": HAND,
        "initial": "8/8/8/3k3r/8/8/8/R5K1 w - - 0 1",
        "acceptable": ["Ra5+"],
        "expect": [
            {"move": "Ra5+", "present": ["skewer: skewer: attacks the king on d5, with the rook on h5 behind it"]}
        ],
        "tags": ["skewer"],
    },
]


def chess_fixtures(*, splits: tuple[str, ...] | None = None) -> list[Fixture]:
    """The built-in chess fixtures, every position and named move validated by python-chess."""
    fixtures = []
    for data in CHESS_FIXTURES:
        fixture = Fixture.from_dict({"game": "chess", **data})
        board = chess.Board(fixture.initial) if fixture.initial else chess.Board()
        for name in fixture.moves:
            board.push_san(name)
        for name in fixture.acceptable:
            board.parse_san(name)
        for expectation in fixture.expect:
            board.parse_san(expectation["move"])
        if splits is None or fixture.split in splits:
            fixtures.append(fixture)
    return fixtures


def chess_hint_report(fixtures: list[Fixture] | None = None) -> HintReport:
    """Check chess hint expectations with :class:`ChessHarness` candidates (history-aware). No model calls."""
    from ..evaluation import _NoCalls
    from .harness import ChessGame, ChessHarness

    def make(concepts: Any) -> ChessHarness:
        return ChessHarness(driver=_NoCalls(), level=1, concepts=concepts)

    return check_hints(ChessGame(), chess_fixtures() if fixtures is None else fixtures, make)
