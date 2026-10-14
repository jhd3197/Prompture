"""Static exchanges, promotion warnings, witnesses and work bounds in the chess hints."""

from __future__ import annotations

import time

import pytest

chess = pytest.importorskip("chess")

from prompture.games.chess import CONCEPTS, chess_fixtures, describe_move  # noqa: E402
from prompture.games.chess import motifs as motif_module  # noqa: E402
from prompture.games.chess.exchange import best_capture, exchange  # noqa: E402
from prompture.games.chess.harness import ChessGame  # noqa: E402
from prompture.games.evaluation import fixture_state  # noqa: E402


def counted(fen: str, san: str, **kwargs):
    board = chess.Board(fen)
    return exchange(board, board.parse_san(san), **kwargs)


def tags(fen: str, san: str, concepts=CONCEPTS):
    board = chess.Board(fen)
    return describe_move(board, board.parse_san(san), concepts)


class TestExchange:
    def test_xray_recapture_counts(self):
        result = counted("4r1k1/8/8/4r3/8/8/4R3/4R1K1 w - - 0 1", "Rxe5")
        assert result.gain == 5 and result.line == ("Rxe5", "Rxe5", "Rxe5") and result.complete

    def test_losing_exchange_with_xray(self):
        assert counted("4k3/8/5p2/4n3/8/8/4R3/4R1K1 w - - 0 1", "Rxe5+").gain == -1

    def test_pinned_defender_cannot_recapture(self):
        result = counted("3k4/8/3b4/4n3/8/8/8/3RR1K1 w - - 0 1", "Rxe5")
        assert result.gain == 3 and result.line == ("Rxe5",)

    def test_side_may_stop_capturing(self):
        # Qxd5 exd5: White should not continue; the count stops at the queen trade.
        assert counted("4k3/8/4p3/3q4/8/8/8/3QK3 w - - 0 1", "Qxd5").gain == 0

    def test_promotion_values_and_en_passant(self):
        assert counted("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "axb8=Q+").gain == 5 + 8
        assert counted("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 2", "dxe6").gain == 1

    def test_promoted_piece_captured_at_once_is_not_a_win(self):
        assert counted("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a8=Q").gain == -1

    def test_bound_marks_the_count_uncertain(self):
        result = counted("4r1k1/8/8/4r3/8/8/4R3/4R1K1 w - - 0 1", "Rxe5", max_plies=1)
        assert not result.complete and "count stopped after 1 captures" in result.evidence

    def test_analysis_leaves_the_board_alone(self):
        board = chess.Board()
        for san in ("e4", "d5"):
            board.push_san(san)
        before = (board.fen(), list(board.move_stack))
        exchange(board, board.parse_san("exd5"))
        best_capture(board, chess.D5)
        assert (board.fen(), list(board.move_stack)) == before


class TestTacticalHints:
    def test_attacked_twice_defended_once_is_a_warning_with_evidence(self):
        motif = next(m for m in tags("3rk3/8/1b6/8/8/2P2N2/8/4K3 w - - 0 1", "Nd4", {"safety"}))
        assert motif.text == "warning: leaves the knight on d4 to be won by the bishop on b6"
        assert motif.severity == "warning" and not motif.exact and motif.evidence == "Bxd4 cxd4 Rxd4"

    def test_xray_exchange_is_an_opportunity_described_as_local(self):
        motif = next(m for m in tags("4r1k1/8/8/4r3/8/8/4R3/4R1K1 w - - 0 1", "Rxe5") if m.kind == "hanging-piece")
        assert "local count" in motif.text and not motif.exact and motif.evidence == "Rxe5 Rxe5 Rxe5"

    def test_allowed_promotion_and_its_negative_control(self):
        fen = "4k3/8/8/8/8/8/1p1K4/8 w - - 0 1"
        assert [m.text for m in tags(fen, "Ke2", {"allows"})] == ["allows b1=Q: promotes to a queen on b1"]
        assert tags(fen, "Kc2", {"allows"}) == []

    def test_underpromotion_is_named(self):
        assert [m.text for m in tags("8/4P1k1/8/8/8/8/8/4K3 w - - 0 1", "e8=N+", ())] == [
            "promotes to a knight (an underpromotion)",
            "gives check",
        ]

    def test_mate_threats_are_urgent(self):
        fen = "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1"
        motif = tags(fen, "Kh8", {"allows"})[0]
        assert motif.severity == "urgent" and motif.exact and motif.evidence == "Ra8#"

    def test_reply_check_is_bounded(self, monkeypatch):
        monkeypatch.setattr(motif_module, "MAX_REPLIES", 3)
        texts = [m.text for m in tags(chess.STARTING_FEN, "e4", {"allows"})]
        assert texts == ["reply check stopped after 3 of 20 replies; the rest were not examined"]


class TestWitnesses:
    def test_every_allows_and_exchange_hint_has_a_legal_witness(self):
        game = ChessGame()
        checked = 0
        for fixture in chess_fixtures():
            board = fixture_state(game, fixture)
            for move in board.legal_moves:
                for motif in describe_move(board, move, CONCEPTS):
                    if not motif.evidence:
                        continue
                    replay = board.copy()
                    if motif.kind in ("allows", "safety"):  # witnesses start after the candidate move
                        replay.push(move)
                    for san in motif.evidence.split(" ...")[0].split():
                        replay.push_san(san)  # raises if the witness is not legal
                    checked += 1
        assert checked > 100


def test_hint_cost_is_recorded_not_asserted(capsys):
    """CPU per position, for the plan's evidence log (no threshold: machines differ)."""
    game = ChessGame()
    started = time.process_time()
    count = 0
    for fixture in chess_fixtures():
        board = fixture_state(game, fixture)
        for move in board.legal_moves:
            describe_move(board, move, CONCEPTS)
            count += 1
    elapsed = time.process_time() - started
    print(f"described {count} moves in {elapsed * 1000:.0f} ms CPU")
    assert count > 0
