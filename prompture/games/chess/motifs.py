"""Motif detectors: plain-language tags for what a candidate move does.

``describe_move(board, move, concepts)`` returns :class:`Motif` tags such as
``fork: attacks the king on g8 and the rook on a8``. Two kinds of tag exist:

* **Facts** (:data:`FACTS`) are visible on the board with no chess knowledge:
  the move captures, gives check, promotes or castles.
* **Concepts** (:data:`CONCEPTS`) are ideas a player has to know to see:
  checkmate, back-rank mate, forks, pins, skewers, discovered attacks, winning
  hanging pieces, and the ``safety`` warning that a move leaves a piece en
  prise. Each one can be switched on or off, so a harness can model a player
  who has never heard of a skewer.

Concept names match the puzzle themes Lichess and Chess Studio use
(``fork``, ``pin``, ``discovered-attack`` ...), so puzzle packs of one theme
can measure whether a concept helps a model.

The detectors are deliberately simple static checks, not search: they look
one move deep and use attacker/defender counts. They describe a move; they
do not prove it is good. Pieces lost on the spot disqualify forks, pins and
skewers, since a "fork" by a piece that is simply captured is no fork.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import chess

from ..base import Motif

FACTS = ("check", "capture", "promotion", "castle")
CONCEPTS = (
    "mate",
    "back-rank-mate",
    "fork",
    "pin",
    "skewer",
    "discovered-attack",
    "hanging-piece",
    "safety",
)

VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
_DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_LINES = ((1, 0), (-1, 0), (0, 1), (0, -1))


def piece_label(board: chess.Board, square: chess.Square) -> str:
    """``"the knight on f3"`` for the piece on ``square``."""
    piece = board.piece_at(square)
    name = chess.piece_name(piece.piece_type) if piece else "piece"
    return f"the {name} on {chess.square_name(square)}"


def _piece(board: chess.Board, square: chess.Square) -> chess.Piece:
    """The piece on a square the caller knows is occupied."""
    piece = board.piece_at(square)
    if piece is None:
        raise ValueError(f"No piece on {chess.square_name(square)}")
    return piece


def _value(board: chess.Board, square: chess.Square) -> int:
    piece = board.piece_at(square)
    return VALUES[piece.piece_type] if piece else 0


def _defended(board: chess.Board, square: chess.Square) -> bool:
    """Whether the piece on ``square`` is protected by its own side."""
    piece = board.piece_at(square)
    return bool(piece and board.attackers(piece.color, square))


def losing_capture(board: chess.Board, square: chess.Square) -> chess.Move | None:
    """A capture of the piece on ``square`` that wins material for the side to move.

    ``board`` has the capturing side to move. A capture wins material when the
    attacker is worth less than the victim, or when the victim's side has no
    legal recapture.
    """
    victim = _value(board, square)
    for capture in board.legal_moves:
        if capture.to_square != square:
            continue
        if _value(board, capture.from_square) < victim:
            return capture
        reply = board.copy(stack=False)
        reply.push(capture)
        if not any(move.to_square == square for move in reply.legal_moves):
            return capture
    return None


def _sliding_pairs(board: chess.Board, square: chess.Square) -> Iterator[tuple[chess.Square, chess.Square]]:
    """For a slider on ``square``: (front, behind) enemy piece pairs along its lines."""
    piece = _piece(board, square)
    directions: tuple[tuple[int, int], ...] = ()
    if piece.piece_type in (chess.BISHOP, chess.QUEEN):
        directions += _DIAGONALS
    if piece.piece_type in (chess.ROOK, chess.QUEEN):
        directions += _LINES
    for file_step, rank_step in directions:
        file, rank = chess.square_file(square), chess.square_rank(square)
        found: list[chess.Square] = []
        while len(found) < 2:
            file, rank = file + file_step, rank + rank_step
            if not (0 <= file < 8 and 0 <= rank < 8):
                break
            target = chess.square(file, rank)
            if board.piece_at(target):
                found.append(target)
        if len(found) == 2 and all(board.color_at(s) == (not piece.color) for s in found):
            yield found[0], found[1]


def _facts(board: chess.Board, move: chess.Move, after: chess.Board) -> list[Motif]:
    facts = []
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "queenside"
        facts.append(Motif("castle", f"castles {side}"))
    if board.is_capture(move):
        if board.is_en_passant(move):
            facts.append(Motif("capture", "captures a pawn en passant"))
        else:
            facts.append(Motif("capture", f"captures {piece_label(board, move.to_square)}"))
    if move.promotion:
        facts.append(Motif("promotion", f"promotes to a {chess.piece_name(move.promotion)}"))
    if after.is_check():
        facts.append(Motif("check", "gives check"))
    return facts


def _mates(after: chess.Board, wanted: set[str]) -> list[Motif]:
    if not after.is_checkmate():
        return []
    found = []
    if "mate" in wanted:
        found.append(Motif("mate", "checkmate"))
    if "back-rank-mate" in wanted:
        king = after.king(after.turn)
        back_rank = 0 if after.turn == chess.WHITE else 7
        if king is not None and chess.square_rank(king) == back_rank:
            for checker in after.checkers():
                if (
                    _piece(after, checker).piece_type in (chess.ROOK, chess.QUEEN)
                    and chess.square_rank(checker) == back_rank
                ):
                    found.append(Motif("back-rank-mate", "back-rank mate: the king is trapped behind its own pieces"))
                    break
    return found


def _fork(after: chess.Board, to: chess.Square) -> Motif | None:
    mover = _piece(after, to)
    enemy = not mover.color
    targets = []
    for target in after.attacks(to):
        piece = after.piece_at(target)
        if not piece or piece.color != enemy:
            continue
        if (
            piece.piece_type == chess.KING
            or VALUES[piece.piece_type] > VALUES[mover.piece_type]
            or not _defended(after, target)
        ):
            targets.append(target)
    if len(targets) < 2 or all(after.piece_type_at(t) == chess.PAWN for t in targets):
        return None
    targets.sort(key=lambda s: -_value(after, s))
    named = " and ".join(piece_label(after, t) for t in targets)
    return Motif("fork", f"fork: attacks {named}", tuple(chess.square_name(t) for t in targets))


def _pins_and_skewers(after: chess.Board, to: chess.Square, wanted: set[str]) -> list[Motif]:
    mover = _piece(after, to)
    if mover.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return []
    found = []
    for front, behind in _sliding_pairs(after, to):
        front_value, behind_value = _value(after, front), _value(after, behind)
        squares = (chess.square_name(front), chess.square_name(behind))
        pinned_pawn = after.piece_type_at(front) == chess.PAWN  # pawn pins are noise, not a tactic
        if "pin" in wanted and behind_value > front_value and not pinned_pawn:
            anchor = "the king" if after.piece_type_at(behind) == chess.KING else piece_label(after, behind)
            found.append(Motif("pin", f"pin: pins {piece_label(after, front)} to {anchor}", squares))
        elif "skewer" in wanted and front_value > behind_value:
            front_matters = after.piece_type_at(front) == chess.KING or front_value > VALUES[mover.piece_type]
            behind_falls = not _defended(after, behind) or behind_value >= VALUES[mover.piece_type]
            if front_matters and behind_falls:
                found.append(
                    Motif(
                        "skewer",
                        f"skewer: attacks {piece_label(after, front)}, with {piece_label(after, behind)} behind it",
                        squares,
                    )
                )
    return found


def _exploits_pin(board: chess.Board, move: chess.Move, after: chess.Board) -> list[Motif]:
    """Capturing a piece pinned to its king, or attacking one with the moved piece."""
    to = move.to_square
    enemy = not board.turn
    found = []
    captured = board.piece_at(to)
    if captured and captured.piece_type != chess.PAWN and board.is_pinned(enemy, to):
        found.append(
            Motif(
                "pin",
                f"pin: captures the pinned {chess.piece_name(captured.piece_type)} on {chess.square_name(to)}",
                (chess.square_name(to),),
            )
        )
    for target in after.attacks(to):
        piece = after.piece_at(target)
        if piece and piece.color == enemy and piece.piece_type != chess.PAWN and after.is_pinned(enemy, target):
            found.append(
                Motif(
                    "pin",
                    f"pin: attacks the pinned {chess.piece_name(piece.piece_type)} on {chess.square_name(target)}",
                    (chess.square_name(target),),
                )
            )
    return found


def _discovered(board: chess.Board, move: chess.Move, after: chess.Board) -> list[Motif]:
    if board.is_castling(move):
        return []  # the castled rook's new lines are not discovered
    color = board.turn
    found = []
    for square in (
        after.pieces(chess.BISHOP, color) | after.pieces(chess.ROOK, color) | after.pieces(chess.QUEEN, color)
    ):
        if square == move.to_square:
            continue
        revealed = after.attacks(square) & after.occupied_co[not color] & ~board.attacks(square)
        for target in revealed:
            slider = piece_label(after, square)
            if after.piece_type_at(target) == chess.KING:
                found.append(
                    Motif("discovered-attack", f"discovered check from {slider}", (chess.square_name(square),))
                )
            elif after.piece_type_at(target) == chess.PAWN:
                continue  # a pawn coming into view is rarely the point
            elif (
                not after.is_check()
                and square in after.attacks(target)
                and _value(after, target) >= _value(after, square)
            ):
                continue  # the target has time to take the attacker
            elif _value(after, target) > _value(after, square) or not _defended(after, target):
                found.append(
                    Motif(
                        "discovered-attack",
                        f"discovered attack: {slider} now attacks {piece_label(after, target)}",
                        (chess.square_name(square), chess.square_name(target)),
                    )
                )
    return found


def _wins_material(board: chess.Board, move: chess.Move, after: chess.Board) -> Motif | None:
    if not board.is_capture(move) or board.is_en_passant(move):
        return None
    captured = _value(board, move.to_square)
    victim = piece_label(board, move.to_square)
    recaptures = any(reply.to_square == move.to_square for reply in after.legal_moves)
    if not recaptures:
        return Motif("hanging-piece", f"wins {victim}, which was undefended", (chess.square_name(move.to_square),))
    mover = _piece(board, move.from_square).piece_type
    if captured > VALUES[mover]:
        return Motif(
            "hanging-piece",
            f"wins material: {victim} for a {chess.piece_name(mover)}",
            (chess.square_name(move.to_square),),
        )
    return None


def _safety(board: chess.Board, move: chess.Move, after: chess.Board) -> list[Motif]:
    """Pieces the move leaves to be won, which were safe before it."""
    if after.is_checkmate():
        return []
    color = board.turn
    already_lost: set[chess.Square] = set()
    if not board.is_check():
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        already_lost = {s for s in chess.SquareSet(passed.occupied_co[color]) if losing_capture(passed, s)}
    captured = _value(board, move.to_square) if board.is_capture(move) else 0
    warnings = []
    for square in chess.SquareSet(after.occupied_co[color]):
        if after.piece_type_at(square) == chess.KING:
            continue
        if square == move.to_square:
            if captured >= _value(after, square):
                continue  # an even or winning trade, not a blunder
        elif square in already_lost:
            continue
        capture = losing_capture(after, square)
        if capture:
            warnings.append(
                Motif(
                    "safety",
                    f"warning: leaves {piece_label(after, square)} to be won by {piece_label(after, capture.from_square)}",
                    (chess.square_name(square),),
                )
            )
    return warnings


def describe_move(
    board: chess.Board,
    move: chess.Move,
    concepts: Iterable[str] = CONCEPTS,
    *,
    facts: bool = True,
) -> list[Motif]:
    """Tags for ``move`` in ``board``: facts (if ``facts``) then the enabled concepts."""
    wanted = set(concepts)
    unknown = wanted - set(CONCEPTS)
    if unknown:
        raise ValueError(f"Unknown chess concepts: {sorted(unknown)}. Known: {list(CONCEPTS)}")
    after = board.copy(stack=False)
    after.push(move)
    motifs = _facts(board, move, after) if facts else []
    motifs += _mates(after, wanted)
    if "pin" in wanted:
        motifs += _exploits_pin(board, move, after)  # also on mate: "takes the pinned queen" is the idea
    if after.is_checkmate():
        return motifs
    piece_survives = losing_capture(after, move.to_square) is None
    if "fork" in wanted and piece_survives:
        fork = _fork(after, move.to_square)
        if fork:
            motifs.append(fork)
    if wanted & {"pin", "skewer"} and piece_survives:
        motifs += _pins_and_skewers(after, move.to_square, wanted)
    if "discovered-attack" in wanted:
        motifs += _discovered(board, move, after)
    if "hanging-piece" in wanted:
        won = _wins_material(board, move, after)
        if won:
            motifs.append(won)
    if "safety" in wanted:
        motifs += _safety(board, move, after)
    return motifs
