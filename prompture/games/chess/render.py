"""Deterministic chess board pictures for the experimental ``text+image`` representation.

The board is drawn by ``chess.svg`` with coordinates, the last move
highlighted and a checked king marked. The same position (and history tail)
always gives the same bytes, so a record's image digest identifies exactly
what the model saw. Most vision APIs want PNG: that needs the optional
``cairosvg`` package. Without it, rendering a PNG raises and the harness falls
back to text, recording why (a text fallback is not an image trial).
"""

from __future__ import annotations

import chess
import chess.svg

from ..base import RenderedImage

RENDERER = "python-chess-svg"
RENDERER_VERSION = "1"


def board_svg(board: chess.Board, size: int = 400) -> str:
    """The board as SVG: coordinates on, last move highlighted, a checked king marked."""
    lastmove = board.peek() if board.move_stack else None
    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(board, lastmove=lastmove, check=check, coordinates=True, size=size)


def render_board(board: chess.Board, image_format: str = "png", size: int = 400) -> RenderedImage:
    """A :class:`~prompture.games.base.RenderedImage` of ``board`` as ``"png"`` or ``"svg"``."""
    svg = board_svg(board, size).encode("utf-8")
    renderer = f"{RENDERER}/{RENDERER_VERSION} python-chess/{chess.__version__}"
    if image_format == "svg":
        return RenderedImage(svg, "image/svg+xml", renderer)
    if image_format != "png":
        raise ValueError(f"image_format must be 'png' or 'svg', got {image_format!r}")
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError("PNG board pictures need cairosvg (pip install cairosvg)") from exc
    png = cairosvg.svg2png(bytestring=svg)
    return RenderedImage(png, "image/png", f"{renderer} cairosvg/{cairosvg.__version__}")
