"""Bounded memory of a side's own accepted decisions, tied to the exact game line.

A harness remembers the moves its model chose, with their short reason and
plan, so the next turn can reassess the plan. Policies:

* ``"none"`` — nothing is remembered.
* ``"previous"`` — the side's last accepted decision, shown only on its very
  next turn on the same line (the chess default).
* an integer ``n`` — the side's last ``n`` accepted decisions on this line.

Memory needs the game's line identity (:class:`~prompture.games.base.GameLine`
via ``game.history``), so nothing crosses sides, games, puzzles or branches:
an entry is only used while the current line still contains its move.

Memory advances when a move is *applied*, not when ``choose`` returns: a
decision is pending until the game line shows its move was played (or the
host calls ``commit``). Choosing again on the same position replaces the
pending decision instead of duplicating it. Rejected attempts and opponents'
explanations are never stored. A rescued move is recorded without any reason
or plan and ends the side's chain of model plans.

:meth:`DecisionMemory.checkpoint` / :meth:`DecisionMemory.restore` save and
validate memory for a resumed game (version, game, initial state and move
prefix); ``restore(None, state)`` explicitly resumes a legacy game with no
saved memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..exceptions import PromptureError
from .base import GameLine

MEMORY_VERSION = "1"
MAX_ENTRIES = 64  # per side; windows are far smaller


class CheckpointError(PromptureError, ValueError):
    """A memory checkpoint does not belong to this game, version or line."""


@dataclass(frozen=True)
class MemoryEntry:
    """One accepted own decision. ``line`` ends with this move (its ply is ``len(line.plies) - 1``)."""

    side: str
    move: str
    reason: str
    plan: str
    line: GameLine
    rescued: bool = False

    @property
    def ply(self) -> int:
        return len(self.line.plies) - 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "move": self.move,
            "reason": self.reason,
            "plan": self.plan,
            "line": self.line.to_dict(),
            "rescued": self.rescued,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            str(data["side"]),
            str(data["move"]),
            str(data.get("reason", "")),
            str(data.get("plan", "")),
            GameLine.from_dict(data["line"]),
            bool(data.get("rescued", False)),
        )


def parse_policy(policy: str | int | None) -> int:
    """``"none"``/``None`` -> 0, ``"previous"`` -> -1 (next-turn only), ``n`` -> window ``n``."""
    if policy in (None, "none", 0):
        return 0
    if policy == "previous":
        return -1
    if isinstance(policy, int) and not isinstance(policy, bool) and policy > 0:
        return min(policy, MAX_ENTRIES)
    raise ValueError(f"memory must be 'none', 'previous' or a positive window size, got {policy!r}")


class DecisionMemory:
    """Per-harness memory of accepted own decisions (see the module docstring)."""

    def __init__(self, game: Any, policy: str | int | None = "none") -> None:
        self.game = game
        self.window = parse_policy(policy)
        self.policy = "none" if self.window == 0 else "previous" if self.window == -1 else self.window
        if self.window and not callable(getattr(game, "history", None)):
            raise ValueError(f"{game.title} has no history(); memory needs line identity (use memory='none')")
        self._entries: dict[str, list[MemoryEntry]] = {}
        self._pending: dict[str, MemoryEntry] = {}
        self.notes: list[str] = []

    @property
    def enabled(self) -> bool:
        return self.window != 0

    def clear(self) -> None:
        self._entries.clear()
        self._pending.clear()

    # -- advancing ------------------------------------------------------------------

    def propose(self, state: Any, decision: Any) -> None:
        """Hold ``decision`` (chosen in ``state``) until its move is seen applied."""
        if not self.enabled:
            return
        entry = self._entry(state, decision)
        if entry.rescued:
            self._entries.pop(entry.side, None)  # a rescue ends the chain of model plans
        self._pending[entry.side] = entry

    def commit(self, state: Any, decision: Any) -> None:
        """Record ``decision`` as applied in ``state`` now (hosts that apply moves themselves)."""
        if not self.enabled:
            return
        entry = self._entry(state, decision)
        self._pending.pop(entry.side, None)
        if entry.rescued:
            self._entries.pop(entry.side, None)
        self._store(entry)

    def sync(self, line: GameLine) -> None:
        """Commit pending decisions whose move the current line contains; drop diverged ones."""
        for side, entry in list(self._pending.items()):
            if line.extends(entry.line):
                self._store(entry)
                del self._pending[side]
            elif not entry.line.extends(line):
                del self._pending[side]  # the host played something else

    def _entry(self, state: Any, decision: Any) -> MemoryEntry:
        before = self.game.history(state)
        side = self.game.player(state)
        after = GameLine(before.root, (*before.plies, (side, decision.name)))
        if decision.rescued:
            return MemoryEntry(side, decision.name, "", "", after, rescued=True)
        return MemoryEntry(side, decision.name, decision.reason, decision.plan, after)

    def _store(self, entry: MemoryEntry) -> None:
        entries = [e for e in self._entries.get(entry.side, []) if e.line != entry.line]
        entries.append(entry)
        entries.sort(key=lambda e: e.ply)
        self._entries[entry.side] = entries[-MAX_ENTRIES:]

    # -- reading ----------------------------------------------------------------------

    def recall(self, state: Any) -> list[MemoryEntry]:
        """The side to move's own decisions to show now, oldest first."""
        if not self.enabled:
            return []
        line = self.game.history(state)
        self.sync(line)
        side = self.game.player(state)
        mine = [e for e in self._entries.get(side, []) if line.extends(e.line)]
        if self.window == -1:
            if not mine:
                return []
            last = mine[-1]
            later_own = any(s == side for s, _ in line.plies[last.ply + 1 :])
            return [] if later_own or last.rescued else [last]
        return [e for e in mine if not e.rescued][-self.window :]

    def render(self, entries: list[MemoryEntry]) -> str:
        if not entries:
            return ""
        if self.window == -1:
            last = entries[-1]
            return (
                f"Your previous move: {last.move}\nYour previous reason: {last.reason}"
                f"\nYour previous plan: {last.plan or '(none given)'}"
                "\nReassess this plan after the opponent's reply; keep it if it still fits."
            )
        lines = [
            "Your recent decisions on this game (oldest first). They are notes, not orders: "
            "check each against the position above."
        ]
        for entry in entries:
            plan = f"; plan: {entry.plan}" if entry.plan else ""
            lines.append(f"- ply {entry.ply + 1}, {entry.move}: {entry.reason or '(no reason given)'}{plan}")
        return "\n".join(lines)

    # -- checkpoints --------------------------------------------------------------------

    def checkpoint(self, state: Any) -> dict[str, Any]:
        """Committed memory valid at ``state`` (pending decisions included if already applied)."""
        line = self.game.history(state)
        self.sync(line)
        entries = [e for side in self._entries.values() for e in side if line.extends(e.line)]
        return {
            "memory_version": MEMORY_VERSION,
            "game": self.game.title,
            "policy": self.policy,
            "line": line.to_dict(),
            "entries": [e.to_dict() for e in sorted(entries, key=lambda e: e.ply)],
        }

    def restore(self, checkpoint: dict[str, Any] | None, state: Any) -> None:
        """Replace memory with a validated checkpoint for ``state``; ``None`` resumes without memory."""
        self.clear()
        if checkpoint is None:
            self.notes.append("resumed without saved memory (legacy game)")
            return
        if checkpoint.get("memory_version") != MEMORY_VERSION:
            raise CheckpointError(
                f"memory checkpoint version {checkpoint.get('memory_version')!r} is not {MEMORY_VERSION}"
            )
        if checkpoint.get("game") != self.game.title:
            raise CheckpointError(f"memory checkpoint is for {checkpoint.get('game')!r}, not {self.game.title}")
        line = self.game.history(state)
        saved = GameLine.from_dict(checkpoint["line"])
        if saved.root != line.root:
            raise CheckpointError("memory checkpoint starts from a different initial position")
        if not line.extends(saved):
            raise CheckpointError("the game diverged from the checkpoint's moves")
        for data in checkpoint.get("entries", []):
            entry = MemoryEntry.from_dict(data)
            if not saved.extends(entry.line):
                raise CheckpointError(f"checkpoint entry at ply {entry.ply + 1} is not on the saved line")
            self._store(entry)
        if checkpoint.get("policy") != self.policy:
            self.notes.append(f"checkpoint saved with memory={checkpoint.get('policy')!r}, restored as {self.policy!r}")
