"""Explicit promotion, retained baselines, and rollback.

Nothing in :mod:`prompture.execution.improve` changes what production uses until
:meth:`PromotionLedger.promote` is called, and that call refuses unless:

* a scorecard exists and its held-out comparison justifies the change,
* the candidate's ``base_version`` still matches the currently active version —
  a candidate measured against a prompt that has since changed was measured
  against something else,
* and the caller has not asked to skip the checks without saying so.

Promotion keeps the previous version.  :meth:`PromotionLedger.rollback` restores
it in one call, and the ledger records every promotion and rollback with the
scorecard that justified it, so "why is production running this prompt?" always
has an answer.

The ledger stores *values*, not side effects: it is the record of which version
is active, and the host applies it (registering the skill, loading the prompt).
That separation is deliberate — a data structure that silently reached into the
global skill registry would be exactly the "silent production self-modification"
this phase exists to prevent.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .candidates import Candidate, CandidateStatus
from .scorecard import Scorecard

__all__ = ["ActiveVersion", "PromotionError", "PromotionLedger", "PromotionRecord"]


class PromotionError(RuntimeError):
    """Raised when a promotion or rollback is refused."""


@dataclass
class ActiveVersion:
    """What is currently live for one target."""

    target: str
    version: str
    content: Any
    candidate_id: str = ""
    promoted_at: float = field(default_factory=time.time)
    origin: str = "baseline"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActiveVersion:
        return cls(
            target=data["target"],
            version=data.get("version", ""),
            content=data.get("content"),
            candidate_id=data.get("candidate_id", ""),
            promoted_at=float(data.get("promoted_at", time.time())),
            origin=data.get("origin", "baseline"),
        )


@dataclass
class PromotionRecord:
    """One promotion or rollback, with the evidence behind it."""

    target: str
    action: str  # "promote" | "rollback" | "baseline"
    to_version: str
    from_version: str = ""
    candidate_id: str = ""
    reason: str = ""
    scorecard: dict[str, Any] | None = None
    forced: bool = False
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromotionRecord:
        return cls(
            target=data["target"],
            action=data.get("action", "promote"),
            to_version=data.get("to_version", ""),
            from_version=data.get("from_version", ""),
            candidate_id=data.get("candidate_id", ""),
            reason=data.get("reason", ""),
            scorecard=data.get("scorecard"),
            forced=bool(data.get("forced", False)),
            at=float(data.get("at", time.time())),
        )


class PromotionLedger:
    """Tracks the active version per target, with history and rollback.

    Example::

        ledger = PromotionLedger("promotions.json")
        ledger.set_baseline("extraction_prompt", base_text)

        ok, why = scorecard.promotable(min_gain=0.02)
        if ok:
            ledger.promote(candidate, scorecard)
        else:
            print("not promoting:", why)

        ledger.rollback("extraction_prompt")   # back to the retained baseline
    """

    def __init__(self, path: str | Path | None = None, *, min_gain: float = 0.0) -> None:
        self.path = Path(path) if path is not None else None
        self.min_gain = min_gain
        self._active: dict[str, ActiveVersion] = {}
        self._history: dict[str, list[ActiveVersion]] = {}
        self._records: list[PromotionRecord] = []
        self._lock = threading.RLock()
        if self.path is not None and self.path.exists():
            self._load()

    # ---- persistence --------------------------------------------------

    def _load(self) -> None:
        assert self.path is not None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for target, item in (payload.get("active") or {}).items():
            self._active[target] = ActiveVersion.from_dict(item)
        for target, items in (payload.get("history") or {}).items():
            self._history[target] = [ActiveVersion.from_dict(i) for i in items]
        self._records = [PromotionRecord.from_dict(r) for r in payload.get("records") or []]

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "active": {k: v.to_dict() for k, v in self._active.items()},
            "history": {k: [i.to_dict() for i in v] for k, v in self._history.items()},
            "records": [r.to_dict() for r in self._records],
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ---- reading ------------------------------------------------------

    def active(self, target: str) -> ActiveVersion | None:
        with self._lock:
            return self._active.get(target)

    def active_content(self, target: str, default: Any = None) -> Any:
        version = self.active(target)
        return default if version is None else version.content

    def history(self, target: str) -> list[ActiveVersion]:
        """Retained previous versions, oldest first."""
        with self._lock:
            return list(self._history.get(target, []))

    def records(self, target: str | None = None) -> list[PromotionRecord]:
        with self._lock:
            if target is None:
                return list(self._records)
            return [r for r in self._records if r.target == target]

    def targets(self) -> list[str]:
        with self._lock:
            return sorted(self._active)

    # ---- writing ------------------------------------------------------

    def set_baseline(self, target: str, content: Any, *, version: str = "") -> ActiveVersion:
        """Record the starting version for *target*.

        Idempotent for the same content; calling it again with different content
        replaces the baseline and retains the old one, so nothing is lost.
        """
        from .candidates import content_version

        resolved = version or content_version(content)
        with self._lock:
            previous = self._active.get(target)
            if previous is not None and previous.version == resolved:
                return previous
            if previous is not None:
                self._history.setdefault(target, []).append(previous)
            active = ActiveVersion(target=target, version=resolved, content=content, origin="baseline")
            self._active[target] = active
            self._records.append(
                PromotionRecord(
                    target=target,
                    action="baseline",
                    to_version=resolved,
                    from_version=previous.version if previous else "",
                    reason="baseline recorded",
                )
            )
            self._flush()
        return active

    def promote(
        self,
        candidate: Candidate,
        scorecard: Scorecard,
        *,
        min_gain: float | None = None,
        force: bool = False,
        force_reason: str = "",
    ) -> ActiveVersion:
        """Make *candidate* the active version for its target.

        Args:
            candidate: The candidate to activate.
            scorecard: The evidence.  Its held-out comparison must justify the
                change; see :meth:`Scorecard.promotable`.
            min_gain: Override the ledger's default minimum held-out gain.
            force: Promote despite a failing or missing scorecard.  Requires
                ``force_reason``, and the record is permanently marked
                ``forced`` so an unjustified promotion stays visible.
            force_reason: Why the checks are being bypassed.

        Raises:
            PromotionError: When the scorecard does not justify the change, when
                the candidate's base version is stale, or when ``force`` is used
                without a reason.
        """
        threshold = self.min_gain if min_gain is None else min_gain
        with self._lock:
            current = self._active.get(candidate.target)

            if force:
                if not force_reason.strip():
                    raise PromotionError(
                        "force=True requires force_reason: a promotion that bypasses its "
                        "evidence must say why, in the permanent record."
                    )
                reason = f"FORCED: {force_reason}"
            else:
                if candidate.base_version and current is not None and current.version != candidate.base_version:
                    raise PromotionError(
                        f"Candidate {candidate.id} was measured against {candidate.target} "
                        f"version {candidate.base_version[:8]}, but {current.version[:8]} is "
                        "active now. Re-evaluate against the current version before promoting."
                    )
                ok, why = scorecard.promotable(min_gain=threshold)
                if not ok:
                    raise PromotionError(f"Refusing to promote candidate {candidate.id}: {why}")
                reason = why

            if current is not None:
                self._history.setdefault(candidate.target, []).append(current)

            active = ActiveVersion(
                target=candidate.target,
                version=candidate.version,
                content=candidate.content,
                candidate_id=candidate.id,
                origin=candidate.origin,
            )
            self._active[candidate.target] = active
            self._records.append(
                PromotionRecord(
                    target=candidate.target,
                    action="promote",
                    to_version=candidate.version,
                    from_version=current.version if current else "",
                    candidate_id=candidate.id,
                    reason=reason,
                    scorecard=scorecard.to_dict(),
                    forced=force,
                )
            )
            self._flush()

        candidate.status = CandidateStatus.PROMOTED
        return active

    def reject(self, candidate: Candidate, scorecard: Scorecard | None = None, *, reason: str = "") -> Candidate:
        """Mark a candidate rejected, keeping it so the idea is not re-proposed blindly."""
        candidate.status = CandidateStatus.REJECTED
        if reason:
            candidate.metadata["rejection_reason"] = reason
        elif scorecard is not None:
            candidate.metadata["rejection_reason"] = scorecard.promotable()[1]
        return candidate

    def rollback(self, target: str, *, reason: str = "manual rollback") -> ActiveVersion:
        """Restore the previous version for *target*.

        Raises:
            PromotionError: When there is nothing to roll back to.
        """
        with self._lock:
            history = self._history.get(target) or []
            if not history:
                raise PromotionError(
                    f"No retained previous version for {target!r}; nothing to roll back to. "
                    "Record a baseline with set_baseline() before promoting."
                )
            current = self._active.get(target)
            restored = history.pop()
            self._active[target] = restored
            self._records.append(
                PromotionRecord(
                    target=target,
                    action="rollback",
                    to_version=restored.version,
                    from_version=current.version if current else "",
                    candidate_id=current.candidate_id if current else "",
                    reason=reason,
                )
            )
            self._flush()
        return restored
