"""Versioned candidates: a proposed change that has not been adopted.

A candidate is the unit that connects the three things the roadmap wants
connected — evaluation failures, opted-in feedback, and mined skill proposals —
to something that can be *tested* before it changes anything.

The critical property is that a candidate is inert.  Creating one modifies no
prompt, registers no skill, and changes no production behaviour.  Only
:mod:`prompture.execution.improve.promotion` can activate one, only with a
scorecard, and always with the previous version retained for rollback.

Every candidate carries its provenance: what produced it, which outcome records
motivated it, and which base version it is a change *to*.  A candidate whose
base version no longer matches the active version is stale and the promotion
ledger refuses it, because it was measured against something else.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "Candidate",
    "CandidateKind",
    "CandidateStatus",
    "CandidateStore",
    "candidate_from_failures",
    "candidate_from_skill_proposal",
]


class CandidateKind(str, Enum):
    """What kind of artefact a candidate proposes to change."""

    PROMPT = "prompt"
    SKILL = "skill"
    STRATEGY_CONFIG = "strategy_config"


class CandidateStatus(str, Enum):
    """Where a candidate is in its lifecycle."""

    #: Created, never evaluated.
    PROPOSED = "proposed"
    #: Evaluated on the development split; a scorecard exists.
    EVALUATED = "evaluated"
    #: Evaluated and rejected — kept, so the same idea is not re-proposed blindly.
    REJECTED = "rejected"
    #: Active.  Exactly one candidate per target may hold this.
    PROMOTED = "promoted"
    #: Was promoted, then rolled back.
    ROLLED_BACK = "rolled_back"


@dataclass
class Candidate:
    """A proposed, versioned, not-yet-active change.

    Attributes:
        target: What this changes — a prompt name, skill name, or strategy id.
            Candidates compete per target.
        kind: See :class:`CandidateKind`.
        content: The proposed text (a prompt body, skill instructions) or, for
            ``STRATEGY_CONFIG``, a JSON-serialisable options dict.
        base_content: The content this was derived from, so a diff can be shown
            and so staleness can be detected at promotion time.
        base_version: Version handle of the active artefact this was built
            against.  Promotion refuses a candidate whose base no longer matches.
        version: This candidate's own version handle, derived from its content.
        origin: What produced it — ``"skill_miner"``, ``"eval_failures"``,
            ``"feedback"``, ``"manual"``, ``"optimizer"``.
        motivating_task_ids: Outcome-record task ids that motivated it, so a
            reviewer can look at the actual failures.
        rationale: Why this change is expected to help.
        status: See :class:`CandidateStatus`.
        metadata: Free-form.
    """

    target: str
    kind: CandidateKind
    content: Any
    base_content: Any = None
    base_version: str = ""
    origin: str = "manual"
    motivating_task_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    status: CandidateStatus = CandidateStatus.PROPOSED
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    version: str = ""

    def __post_init__(self) -> None:
        if not self.version:
            self.version = content_version(self.content)

    # ---- inspection ---------------------------------------------------

    def diff(self, *, context_lines: int = 3) -> str:
        """Unified diff from ``base_content`` to ``content``.

        Returns an empty string when there is no base to diff against — a
        brand-new skill is an addition, not a change.
        """
        if self.base_content is None:
            return ""
        before = _as_lines(self.base_content)
        after = _as_lines(self.content)
        return "\n".join(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"{self.target}@{self.base_version or 'base'}",
                tofile=f"{self.target}@{self.version}",
                lineterm="",
                n=context_lines,
            )
        )

    def summary(self) -> str:
        """One line for a review list."""
        return f"{self.id} {self.kind.value}:{self.target} v{self.version[:8]} [{self.status.value}] from {self.origin}"

    # ---- serialisation ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        return cls(
            target=data["target"],
            kind=CandidateKind(data["kind"]),
            content=data.get("content"),
            base_content=data.get("base_content"),
            base_version=data.get("base_version", ""),
            origin=data.get("origin", "manual"),
            motivating_task_ids=list(data.get("motivating_task_ids") or []),
            rationale=data.get("rationale", ""),
            status=CandidateStatus(data.get("status", CandidateStatus.PROPOSED.value)),
            metadata=dict(data.get("metadata") or {}),
            id=data.get("id", uuid.uuid4().hex[:12]),
            created_at=float(data.get("created_at", time.time())),
            version=data.get("version", ""),
        )


def content_version(content: Any) -> str:
    """A stable version handle derived from content.

    Content-addressed on purpose: two candidates with identical text are the
    same version, and an edited prompt cannot keep an old version number by
    accident.
    """
    if isinstance(content, str):
        payload = content
    else:
        payload = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _as_lines(content: Any) -> list[str]:
    if isinstance(content, str):
        return content.splitlines()
    return json.dumps(content, indent=2, sort_keys=True, default=str).splitlines()


class CandidateStore:
    """A JSON-file-backed collection of candidates.

    Thread-safe, and every mutation rewrites the file atomically, so a crash
    mid-write cannot corrupt the record of what has been proposed and tried.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._items: dict[str, Candidate] = {}
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
        for item in payload.get("candidates", []):
            candidate = Candidate.from_dict(item)
            self._items[candidate.id] = candidate

    def _flush(self) -> None:
        if self.path is None:
            return
        import os
        import tempfile

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "candidates": [c.to_dict() for c in self._items.values()]}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ---- access -------------------------------------------------------

    def add(self, candidate: Candidate) -> Candidate:
        with self._lock:
            self._items[candidate.id] = candidate
            self._flush()
        return candidate

    def extend(self, candidates: Iterable[Candidate]) -> int:
        count = 0
        with self._lock:
            for candidate in candidates:
                self._items[candidate.id] = candidate
                count += 1
            self._flush()
        return count

    def get(self, candidate_id: str) -> Candidate | None:
        with self._lock:
            return self._items.get(candidate_id)

    def update(self, candidate: Candidate) -> Candidate:
        return self.add(candidate)

    def for_target(self, target: str) -> list[Candidate]:
        with self._lock:
            return [c for c in self._items.values() if c.target == target]

    def with_status(self, status: CandidateStatus) -> list[Candidate]:
        with self._lock:
            return [c for c in self._items.values() if c.status is status]

    def __iter__(self) -> Iterator[Candidate]:
        with self._lock:
            return iter(list(self._items.values()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# ---------------------------------------------------------------------------
# Bridges from the two sources the roadmap names
# ---------------------------------------------------------------------------


def candidate_from_skill_proposal(
    proposal: Any,
    *,
    base_content: str | None = None,
    base_version: str = "",
    motivating_task_ids: Iterable[str] = (),
) -> Candidate:
    """Turn a mined :class:`~prompture.agents.skill_miner.SkillProposal` into a candidate.

    This is the roadmap's "a recurring procedure is a *proposal*, then test
    whether it improves task results" step made concrete: the miner's output
    stops at a candidate and does not reach the skill registry until a scorecard
    and an explicit promotion say so.
    """
    return Candidate(
        target=str(getattr(proposal, "name", "unnamed-skill")),
        kind=CandidateKind.SKILL,
        content=str(getattr(proposal, "instructions", "") or ""),
        base_content=base_content,
        base_version=base_version,
        origin="skill_miner",
        motivating_task_ids=list(motivating_task_ids),
        rationale=str(getattr(proposal, "rationale", "") or ""),
        metadata={
            "description": str(getattr(proposal, "description", "") or ""),
            "tool_sequence": list(getattr(proposal, "tool_sequence", ()) or ()),
            "occurrences": int(getattr(proposal, "occurrences", 0) or 0),
            "miner_confidence": float(getattr(proposal, "confidence", 0.0) or 0.0),
            "confidence_is_self_reported": True,
        },
    )


def candidate_from_failures(
    target: str,
    *,
    base_content: str,
    proposed_content: str,
    failures: Iterable[Any],
    base_version: str = "",
    rationale: str = "",
) -> Candidate:
    """Build a prompt candidate motivated by specific failing outcome records.

    Args:
        target: The prompt being changed.
        base_content: The current prompt text.
        proposed_content: The proposed replacement.
        failures: :class:`~prompture.execution.outcomes.OutcomeRecord` objects (or
            anything with a ``task_id``) that motivated the change.  Recorded so
            a reviewer can read the actual failures, not just the claim.
        base_version: Version handle of ``base_content``.
        rationale: Why this is expected to help.
    """
    ids: list[str] = []
    error_kinds: dict[str, int] = {}
    for failure in failures:
        task_id = getattr(failure, "task_id", None)
        if task_id:
            ids.append(str(task_id))
        termination = getattr(failure, "termination", None)
        key = getattr(termination, "value", None) or str(termination or "unknown")
        error_kinds[key] = error_kinds.get(key, 0) + 1

    return Candidate(
        target=target,
        kind=CandidateKind.PROMPT,
        content=proposed_content,
        base_content=base_content,
        base_version=base_version or content_version(base_content),
        origin="eval_failures",
        motivating_task_ids=ids,
        rationale=rationale,
        metadata={"failure_terminations": error_kinds, "failure_count": len(ids)},
    )
