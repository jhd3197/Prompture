"""Experimental decision strategies. Off by default; none is known to improve play.

Pass one as ``GameHarness(..., strategy=...)``. Every strategy goes through
the harness's own request path, so legal-move validation, deadlines,
cancellation, budgets, events, accounting and traces apply to every extra
call, and each strategy's name, version and settings enter the configuration
fingerprint.

* :class:`Reconsider` — after the model picks a move that carries a configured
  warning (``safety``/``allows`` by default), ask once more with that warning
  and its witness line. Exactly one review: never a review of the review.
* :class:`Proposals` — ``workers`` independent single-call proposals (two by
  default), then one selector constrained to the proposed moves. Agreement is
  recorded, but agreement is not correctness.

Whether either helps is an empirical question for
:func:`prompture.games.evaluation.compare` under a preregistered design; until
then :func:`experiment_status` reports them as ``live_pending``. Nothing here
promotes a strategy to a default; the baseline is always ``strategy=None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import IMAGE_NOTE, Candidate, MoveDecision
from .records import BudgetExhausted, FailureInfo, FailureKind, TurnCancelled, TurnFailed

_RANK = {"urgent": 0, "warning": 1, "info": 2}


def _checker(harness: Any, state: Any, candidates: list[Candidate], matched: list[Candidate]) -> Any:
    def check(answer: dict[str, Any]) -> str | None:
        chosen, error = harness._match(state, candidates, answer.get("move"))
        if chosen is not None:
            matched.append(chosen)
        return error

    return check


def _stop_on(attempt: Any, turn: Any, harness: Any) -> None:
    """Cancellation and fatal failures end the turn in every strategy."""
    failure = attempt.failure
    if failure is None:
        return
    if failure.kind is FailureKind.CANCELLED:
        raise TurnCancelled(harness._cancel_message(turn), record=turn.record)
    if failure.kind.fatal:
        raise TurnFailed(f"{failure.kind.value} failure: {attempt.error}", kind=failure.kind, record=turn.record)


@dataclass(frozen=True)
class Reconsider:
    """One targeted review when the chosen move carries a configured warning.

    ``on_review_failure`` decides what a failed or refused review does:
    ``"keep"`` plays the original valid choice (marked in the record), ``"stop"``
    fails the turn.
    """

    trigger_kinds: tuple[str, ...] = ("safety", "allows")
    min_severity: str = "warning"
    on_review_failure: str = "keep"
    version: str = "1"

    def __post_init__(self) -> None:
        if self.on_review_failure not in ("keep", "stop"):
            raise ValueError("on_review_failure must be 'keep' or 'stop'")
        if self.min_severity not in _RANK:
            raise ValueError(f"min_severity must be one of {tuple(_RANK)}")

    def identity(self) -> dict[str, Any]:
        return {
            "name": "reconsider",
            "version": self.version,
            "trigger_kinds": list(self.trigger_kinds),
            "min_severity": self.min_severity,
            "on_review_failure": self.on_review_failure,
        }

    def decide(self, harness: Any, turn: Any, state: Any) -> MoveDecision:
        candidates = turn.candidates
        chosen, reason, plan = harness._select(turn, state, candidates)
        if chosen is None:
            return harness._exhausted(turn, candidates)
        info: dict[str, Any] = {**self.identity(), "initial": chosen.name, "final": chosen.name, "reviewed": False}
        turn.record.strategy = info
        warnings = [
            m
            for m in chosen.motifs
            if m.kind in self.trigger_kinds and _RANK.get(m.severity, 2) <= _RANK[self.min_severity]
        ]
        if not warnings:
            return harness._accepted(turn, chosen, reason, plan)
        info["warnings"] = [m.text for m in warnings]
        lines = [f"- {m.text}" + (f" (line: {m.evidence})" if m.evidence else "") for m in warnings]
        prompt = (
            harness._compose(state, candidates)[0]
            + f"\n\nYou chose {chosen.name}. Before it is played, check this warning about it:\n"
            + "\n".join(lines)
            + f"\nKeep {chosen.name} if the warning does not change your choice, or pick another listed move."
        )
        if turn.images:
            prompt += IMAGE_NOTE
        failure = None
        matched: list[Candidate] = []
        try:
            if harness._may_dispatch(turn):
                _answer, attempt = harness._call(
                    turn,
                    prompt,
                    harness.schema(candidates),
                    "review",
                    _checker(harness, state, candidates, matched),
                    images=turn.images,
                )
                _stop_on(attempt, turn, harness)
                if attempt.valid and matched:
                    final, reason, plan = matched[0], attempt.reason, attempt.plan
                    info.update(reviewed=True, final=final.name, revised=final.name != chosen.name)
                    return harness._accepted(turn, final, reason, plan)
                failure = attempt.failure
            else:
                failure = turn.record.failure
        except BudgetExhausted as exc:
            failure = FailureInfo(FailureKind.BUDGET, "budget", str(exc))
        info.update(reviewed=failure is not None, revised=False, review_failure=failure.kind.value if failure else None)
        if self.on_review_failure == "stop":
            raise TurnFailed(
                f"the review failed ({failure.kind.value if failure else 'unknown'})",
                kind=failure.kind if failure else FailureKind.INVALID_OUTPUT,
                record=turn.record,
            )
        turn.record.failure = None  # the original choice stands
        return harness._accepted(turn, chosen, reason, plan)


@dataclass(frozen=True)
class Proposals:
    """``workers`` independent proposals, then one selector constrained to them.

    Each worker makes one call (no retries) with the harness options merged
    with its entry in ``worker_options``. ``on_partial`` decides what a failed
    worker does: ``"use_valid"`` continues with the valid proposals, ``"stop"``
    fails the turn. With no valid proposal the harness's failure policy
    applies. A failed selector plays the first valid proposal (``"first"``) or
    fails the turn (``"stop"``). At most ``workers + 1`` calls per turn.
    """

    workers: int = 2
    select_on_agreement: bool = False
    on_partial: str = "use_valid"
    on_selector_failure: str = "first"
    worker_options: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    version: str = "1"

    def __post_init__(self) -> None:
        if self.workers < 2:
            raise ValueError("Proposals needs at least two workers")
        if self.on_partial not in ("use_valid", "stop") or self.on_selector_failure not in ("first", "stop"):
            raise ValueError("on_partial must be use_valid|stop and on_selector_failure first|stop")

    def identity(self) -> dict[str, Any]:
        return {
            "name": "proposals",
            "version": self.version,
            "workers": self.workers,
            "select_on_agreement": self.select_on_agreement,
            "on_partial": self.on_partial,
            "on_selector_failure": self.on_selector_failure,
            "worker_options": [dict(o) for o in self.worker_options],
        }

    def decide(self, harness: Any, turn: Any, state: Any) -> MoveDecision:
        candidates = turn.candidates
        schema = harness.schema(candidates)
        prompt = harness._compose(state, candidates)[0] + (IMAGE_NOTE if turn.images else "")
        info: dict[str, Any] = {**self.identity(), "proposals": [], "agreement": False, "selector": None}
        turn.record.strategy = info
        valid: list[tuple[Candidate, str, str]] = []
        budget_stopped = False
        for worker in range(self.workers):
            if not harness._may_dispatch(turn):
                break
            options = {**harness.options, **(self.worker_options[worker] if worker < len(self.worker_options) else {})}
            matched: list[Candidate] = []
            try:
                _answer, attempt = harness._call(
                    turn,
                    prompt,
                    schema,
                    "proposal",
                    _checker(harness, state, candidates, matched),
                    images=turn.images,
                    options=options,
                )
            except BudgetExhausted:
                budget_stopped = True
                break
            _stop_on(attempt, turn, harness)
            ok = attempt.valid and bool(matched)
            info["proposals"].append(
                {
                    "worker": worker,
                    "attempt_id": attempt.attempt_id,
                    "move": matched[0].name if ok else None,
                    "failure": attempt.failure.kind.value if attempt.failure else None,
                }
            )
            if ok:
                valid.append((matched[0], attempt.reason, attempt.plan))
            elif attempt.failure is not None:
                turn.record.failure = attempt.failure
        info["partial"] = len(valid) < self.workers
        info["budget_stopped"] = budget_stopped
        if not valid:
            return harness._exhausted(turn, candidates)
        if info["partial"] and self.on_partial == "stop":
            raise TurnFailed(
                f"only {len(valid)} of {self.workers} proposals were valid",
                kind=FailureKind.INVALID_OUTPUT,
                record=turn.record,
            )
        turn.record.failure = None
        names = list(dict.fromkeys(c.name for c, _, _ in valid))
        info["agreement"] = len(names) == 1 and len(valid) > 1
        if len(names) == 1 and (not self.select_on_agreement or len(valid) == 1):
            first = valid[0]
            return harness._accepted(turn, *first)
        if budget_stopped or not harness._may_dispatch(turn):
            info["selector_failure"] = "budget" if budget_stopped else "deadline"
            return self._fallback(harness, turn, valid, info)
        return self._select(harness, turn, state, valid, names, info)

    def _select(
        self,
        harness: Any,
        turn: Any,
        state: Any,
        valid: list[tuple[Candidate, str, str]],
        names: list[str],
        info: dict[str, Any],
    ) -> MoveDecision:
        subset = [c for c in turn.candidates if c.name in names]
        summary = "\n".join(
            f"- {c.name}: {reason or '(no reason)'}" + (f"; plan: {plan}" if plan else "") for c, reason, plan in valid
        )
        prompt = (
            harness._compose(state, subset)[0]
            + "\n\nIndependent analyses proposed these moves (agreement does not make a move correct):\n"
            + summary
            + "\nChoose the best of the proposed moves."
        )
        if turn.images:
            prompt += IMAGE_NOTE
        matched: list[Candidate] = []
        try:
            _answer, attempt = harness._call(
                turn,
                prompt,
                harness.schema(subset),
                "selector",
                _checker(harness, state, subset, matched),
                images=turn.images,
            )
        except BudgetExhausted:
            info["selector_failure"] = "budget"
            return self._fallback(harness, turn, valid, info)
        _stop_on(attempt, turn, harness)
        if attempt.valid and matched:
            info["selector"] = matched[0].name
            return harness._accepted(turn, matched[0], attempt.reason, attempt.plan)
        info["selector_failure"] = attempt.failure.kind.value if attempt.failure else "invalid_output"
        return self._fallback(harness, turn, valid, info)

    def _fallback(
        self, harness: Any, turn: Any, valid: list[tuple[Candidate, str, str]], info: dict[str, Any]
    ) -> MoveDecision:
        if self.on_selector_failure == "stop":
            raise TurnFailed(
                f"the selector failed ({info.get('selector_failure')})",
                kind=FailureKind.INVALID_OUTPUT,
                record=turn.record,
            )
        info["selector"] = None
        info["fallback"] = "first_valid_proposal"
        return harness._accepted(turn, *valid[0])


def experiment_status(comparison: Any | None) -> str:
    """``measured_benefit``, ``regression``, ``inconclusive`` or ``live_pending`` for one experiment."""
    if comparison is None:
        return "live_pending"
    return {
        "improved": "measured_benefit",
        "regression": "regression",
        "not_evidence": "live_pending",
    }.get(comparison.verdict, "inconclusive")


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "reconsider": {"enable": "strategy=Reconsider()", "baseline": "strategy=None", "rollback": "strategy=None"},
    "proposals": {"enable": "strategy=Proposals()", "baseline": "strategy=None", "rollback": "strategy=None"},
    "image": {
        "enable": 'representation="text+image"',
        "baseline": 'representation="text"',
        "rollback": 'representation="text"',
    },
}
