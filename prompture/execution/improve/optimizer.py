"""The optimizer adapter, and the two implementations behind it.

The roadmap asks for "an optimizer adapter, initially comparing simple candidate
search with a GEPA integration instead of reimplementing an optimizer".  That is
what this is:

:class:`Optimizer`
    The adapter interface.  Given a set of proposed candidates and an
    ``evaluate`` callable, return the ranked results.  It sees the **development
    split only** — :func:`~prompture.execution.fixtures.assert_optimization_safe`
    is called on the cases before anything runs, so a held-out leak fails loudly
    instead of quietly inflating a scorecard.
:class:`CandidateSearchOptimizer`
    The built-in: evaluate every proposed candidate on dev, rank, return.  It is
    deliberately not clever.  It has no dependencies, it is deterministic, and
    it is the honest baseline any fancier optimiser has to beat.
:class:`GEPAOptimizerAdapter`
    An optional bridge to an installed GEPA implementation.  It is **not**
    a reimplementation: when the dependency is absent it raises a clear,
    actionable error rather than silently degrading to something else and
    reporting the result as GEPA.

Which one to use is an evaluation question, not a taste question — run both
through :func:`compare_optimizers` and read the scorecards.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..fixtures import FixtureCase, assert_optimization_safe
from .candidates import Candidate

logger = logging.getLogger("prompture.execution.improve")

__all__ = [
    "CandidateSearchOptimizer",
    "GEPAOptimizerAdapter",
    "OptimizationResult",
    "Optimizer",
    "OptimizerUnavailable",
    "compare_optimizers",
]


class OptimizerUnavailable(RuntimeError):
    """Raised when an optimizer's external dependency is not installed.

    Deliberately not caught internally: an optimiser that silently falls back to
    a different algorithm and keeps its own name in the report is worse than one
    that refuses to run.
    """


#: ``(candidate, cases) -> mean score in 0..1``.  Supplied by the caller so the
#: optimizer never has to know how the artefact is applied or how it is graded.
EvaluateFn = Callable[[Candidate, Sequence[FixtureCase]], float]


@dataclass
class OptimizationResult:
    """The outcome of one optimizer run over a candidate set.

    Attributes:
        optimizer: Which optimizer produced this.
        ranked: ``[(candidate, dev score)]`` best first.
        evaluations: How many candidate evaluations ran.
        elapsed_ms: Wall clock.
        notes: Caveats.
    """

    optimizer: str
    ranked: list[tuple[Candidate, float]] = field(default_factory=list)
    evaluations: int = 0
    elapsed_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Candidate | None:
        return self.ranked[0][0] if self.ranked else None

    @property
    def best_score(self) -> float | None:
        return self.ranked[0][1] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "ranked": [{"candidate_id": c.id, "version": c.version, "score": s} for c, s in self.ranked],
            "evaluations": self.evaluations,
            "elapsed_ms": self.elapsed_ms,
            "notes": list(self.notes),
        }


class Optimizer(Protocol):
    """The adapter interface every optimizer implements."""

    name: str

    def optimize(
        self,
        candidates: Sequence[Candidate],
        cases: Sequence[FixtureCase],
        evaluate: EvaluateFn,
    ) -> OptimizationResult:  # pragma: no cover - protocol
        ...


class CandidateSearchOptimizer:
    """Evaluate every proposed candidate on the dev split and rank them.

    No search heuristics, no mutation, no model calls of its own.  Its value is
    that it is trustworthy: the ranking is exactly the measured dev score, and
    the number of evaluations is exactly the number of candidates.

    Args:
        max_candidates: Cap on how many candidates are evaluated, so a large
            proposal set cannot silently turn into a large evaluation bill.
            Excess candidates are reported in the result's notes, not dropped
            silently.
        include_baseline: When a baseline candidate is passed in the set, keep
            it in the ranking so "no change" can win.

    Example::

        optimizer = CandidateSearchOptimizer(max_candidates=8)
        result = optimizer.optimize(candidates, fixtures.dev(), evaluate)
        print(result.best.id, result.best_score)
    """

    name = "candidate_search"

    def __init__(self, *, max_candidates: int = 10, include_baseline: bool = True) -> None:
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1 (got {max_candidates})")
        self.max_candidates = max_candidates
        self.include_baseline = include_baseline

    def optimize(
        self,
        candidates: Sequence[Candidate],
        cases: Sequence[FixtureCase],
        evaluate: EvaluateFn,
    ) -> OptimizationResult:
        assert_optimization_safe(list(cases))
        result = OptimizationResult(optimizer=self.name)
        started = time.perf_counter()

        selected = list(candidates)[: self.max_candidates]
        if len(candidates) > self.max_candidates:
            result.notes.append(
                f"{len(candidates) - self.max_candidates} candidate(s) were not evaluated "
                f"(max_candidates={self.max_candidates}); they remain proposed, not rejected"
            )

        scored: list[tuple[Candidate, float]] = []
        for candidate in selected:
            try:
                score = float(evaluate(candidate, cases))
            except Exception as exc:
                logger.warning("candidate %s failed to evaluate: %s", candidate.id, exc)
                result.notes.append(f"candidate {candidate.id} failed to evaluate: {type(exc).__name__}: {exc}")
                continue
            result.evaluations += 1
            scored.append((candidate, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        result.ranked = scored
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        result.notes.append(f"ranking is the measured mean score on {len(cases)} development case(s)")
        return result


class GEPAOptimizerAdapter:
    """Bridge to an installed GEPA optimizer.  Optional dependency.

    GEPA (reflective prompt evolution driven by execution traces and textual
    feedback) is not reimplemented here.  This adapter hands it the candidate
    set and the evaluation callable and translates its output back into an
    :class:`OptimizationResult`.

    Availability is checked at construction and again at
    :meth:`optimize`; when the dependency is missing the adapter raises
    :class:`OptimizerUnavailable` with an install hint.  It never falls back to
    :class:`CandidateSearchOptimizer` on its own — the caller decides that, and
    the result then honestly says which optimizer ran.

    Args:
        module: Import path of the GEPA implementation to bridge to.  Defaults
            to ``"gepa"``; DSPy users typically want ``"dspy"`` and their own
            ``runner``.
        runner: Callable that actually drives the external optimizer, given
            ``(module, candidates, cases, evaluate)`` and returning
            ``[(candidate, score)]``.  Supplying this is how a host adapts to
            whichever GEPA API version it has installed, since the upstream
            surface is not stable across releases.
        require: Raise at construction when the module is missing.  Set
            ``False`` to build the adapter and defer the error to
            :meth:`optimize`.
    """

    name = "gepa"

    def __init__(
        self,
        *,
        module: str = "gepa",
        runner: Callable[..., list[tuple[Candidate, float]]] | None = None,
        require: bool = True,
    ) -> None:
        self.module = module
        self.runner = runner
        if require:
            self._require_module()

    @staticmethod
    def is_available(module: str = "gepa") -> bool:
        """Whether the external optimizer can be imported."""
        import importlib.util

        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    def _require_module(self) -> Any:
        import importlib

        try:
            return importlib.import_module(self.module)
        except ImportError as exc:
            raise OptimizerUnavailable(
                f"GEPAOptimizerAdapter needs the {self.module!r} package, which is not "
                f"installed ({exc}). Install it, or use CandidateSearchOptimizer — which "
                "has no dependencies and is the baseline GEPA has to beat anyway."
            ) from exc

    def optimize(
        self,
        candidates: Sequence[Candidate],
        cases: Sequence[FixtureCase],
        evaluate: EvaluateFn,
    ) -> OptimizationResult:
        assert_optimization_safe(list(cases))
        module = self._require_module()
        if self.runner is None:
            raise OptimizerUnavailable(
                "GEPAOptimizerAdapter needs a `runner=` callable that drives your "
                f"installed {self.module!r} version. Its public API is not stable across "
                "releases, so this adapter deliberately does not guess at it: pass a "
                "small function of (module, candidates, cases, evaluate) -> [(candidate, score)]."
            )
        started = time.perf_counter()
        ranked = list(self.runner(module, candidates, cases, evaluate))
        ranked.sort(key=lambda pair: (-pair[1], pair[0].id))
        return OptimizationResult(
            optimizer=self.name,
            ranked=ranked,
            evaluations=len(ranked),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            notes=[f"driven by the installed {self.module!r} package via a caller-supplied runner"],
        )


def compare_optimizers(
    optimizers: dict[str, Optimizer],
    candidates: Sequence[Candidate],
    cases: Sequence[FixtureCase],
    evaluate: EvaluateFn,
) -> dict[str, OptimizationResult]:
    """Run several optimizers over the same candidates and dev cases.

    An optimizer whose dependency is missing is recorded as an unavailable
    result rather than aborting the comparison — the point is to find out
    whether the extra machinery earns its place, and "it was not installed" is a
    legitimate finding.
    """
    results: dict[str, OptimizationResult] = {}
    for label, optimizer in optimizers.items():
        try:
            results[label] = optimizer.optimize(candidates, cases, evaluate)
        except OptimizerUnavailable as exc:
            results[label] = OptimizationResult(
                optimizer=getattr(optimizer, "name", label),
                notes=[f"unavailable: {exc}"],
            )
    return results
