"""Reproducible evaluation for any game: fixtures, scorecards and paired comparisons.

Three questions are kept apart because one number cannot answer them all:

* **strength** — did the chosen move meet the fixture's criterion, without
  harness help (``unassisted_success``)?
* **reliability** — how often was the first answer valid, how many replies
  were invalid, how often did the harness rescue, how did turns fail?
* **efficiency** — tokens, cost and latency per accepted action, with cost
  completeness carried along (an unknown price is never a zero).

A :class:`Fixture` is a position (``initial`` plus ``moves`` so history-aware
rules such as repetition see the real game), with acceptable moves or a named
evaluator for decisions, and/or ``expect``ed hint facts for the harness's own
deterministic checks. Fixtures carry provenance and a ``split``
(``dev``/``validation``/``test``); the held-out ``test`` split is only
evaluated when explicitly requested. A historic blunder is a position where a
warning should appear, not automatically a unique-best-move puzzle.

:func:`evaluate` runs a player over fixtures in a seeded, randomized order and
keeps raw per-sample records; :func:`scorecard` derives every metric from
those records with numerators and denominators. A run is labelled
``evidence="scripted"`` unless the caller declares a live model run, and
:func:`compare` refuses a quality verdict for scripted evidence, for
configurations that differ in more than the declared factor, or without a
:class:`Preregistration`. Small or mixed results are labelled inconclusive.
Harness seeds fix the harness's own randomness, not a provider's sampling.
"""

from __future__ import annotations

import json
import random
import statistics
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .base import Candidate, Game, GameHarness
from .records import FailureKind, TurnError

EVAL_VERSION = "1"
SPLITS = ("dev", "validation", "test")
EVIDENCE = ("scripted", "live")

Evaluator = Callable[[Game, Any, Any], "bool | None"]
_EVALUATORS: dict[tuple[str, str], Evaluator] = {}


class FixtureError(ValueError):
    """A fixture is malformed or its history is not legal in its game."""


def register_evaluator(game_title: str, name: str, evaluator: Evaluator) -> None:
    """Register a named decision evaluator ``(game, state, move) -> bool | None`` for one game."""
    _EVALUATORS[(game_title, name)] = evaluator


def get_evaluator(game_title: str, name: str) -> Evaluator:
    try:
        return _EVALUATORS[(game_title, name)]
    except KeyError:
        raise FixtureError(f"no evaluator {name!r} registered for {game_title}") from None


# -- fixtures --------------------------------------------------------------------


@dataclass(frozen=True)
class Fixture:
    """One evaluation position.

    ``expect`` entries are hint checks: ``{"move": name, "concepts": [...] or
    "all", "facts": bool, "present": [...], "absent": [...], "gap": str}``.
    A ``present``/``absent`` item is a motif kind (``"fork"``) or an exact
    ``"kind: text"``. ``gap`` marks a known detector gap: the check is expected
    to fail until the named work lands, and is reported separately.
    """

    id: str
    game: str
    split: str
    provenance: str
    initial: Any = None
    moves: tuple[str, ...] = ()
    acceptable: tuple[str, ...] = ()
    evaluator: str | None = None
    expect: tuple[dict[str, Any], ...] = ()
    tags: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fixture:
        missing = [k for k in ("id", "game", "split", "provenance") if not data.get(k)]
        if missing:
            raise FixtureError(f"fixture {data.get('id', '?')} is missing {missing}")
        if data["split"] not in SPLITS:
            raise FixtureError(f"fixture {data['id']}: split must be one of {SPLITS}")
        fixture = cls(
            id=str(data["id"]),
            game=str(data["game"]),
            split=str(data["split"]),
            provenance=str(data["provenance"]),
            initial=data.get("initial"),
            moves=tuple(str(m) for m in data.get("moves", ())),
            acceptable=tuple(str(m) for m in data.get("acceptable", ())),
            evaluator=data.get("evaluator"),
            expect=tuple(dict(e) for e in data.get("expect", ())),
            tags=tuple(str(t) for t in data.get("tags", ())),
            notes=str(data.get("notes", "")),
        )
        if not (fixture.acceptable or fixture.evaluator or fixture.expect):
            raise FixtureError(f"fixture {fixture.id} has no acceptable moves, evaluator or hint expectations")
        return fixture

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("moves", "acceptable", "expect", "tags"):
            data[key] = list(data[key])
        return data

    @property
    def decides(self) -> bool:
        """Whether a chosen move can be scored against this fixture."""
        return bool(self.acceptable or self.evaluator)


def load_fixtures(path: str | Path) -> list[Fixture]:
    """Fixtures from ``{"fixtures": [...]}`` or a bare list in a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["fixtures"] if isinstance(data, dict) else data
    return [Fixture.from_dict(item) for item in items]


def fixture_state(game: Game, fixture: Fixture) -> Any:
    """The fixture's position, replayed through the game's own rules (illegal history raises)."""
    if fixture.game != game.title:
        raise FixtureError(f"fixture {fixture.id} is for {fixture.game}, not {game.title}")
    load = getattr(game, "load_state", None)
    if not callable(load):
        raise FixtureError(f"{game.title} cannot load fixture states (no load_state)")
    try:
        return load({"initial": fixture.initial, "moves": list(fixture.moves)})
    except (ValueError, KeyError) as exc:
        raise FixtureError(f"fixture {fixture.id}: {exc}") from exc


def select(
    fixtures: Iterable[Fixture], splits: Iterable[str] | None = None, *, include_test: bool = False
) -> list[Fixture]:
    """Fixtures in ``splits`` (default: all but ``test``). ``test`` needs ``include_test=True``."""
    wanted = set(splits) if splits is not None else {"dev", "validation"}
    if "test" in wanted and not include_test:
        raise ValueError("the held-out test split is only evaluated with include_test=True")
    return [f for f in fixtures if f.split in wanted]


# -- deterministic hint checks -----------------------------------------------------


class _NoCalls:
    """A driver for building candidates only: any model call is a bug."""

    model = "none/no-calls"

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"hint checks must not call a model (tried {name})")


@dataclass
class HintCheck:
    fixture: str
    move: str
    passed: bool
    gap: str | None
    problems: list[str]
    tags: list[str]


@dataclass
class HintReport:
    """Per-expectation results. ``regressions`` are failures not marked as known gaps."""

    checks: list[HintCheck]
    cpu_ms: float

    @property
    def regressions(self) -> list[HintCheck]:
        return [c for c in self.checks if not c.passed and not c.gap]

    @property
    def known_gaps(self) -> list[HintCheck]:
        return [c for c in self.checks if not c.passed and c.gap]

    @property
    def closed_gaps(self) -> list[HintCheck]:
        """Checks marked as gaps that now pass: remove their ``gap`` marker."""
        return [c for c in self.checks if c.passed and c.gap]

    def summary(self) -> dict[str, Any]:
        return {
            "checks": len(self.checks),
            "passed": sum(c.passed for c in self.checks),
            "regressions": len(self.regressions),
            "known_gaps": len(self.known_gaps),
            "closed_gaps": len(self.closed_gaps),
            "cpu_ms": round(self.cpu_ms, 2),
        }


def check_hints(
    game: Game,
    fixtures: Iterable[Fixture],
    make_harness: Callable[[Iterable[str] | str], GameHarness] | None = None,
) -> HintReport:
    """Check fixture hint expectations against the harness's level-1 candidates. No model is called.

    ``make_harness(concepts)`` builds the harness whose candidates are checked
    (a game adapter may add history-aware tags, as chess does for repetition).
    """
    if make_harness is None:

        def make_harness(concepts: Iterable[str] | str) -> GameHarness:
            return GameHarness(game, driver=_NoCalls(), level=1, concepts=concepts)

    checks: list[HintCheck] = []
    started = time.process_time()
    for fixture in fixtures:
        if not fixture.expect:
            continue
        state = fixture_state(game, fixture)
        for expectation in fixture.expect:
            harness = make_harness(expectation.get("concepts", "all"))
            if not expectation.get("facts", True):
                harness.level = 0
            candidates = {c.name: c for c in harness.candidates(state)}
            checks.append(_check_one(fixture, expectation, candidates))
    return HintReport(checks, (time.process_time() - started) * 1000)


def _check_one(fixture: Fixture, expectation: dict[str, Any], candidates: dict[str, Candidate]) -> HintCheck:
    move = str(expectation["move"])
    gap = expectation.get("gap")
    candidate = candidates.get(move)
    if candidate is None:
        return HintCheck(fixture.id, move, False, gap, [f"{move} is not a legal candidate"], [])
    kinds = {m.kind for m in candidate.motifs}
    tags = [f"{m.kind}: {m.text}" for m in candidate.motifs]
    problems = []
    for item in expectation.get("present", ()):
        if not (item in tags if ": " in item else item in kinds):
            problems.append(f"missing {item!r}")
    for item in expectation.get("absent", ()):
        if item in tags if ": " in item else item in kinds:
            problems.append(f"unexpected {item!r}")
    return HintCheck(fixture.id, move, not problems, gap, problems, tags)


# -- decision runs -------------------------------------------------------------------


@dataclass
class Sample:
    """One player decision on one fixture: the raw record every metric is derived from."""

    fixture: str
    split: str
    repeat: int
    order: int
    move: str | None
    success: bool | None
    verdict: str | None
    outcome: str
    model_selected: bool
    rescued: bool
    attempts: int
    model_attempts: int
    invalid_attempts: int
    first_valid: bool | None
    failure: str | None
    attempt_failures: dict[str, int]
    accounting: dict[str, Any]
    latency_ms: float
    blunder: bool | None = None
    error: str | None = None


@dataclass
class EvaluationRun:
    """A complete, exportable evaluation run with its identities and raw samples."""

    run_id: str
    label: str
    game: str
    evidence: str
    seed: int
    repeats: int
    fixtures: dict[str, str]
    fingerprint: str | None
    config: dict[str, Any]
    model: str | None
    samples: list[Sample]
    started_at: float
    finished_at: float
    version: str = EVAL_VERSION
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scorecard"] = scorecard(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationRun:
        if data.get("version") != EVAL_VERSION:
            raise ValueError(f"evaluation run version {data.get('version')!r} is not {EVAL_VERSION!r}")
        fields = {k: v for k, v in data.items() if k != "scorecard"}
        fields["samples"] = [Sample(**s) for s in data["samples"]]
        return cls(**fields)


def save_run(run: EvaluationRun, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def load_run(path: str | Path) -> EvaluationRun:
    return EvaluationRun.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def evaluate(
    make_player: Callable[[], Any],
    fixtures: Iterable[Fixture],
    *,
    game: Game,
    repeats: int = 1,
    seed: int = 0,
    evidence: str = "scripted",
    label: str = "",
    judge: Evaluator | None = None,
) -> EvaluationRun:
    """Ask a fresh player (``make_player()``) for a move on every decision fixture.

    Samples run in a seeded random order over ``fixtures x repeats``. A turn
    that raises :class:`~prompture.games.records.TurnError` becomes a failed
    sample, never a played move. ``judge(game, state, move)`` optionally marks
    blunders; without it blunders are reported as not assessed.
    """
    if evidence not in EVIDENCE:
        raise ValueError(f"evidence must be one of {EVIDENCE}")
    pool = [f for f in fixtures if f.decides]
    order = [(f, r) for f in pool for r in range(repeats)]
    random.Random(seed).shuffle(order)  # nosec B311 - run order, not crypto
    samples: list[Sample] = []
    started = time.time()
    fingerprint = config = model = None
    for index, (fixture, repeat) in enumerate(order):
        state = fixture_state(game, fixture)
        player = make_player()
        player_game = getattr(player, "game", None)
        if player_game is not None and getattr(player_game, "title", None) != game.title:
            raise ValueError(f"the player plays {player_game.title}, but the fixtures are for {game.title}")
        if fingerprint is None and hasattr(player, "fingerprint"):
            fp = player.fingerprint()
            fingerprint, config, model = fp.digest, fp.config, getattr(player, "model", None)
        start_game = getattr(player, "start_game", None)
        if callable(start_game):
            start_game(f"{fixture.id}#{repeat}")
        samples.append(_sample(game, fixture, repeat, index, state, player, judge))
    return EvaluationRun(
        run_id=uuid.uuid4().hex[:12],
        label=label,
        game=game.title,
        evidence=evidence,
        seed=seed,
        repeats=repeats,
        fixtures={f.id: f.split for f in pool},
        fingerprint=fingerprint,
        config=config or {},
        model=model,
        samples=samples,
        started_at=started,
        finished_at=time.time(),
        notes=[
            "harness seeds fix run order and fallback choices, not provider sampling",
            *([] if evidence == "live" else ["scripted evidence: not a measure of model quality"]),
        ],
    )


def _sample(
    game: Game, fixture: Fixture, repeat: int, order: int, state: Any, player: Any, judge: Evaluator | None
) -> Sample:
    started = time.perf_counter()
    try:
        decision = player.choose(state)
    except TurnError as exc:
        record = exc.record
        attempts = record.attempts if record else []
        return Sample(
            fixture.id,
            fixture.split,
            repeat,
            order,
            None,
            False if fixture.decides else None,
            "turn_failed",
            record.outcome if record else "failed",
            False,
            False,
            len(attempts),
            _model_attempts(attempts),
            _invalid(attempts),
            attempts[0].valid if attempts else None,
            exc.kind.value,
            _failure_counts(attempts),
            record.accounting.to_dict() if record else {},
            (time.perf_counter() - started) * 1000,
            error=str(exc),
        )
    latency = (time.perf_counter() - started) * 1000
    success, verdict = _verdict(game, fixture, state, decision)
    record = decision.record
    attempts = decision.attempts
    return Sample(
        fixture.id,
        fixture.split,
        repeat,
        order,
        decision.name,
        success,
        verdict,
        record.outcome if record else ("assisted" if decision.assistance else "accepted"),
        decision.model_selected,
        decision.rescued,
        len(attempts),
        _model_attempts(attempts),
        _invalid(attempts),
        attempts[0].valid if attempts else None,
        record.failure.kind.value if record and record.failure and record.outcome != "accepted" else None,
        _failure_counts(attempts),
        decision.accounting.to_dict(),
        record.elapsed_ms if record else latency,
        blunder=judge(game, state, decision.move) if judge else None,
    )


def _verdict(game: Game, fixture: Fixture, state: Any, decision: Any) -> tuple[bool | None, str | None]:
    if fixture.evaluator:
        return get_evaluator(game.title, fixture.evaluator)(
            game, state, decision.move
        ), f"evaluator:{fixture.evaluator}"
    if fixture.acceptable:
        return decision.name in fixture.acceptable, "acceptable"
    return None, None


def _model_attempts(attempts: list[Any]) -> int:
    """Attempts that reached the model and came back (a reply, valid or not)."""
    return sum(1 for a in attempts if a.failure is None or a.failure.kind is FailureKind.INVALID_OUTPUT)


def _invalid(attempts: list[Any]) -> int:
    return sum(1 for a in attempts if a.failure is not None and a.failure.kind is FailureKind.INVALID_OUTPUT)


def _failure_counts(attempts: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        if attempt.failure is not None:
            counts[attempt.failure.kind.value] = counts.get(attempt.failure.kind.value, 0) + 1
    return counts


# -- metrics -----------------------------------------------------------------------------


def _metric(numerator: float, denominator: float, scale: float = 1.0) -> dict[str, Any]:
    value = round(numerator / denominator * scale, 6) if denominator else None
    return {"value": value, "numerator": numerator, "denominator": denominator}


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return round(ordered[rank], 3)


def scorecard(run: EvaluationRun | Iterable[Sample], matches: Iterable[Any] = ()) -> dict[str, Any]:
    """Every metric with its counts and denominators, derived only from raw records.

    ``matches`` (``MatchResult`` objects) add clean completions per started
    game. Missing measurements are listed under ``missing`` rather than shown
    as zero.
    """
    samples = list(run.samples if isinstance(run, EvaluationRun) else run)
    judged = [s for s in samples if s.success is not None]
    asked = [s for s in samples if s.attempts]
    attempts = sum(s.model_attempts for s in samples)
    accepted = [s for s in samples if s.model_selected]
    tokens = sum(float(s.accounting.get("total_tokens") or 0) for s in samples)
    cost = sum(float(s.accounting.get("cost") or 0) for s in samples)
    cost_complete = all(s.accounting.get("cost_complete", False) for s in samples if s.attempts)
    latencies = [s.latency_ms for s in asked]
    failures: dict[str, int] = {}
    attempt_failures: dict[str, int] = {}
    for sample in samples:
        if sample.failure:
            failures[sample.failure] = failures.get(sample.failure, 0) + 1
        for kind, count in sample.attempt_failures.items():
            attempt_failures[kind] = attempt_failures.get(kind, 0) + count
    missing = []
    blunders: dict[str, Any] | str
    assessed = [s for s in samples if s.blunder is not None]
    if assessed:
        blunders = _metric(sum(bool(s.blunder) for s in assessed), len(assessed))
    else:
        blunders = "not_assessed"
        missing.append("blunders: no independent judge was supplied")
    if not judged:
        missing.append("success: no fixture had acceptable moves or an evaluator")
    if not cost_complete:
        missing.append("cost: some calls were not fully priced; cost figures are lower bounds")
    card: dict[str, Any] = {
        "version": EVAL_VERSION,
        "samples": len(samples),
        "strength": {
            "unassisted_success": _metric(sum(1 for s in judged if s.success and s.model_selected), len(judged)),
            "success_including_assistance": _metric(sum(1 for s in judged if s.success), len(judged)),
            "blunders": blunders,
        },
        "reliability": {
            "first_answer_validity": _metric(sum(1 for s in asked if s.first_valid), len(asked)),
            "invalid_per_100_model_attempts": _metric(sum(s.invalid_attempts for s in samples), attempts, 100),
            "rescues_per_model_turn": _metric(sum(s.rescued for s in asked), len(asked)),
            "failed_turns": _metric(sum(1 for s in samples if s.outcome in ("failed", "cancelled")), len(samples)),
            "turn_failures": failures,
            "attempt_failures": attempt_failures,
        },
        "efficiency": {
            "tokens_per_accepted_action": _metric(tokens, len(accepted)),
            "cost_per_accepted_action": {**_metric(cost, len(accepted)), "complete": cost_complete},
            "latency_ms_median": round(statistics.median(latencies), 3) if latencies else None,
            "latency_ms_p95": _percentile(latencies, 0.95),
        },
        "missing": missing,
    }
    games = list(matches)
    if games:
        clean = sum(1 for m in games if m.terminal_reason == "game_over" and not any(d.rescued for _, d in m.decisions))
        card["reliability"]["clean_completions_per_started_game"] = _metric(clean, len(games))
        card["reliability"]["terminal_reasons"] = _count(m.terminal_reason for m in games)
    return card


def _count(items: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def summary_text(card: dict[str, Any], *, title: str = "Scorecard") -> str:
    """A readable summary of :func:`scorecard` output (same numbers, with denominators)."""

    def show(metric: Any, unit: str = "") -> str:
        if not isinstance(metric, dict):
            return str(metric).replace("_", " ")
        if metric["value"] is None:
            return f"n/a (0/{metric['denominator']})"
        return f"{metric['value']:.3g}{unit} ({metric['numerator']:g}/{metric['denominator']:g})"

    strength, reliability, efficiency = card["strength"], card["reliability"], card["efficiency"]
    lines = [
        f"{title}: {card['samples']} samples",
        f"  unassisted success      {show(strength['unassisted_success'])}",
        f"  success incl. help      {show(strength['success_including_assistance'])}",
        f"  blunders                {show(strength['blunders'])}",
        f"  first-answer validity   {show(reliability['first_answer_validity'])}",
        f"  invalid / 100 attempts  {show(reliability['invalid_per_100_model_attempts'])}",
        f"  rescues per model turn  {show(reliability['rescues_per_model_turn'])}",
        f"  failed turns            {show(reliability['failed_turns'])} {reliability['turn_failures'] or ''}",
    ]
    if "clean_completions_per_started_game" in reliability:
        lines.append(f"  clean games             {show(reliability['clean_completions_per_started_game'])}")
    cost = efficiency["cost_per_accepted_action"]
    lines += [
        f"  tokens / accepted move  {show(efficiency['tokens_per_accepted_action'])}",
        f"  cost / accepted move    {show(cost)}{'' if cost['complete'] else ' (lower bound)'}",
        f"  latency median / p95    {efficiency['latency_ms_median']} / {efficiency['latency_ms_p95']} ms",
    ]
    lines += [f"  missing: {item}" for item in card["missing"]]
    return "\n".join(lines)


# -- comparisons --------------------------------------------------------------------------


@dataclass(frozen=True)
class Preregistration:
    """Declared before live calls: what is compared, how, and what counts as a result.

    ``factor`` is the one configuration key allowed to differ (``"concepts"``,
    ``"options"``, ``"memory"``, ``"strategy"`` ...). A quality gain needs the
    paired bootstrap interval of ``candidate - baseline`` success to sit above
    zero with at least ``min_pairs`` pairs. ``cost_margin`` is the
    non-inferiority margin on cost per accepted action (``0.1`` = at most 10%
    more); ``reliability_margin`` the allowed rise in invalid replies per 100
    attempts. ``max_cost``/``max_latency_ms`` record the run's declared limits.
    """

    factor: str
    primary_endpoint: str = "unassisted_success"
    min_pairs: int = 30
    confidence: float = 0.95
    resamples: int = 2000
    seed: int = 0
    cost_margin: float | None = None
    reliability_margin: float = 0.0
    max_cost: float | None = None
    max_latency_ms: float | None = None


@dataclass
class Comparison:
    verdict: str
    reasons: list[str]
    pairs: int
    difference: float | None
    interval: tuple[float, float] | None
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    differing_config: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare(baseline: EvaluationRun, candidate: EvaluationRun, prereg: Preregistration | None) -> Comparison:
    """Paired comparison of two runs on the same fixtures and repeats.

    Verdicts: ``improved``, ``regression``, ``inconclusive``, ``not_comparable``
    (different fixtures/splits/game/model, or more than the declared factor
    changed), ``not_evidence`` (scripted runs) and ``not_preregistered``.
    """
    base_card, cand_card = scorecard(baseline), scorecard(candidate)
    differing = sorted(
        k for k in set(baseline.config) | set(candidate.config) if baseline.config.get(k) != candidate.config.get(k)
    )
    reasons: list[str] = []
    pairs = _pairs(baseline, candidate)
    diffs = [c - b for b, c in pairs]
    mean = sum(diffs) / len(diffs) if diffs else None

    def result(verdict: str, interval: tuple[float, float] | None = None) -> Comparison:
        return Comparison(verdict, reasons, len(pairs), mean, interval, base_card, cand_card, differing)

    if (
        baseline.game != candidate.game
        or baseline.fixtures != candidate.fixtures
        or baseline.repeats != candidate.repeats
    ):
        reasons.append("runs used different games, fixtures, splits or repeat counts")
        return result("not_comparable")
    if baseline.model != candidate.model and (prereg is None or prereg.factor != "model"):
        reasons.append(f"models differ ({baseline.model} vs {candidate.model})")
        return result("not_comparable")
    if prereg is not None and set(differing) - {prereg.factor}:
        reasons.append(f"configuration differs beyond {prereg.factor!r}: {sorted(set(differing) - {prereg.factor})}")
        return result("not_comparable")
    if baseline.evidence != "live" or candidate.evidence != "live":
        reasons.append("scripted runs exercise the harness; they are not model-quality evidence")
        return result("not_evidence")
    if prereg is None:
        reasons.append("no preregistration: numbers are descriptive only")
        return result("not_preregistered")
    if len(pairs) < prereg.min_pairs:
        reasons.append(f"{len(pairs)} pairs < preregistered minimum {prereg.min_pairs}")
        return result("inconclusive")
    interval = _bootstrap(diffs, prereg)
    reliability_ok = _reliability_ok(base_card, cand_card, prereg, reasons)
    efficiency_ok = _efficiency_ok(base_card, cand_card, prereg, reasons)
    if interval[1] < 0:
        reasons.append("paired interval lies below zero")
        return result("regression", interval)
    if interval[0] > 0 and reliability_ok and efficiency_ok:
        reasons.append("paired interval lies above zero and the declared gates held")
        return result("improved", interval)
    if interval[0] <= 0:
        reasons.append("paired interval includes zero")
    return result("inconclusive", interval)


def _pairs(baseline: EvaluationRun, candidate: EvaluationRun) -> list[tuple[float, float]]:
    def key(sample: Sample) -> tuple[str, int]:
        return sample.fixture, sample.repeat

    base = {key(s): s for s in baseline.samples if s.success is not None}
    pairs = []
    for sample in candidate.samples:
        other = base.get(key(sample))
        if other is not None and sample.success is not None:
            pairs.append(
                (
                    float(bool(other.success and other.model_selected)),
                    float(bool(sample.success and sample.model_selected)),
                )
            )
    return pairs


def _bootstrap(diffs: list[float], prereg: Preregistration) -> tuple[float, float]:
    rng = random.Random(prereg.seed)  # nosec B311 - resampling, not crypto
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(prereg.resamples))
    tail = (1 - prereg.confidence) / 2
    low = means[int(tail * (len(means) - 1))]
    high = means[int((1 - tail) * (len(means) - 1))]
    return round(low, 6), round(high, 6)


def _reliability_ok(base: dict[str, Any], cand: dict[str, Any], prereg: Preregistration, reasons: list[str]) -> bool:
    b = base["reliability"]["invalid_per_100_model_attempts"]["value"]
    c = cand["reliability"]["invalid_per_100_model_attempts"]["value"]
    if b is None or c is None:
        reasons.append("reliability gate could not be evaluated")
        return False
    if c - b > prereg.reliability_margin:
        reasons.append(f"invalid replies rose by {c - b:.3g}/100 (> {prereg.reliability_margin})")
        return False
    return True


def _efficiency_ok(base: dict[str, Any], cand: dict[str, Any], prereg: Preregistration, reasons: list[str]) -> bool:
    if prereg.cost_margin is None:
        return True
    b, c = base["efficiency"]["cost_per_accepted_action"], cand["efficiency"]["cost_per_accepted_action"]
    if not (b["complete"] and c["complete"]) or not b["value"] or c["value"] is None:
        reasons.append("cost non-inferiority could not be evaluated (incomplete or zero cost)")
        return False
    if c["value"] > b["value"] * (1 + prereg.cost_margin):
        reasons.append(f"cost per action rose beyond the {prereg.cost_margin:.0%} margin")
        return False
    return True


def paired_schedule(
    fixtures: Iterable[Fixture], configs: Iterable[str], *, repeats: int = 1, seed: int = 0
) -> list[tuple[str, str, int]]:
    """``(config, fixture id, repeat)`` jobs: every config gets the same fixtures and repeats, in random order."""
    jobs = [(config, fixture.id, repeat) for fixture in fixtures for repeat in range(repeats) for config in configs]
    random.Random(seed).shuffle(jobs)  # nosec B311 - run order, not crypto
    return jobs


def decisions_sample(
    fixture_id: str, decisions: list[Any], success: bool | None, *, split: str = "dev", order: int = 0
) -> Sample:
    """One :class:`Sample` for a multi-decision task (a puzzle line), so it shares the same metrics."""
    attempts = [a for d in decisions for a in d.attempts]
    accounting: dict[str, Any] = {}
    for decision in decisions:
        for key, value in decision.accounting.to_dict().items():
            if isinstance(value, bool):
                accounting[key] = accounting.get(key, True) and value
            elif isinstance(value, (int, float)):
                accounting[key] = accounting.get(key, 0) + value
    asked = [d for d in decisions if d.attempts]
    return Sample(
        fixture_id,
        split,
        0,
        order,
        " ".join(d.name for d in decisions) or None,
        success,
        "line" if success is not None else None,
        "rescued" if any(d.rescued for d in decisions) else "accepted",
        bool(decisions) and all(d.model_selected for d in decisions),
        any(d.rescued for d in decisions),
        len(attempts),
        _model_attempts(attempts),
        _invalid(attempts),
        asked[0].attempts[0].valid if asked else None,
        None,
        _failure_counts(attempts),
        accounting,
        sum(d.record.elapsed_ms for d in decisions if d.record),
    )
