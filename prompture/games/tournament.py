"""Batches of games: side-balanced schedules, bounded concurrency, checkpoints and calibration.

* :class:`Entrant` — a frozen player configuration: a name, a factory that
  builds a fresh player for one game, and the configuration it declares.
* :func:`pair_schedule` / :func:`double_round_robin` — side-balanced pairings.
  Side names and starting positions come from the game (``game.sides``) or the
  caller, never from chess constants. Four entrants make 12 games per
  repetition in a double round robin.
* :func:`run_batch` — plays a schedule serially or with bounded concurrency,
  with per-provider admission limits, a shared
  :class:`~prompture.games.control.GameBudget` and a
  :class:`~prompture.games.control.CancelToken` that stops admission. Results
  go to a :class:`BatchStore` (a JSONL file the caller chooses). Game IDs are
  derived from the schedule, so a restarted batch skips finished games instead
  of playing them twice; a game that was running when the process stopped is
  first recorded as interrupted (its spend unknown) before it is replayed.
  This prevents duplicate *results*; it cannot prevent a provider from having
  billed the interrupted calls.
* :func:`calibrate` — an opt-in, exploratory search over ordered opponent
  profiles for an informative score range (35-65% by default), bounded by
  games, time and budget. It never replaces a fixed confirmatory schedule, and
  an engine's nominal settings are not a human rating.

Results export assisted, failed and incompatible outcomes with eligibility
flags for a host's leaderboard. No rating system is implemented here.
Final outcomes are ``game_over``, ``forfeit``, ``move_cap``, ``stopped`` and
``error``; ``cancelled`` and ``budget_exhausted`` games are replayed by a
resumed batch.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Game, MoveDecision
from .control import CancelToken, GameBudget
from .match import MatchResult, Player, play_match
from .records import Accounting, BudgetExhausted

BATCH_VERSION = "1"
FINAL_REASONS = ("game_over", "forfeit", "move_cap", "stopped", "error")


@dataclass(frozen=True)
class MatchContext:
    """What an entrant's factory gets to build a fresh player for one game."""

    game: Game
    side: str
    game_id: str
    budget: GameBudget | None
    cancel: CancelToken


@dataclass(frozen=True)
class Entrant:
    """A frozen player configuration. ``config`` is what it declares about itself (no secrets).

    ``provider`` groups entrants for admission limits. ``fingerprint`` is
    filled from the first player built (``player.fingerprint()``) when not given.
    """

    name: str
    make: Callable[[MatchContext], Player] = field(compare=False, repr=False)
    config: dict[str, Any] = field(default_factory=dict, compare=False)
    provider: str | None = None
    fingerprint: str | None = None


@dataclass(frozen=True)
class Pairing:
    """One scheduled game: who plays which side, from which start."""

    game_no: int
    round: int
    sides: tuple[tuple[str, str], ...]  # (side, entrant name)
    start: Any = None
    repeat: int = 0

    def entrant_for(self, side: str) -> str:
        return dict(self.sides)[side]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_no": self.game_no,
            "round": self.round,
            "sides": [list(s) for s in self.sides],
            "start": self.start,
            "repeat": self.repeat,
        }


def game_sides(game: Game, sides: Iterable[str] | None = None) -> tuple[str, str]:
    found = tuple(sides) if sides is not None else tuple(getattr(game, "sides", ()))
    if len(found) != 2:
        raise ValueError(f"{game.title}: pass the two side names (the game does not declare `sides`)")
    return found[0], found[1]


def pair_schedule(
    a: str, b: str, sides: tuple[str, str], *, games: int = 2, starts: Iterable[Any] = (None,), first_no: int = 0
) -> list[Pairing]:
    """``games`` per start between ``a`` and ``b``, alternating sides (balanced when ``games`` is even)."""
    pairings = []
    number = first_no
    for start in starts:
        for index in range(games):
            first, second = (a, b) if index % 2 == 0 else (b, a)
            pairings.append(Pairing(number, index // 2, ((sides[0], first), (sides[1], second)), start, index // 2))
            number += 1
    return pairings


def double_round_robin(
    names: Iterable[str], sides: tuple[str, str], *, starts: Iterable[Any] = (None,), repeats: int = 1
) -> list[Pairing]:
    """Every pair meets twice per start and repeat, once on each side."""
    entrants = list(names)
    if len(set(entrants)) != len(entrants):
        raise ValueError("entrant names must be unique")
    pairings: list[Pairing] = []
    starts = list(starts)
    for repeat in range(repeats):
        round_no = 0
        for i, a in enumerate(entrants):
            for b in entrants[i + 1 :]:
                for start in starts:
                    for first, second in ((a, b), (b, a)):
                        pairings.append(
                            Pairing(len(pairings), round_no, ((sides[0], first), (sides[1], second)), start, repeat)
                        )
                round_no += 1
    return pairings


def schedule_id(game: Game, pairings: list[Pairing], entrants: dict[str, Entrant]) -> str:
    """A stable identity for a schedule over frozen entrant configurations."""
    blob = json.dumps(
        {
            "game": game.title,
            "pairings": [p.to_dict() for p in pairings],
            "entrants": {n: {"config": e.config, "fingerprint": e.fingerprint} for n, e in sorted(entrants.items())},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# -- storage ----------------------------------------------------------------------------


class BatchStore:
    """A JSONL record of a batch: its schedule, started games and results. The caller picks the path."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn final line from an interrupted write
        return out

    def outcomes(self) -> list[GameOutcome]:
        """The latest final result for every game in the store, across all runs."""
        latest: dict[str, GameOutcome] = {}
        for record in self.records():
            if record.get("type") == "result" and record.get("terminal_reason") in FINAL_REASONS:
                data = {k: v for k, v in record.items() if k not in ("type", "schedule_id")}
                latest[record["game_id"]] = GameOutcome.from_dict(data)
        return sorted(latest.values(), key=lambda o: o.pairing.game_no)


# -- running ------------------------------------------------------------------------------


@dataclass
class GameOutcome:
    """One game's exportable result. ``entrants`` maps side -> entrant name."""

    game_id: str
    pairing: Pairing
    attempt: int
    terminal_reason: str
    winner_side: str | None
    entrants: dict[str, str]
    moves: list[str]
    rescues: dict[str, int]
    model_moves: dict[str, int]
    assisted: dict[str, bool]
    accounting: dict[str, dict[str, Any]]
    fingerprints: dict[str, str | None]
    forfeited_by: str | None = None
    error: str | None = None
    final_state: Any = None
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def winner(self) -> str | None:
        return self.entrants.get(self.winner_side) if self.winner_side else None

    @property
    def completed(self) -> bool:
        return self.terminal_reason in FINAL_REASONS

    def eligibility(self, declared: dict[str, str | None] | None = None) -> dict[str, Any]:
        """Flags a host can use to decide rating eligibility (this module rates nothing)."""
        incompatible = sorted(
            side
            for side, fp in self.fingerprints.items()
            if declared and declared.get(self.entrants[side]) not in (None, fp)
        )
        reasons = []
        if self.terminal_reason not in ("game_over", "forfeit"):
            reasons.append(f"terminal reason {self.terminal_reason}")
        if any(self.assisted.values()):
            reasons.append("assisted moves (rescues or engine help)")
        if incompatible:
            reasons.append(f"configuration changed for {incompatible}")
        return {
            "eligible": not reasons,
            "reasons": reasons,
            "assisted": dict(self.assisted),
            "forfeit": self.terminal_reason == "forfeit",
            "failed": self.terminal_reason in ("stopped", "error"),
            "incompatible": incompatible,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "pairing": self.pairing.to_dict(),
            "attempt": self.attempt,
            "terminal_reason": self.terminal_reason,
            "winner_side": self.winner_side,
            "winner": self.winner,
            "entrants": dict(self.entrants),
            "moves": list(self.moves),
            "rescues": dict(self.rescues),
            "model_moves": dict(self.model_moves),
            "assisted": dict(self.assisted),
            "accounting": self.accounting,
            "fingerprints": dict(self.fingerprints),
            "forfeited_by": self.forfeited_by,
            "error": self.error,
            "final_state": self.final_state,
            "elapsed_s": round(self.elapsed_s, 3),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameOutcome:
        p = data["pairing"]
        pairing = Pairing(
            p["game_no"], p["round"], tuple((s, e) for s, e in p["sides"]), p.get("start"), p.get("repeat", 0)
        )
        fields = {k: v for k, v in data.items() if k not in ("pairing", "winner")}
        return cls(pairing=pairing, **fields)


@dataclass
class BatchReport:
    """Everything a batch produced, including what it could not finish."""

    schedule_id: str
    design: str
    outcomes: list[GameOutcome]
    skipped: list[str]
    not_started: list[str]
    interrupted: list[str]
    budget: dict[str, Any] | None
    cancelled: bool
    entrants: dict[str, str | None]

    def standings(self) -> dict[str, dict[str, float]]:
        """:func:`standings` over this run's outcomes (use ``BatchStore.outcomes()`` for every run)."""
        return standings(self.outcomes)

    def export(self) -> list[dict[str, Any]]:
        """Result rows for a host leaderboard, each with eligibility flags."""
        return [{**o.to_dict(), "eligibility": o.eligibility(self.entrants)} for o in self.outcomes]


def standings(outcomes: Iterable[GameOutcome]) -> dict[str, dict[str, float]]:
    """Wins/draws/losses per entrant over decided games only (stopped, failed, capped and cancelled games are excluded)."""
    table: dict[str, dict[str, float]] = {}
    for outcome in outcomes:
        if outcome.terminal_reason not in ("game_over", "forfeit"):
            continue
        for side, name in outcome.entrants.items():
            row = table.setdefault(name, {"games": 0, "wins": 0, "draws": 0, "losses": 0, "score": 0.0})
            row["games"] += 1
            if outcome.winner_side is None:
                row["draws"] += 1
                row["score"] += 0.5
            elif outcome.winner_side == side:
                row["wins"] += 1
                row["score"] += 1
            else:
                row["losses"] += 1
    return table


def run_batch(
    game_factory: Callable[[], Game],
    entrants: Iterable[Entrant],
    schedule: list[Pairing],
    *,
    store: BatchStore | str | Path | None = None,
    max_concurrency: int = 1,
    provider_limits: dict[str, int] | None = None,
    budget: GameBudget | None = None,
    cancel: CancelToken | None = None,
    max_moves: int = 400,
    design: str = "confirmatory",
    on_outcome: Callable[[GameOutcome], None] | None = None,
) -> BatchReport:
    """Play ``schedule``; resume from ``store`` when it already holds results for this schedule.

    ``max_concurrency`` above 1 runs games on threads; each game gets fresh
    players, so memory never crosses games. A game that raises a non-turn
    error is recorded as ``error`` without a winner; other games continue.
    """
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if any(limit < 1 for limit in (provider_limits or {}).values()):
        raise ValueError("provider limits must be at least 1")
    game = game_factory()
    roster = {e.name: e for e in entrants}
    for pairing in schedule:
        for _, name in pairing.sides:
            if name not in roster:
                raise ValueError(f"schedule names unknown entrant {name!r}")
    sid = schedule_id(game, schedule, roster)
    store = BatchStore(store) if isinstance(store, (str, Path)) else store
    cancel = cancel or CancelToken()
    done, attempts, interrupted = _resume(store, sid, game, schedule, design)
    skipped = sorted(done)
    pending = [p for p in schedule if _game_id(sid, p) not in done]
    fingerprints: dict[str, str | None] = {n: e.fingerprint for n, e in roster.items()}
    outcomes: list[GameOutcome] = []
    limits = {k: threading.Semaphore(v) for k, v in (provider_limits or {}).items()}
    lock = threading.Lock()

    def admit(pairing: Pairing) -> list[threading.Semaphore] | None:
        held = []
        providers = {roster[name].provider for _, name in pairing.sides} - {None}
        for provider in sorted(providers):  # one slot per provider per game
            sem = limits.get(provider)
            if sem is not None:
                if not sem.acquire(blocking=False):
                    for h in held:
                        h.release()
                    return None
                held.append(sem)
        return held

    def play_one(pairing: Pairing, held: list[threading.Semaphore]) -> GameOutcome:
        try:
            game_id = _game_id(sid, pairing)
            attempt = attempts.get(game_id, 0) + 1
            if store is not None:
                store.append(
                    {"type": "started", "schedule_id": sid, "game_id": game_id, "attempt": attempt, "time": time.time()}
                )
            outcome = _play(
                game_factory, roster, pairing, game_id, attempt, budget, cancel.child(), max_moves, fingerprints, lock
            )
            if store is not None:
                store.append({"type": "result", "schedule_id": sid, **outcome.to_dict()})
            if on_outcome is not None:
                on_outcome(outcome)
            return outcome
        finally:
            for sem in held:
                sem.release()

    queue = list(pending)
    not_started: list[str] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        running: set[Future[GameOutcome]] = set()
        while queue or running:
            stop = cancel.cancelled or (budget is not None and budget.check() is not None)
            if stop:
                not_started += [_game_id(sid, p) for p in queue]
                queue.clear()
            started_any = False
            while queue and len(running) < max_concurrency:
                index = next((i for i, p in enumerate(queue) if (held := admit(p)) is not None), None)
                if index is None:
                    break
                running.add(pool.submit(play_one, queue.pop(index), held))
                started_any = True
            if not running:
                if queue and not started_any:
                    time.sleep(0.01)  # waiting for provider admission
                continue
            finished, running = wait(running, timeout=0.05, return_when=FIRST_COMPLETED)
            for future in finished:
                outcomes.append(future.result())
    outcomes.sort(key=lambda o: o.pairing.game_no)
    return BatchReport(
        sid,
        design,
        outcomes,
        skipped,
        not_started,
        interrupted,
        budget.snapshot() if budget is not None else None,
        cancel.cancelled,
        fingerprints,
    )


def _game_id(sid: str, pairing: Pairing) -> str:
    return f"{sid}-{pairing.game_no:04d}"


def _resume(
    store: BatchStore | None, sid: str, game: Game, schedule: list[Pairing], design: str
) -> tuple[set[str], dict[str, int], list[str]]:
    """Finished game ids, attempt counts and newly reconciled interrupted games."""
    if store is None:
        return set(), {}, []
    records = store.records()
    header = next((r for r in records if r.get("type") == "schedule"), None)
    if header is None:
        store.append(
            {
                "type": "schedule",
                "batch_version": BATCH_VERSION,
                "schedule_id": sid,
                "game": game.title,
                "design": design,
                "games": len(schedule),
                "time": time.time(),
            }
        )
        return set(), {}, []
    if header.get("schedule_id") != sid:
        raise ValueError(f"{store.path} holds schedule {header.get('schedule_id')}, not {sid}; use a new store")
    done: set[str] = set()
    attempts: dict[str, int] = {}
    open_attempts: dict[str, int] = {}
    for record in records:
        game_id = record.get("game_id")
        if record.get("type") == "started":
            attempts[game_id] = max(attempts.get(game_id, 0), record["attempt"])
            open_attempts[game_id] = record["attempt"]
        elif record.get("type") in ("result", "interrupted"):
            open_attempts.pop(game_id, None)
            if record.get("type") == "result" and record.get("terminal_reason") in FINAL_REASONS:
                done.add(game_id)
    interrupted = []
    for game_id, attempt in sorted(open_attempts.items()):
        # Reconcile before sending anything new: this attempt's calls may have been billed.
        store.append(
            {
                "type": "interrupted",
                "schedule_id": sid,
                "game_id": game_id,
                "attempt": attempt,
                "usage": "unknown",
                "note": "process stopped mid-game; provider charges for this attempt are unknown",
                "time": time.time(),
            }
        )
        interrupted.append(game_id)
    return done, attempts, interrupted


def _play(
    game_factory: Callable[[], Game],
    roster: dict[str, Entrant],
    pairing: Pairing,
    game_id: str,
    attempt: int,
    budget: GameBudget | None,
    cancel: CancelToken,
    max_moves: int,
    fingerprints: dict[str, str | None],
    lock: threading.Lock,
) -> GameOutcome:
    game = game_factory()
    started = time.monotonic()
    sides = dict(pairing.sides)
    played: list[tuple[str, MoveDecision]] = []
    players: dict[str, Any] = {}
    try:
        players = {
            side: _Tap(roster[name].make(MatchContext(game, side, game_id, budget, cancel)), side, played)
            for side, name in sides.items()
        }
        for side, player in players.items():
            fp = getattr(player, "fingerprint", None)
            if callable(fp):
                with lock:
                    fingerprints.setdefault(sides[side], None)
                    if fingerprints[sides[side]] is None:
                        fingerprints[sides[side]] = fp().digest
        state = game.load_state(pairing.start) if pairing.start is not None else None
        result = play_match(game, players, state=state, max_moves=max_moves, game_id=game_id, cancel=cancel)
    except Exception as exc:  # an entrant or engine bug ends this game only
        accounts = {side: Accounting() for side in sides}
        for side, decision in played:
            accounts[side].merge(decision.accounting)
        return GameOutcome(
            game_id,
            pairing,
            attempt,
            "error",
            None,
            sides,
            [d.name for _, d in played],
            {side: sum(d.rescued for s, d in played if s == side) for side in sides},
            {side: sum(d.model_selected for s, d in played if s == side) for side in sides},
            {side: any(d.rescued or d.assistance is not None for s, d in played if s == side) for side in sides},
            {side: a.to_dict() for side, a in accounts.items()},
            {side: _fp(players[side]) if side in players else None for side in sides},
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
            notes=["accounting covers completed turns only; the failing turn's usage is unknown"],
        )
    return _outcome(game, pairing, game_id, attempt, sides, players, result, time.monotonic() - started)


class _Tap:
    """Records each decision as it is made, so a game that crashes still reports what it spent."""

    def __init__(self, player: Any, side: str, played: list[tuple[str, MoveDecision]]) -> None:
        self._player = player
        self._side = side
        self._played = played

    def choose(self, state: Any) -> MoveDecision:
        decision = self._player.choose(state)
        self._played.append((self._side, decision))
        return decision

    def __getattr__(self, name: str) -> Any:
        return getattr(self._player, name)


def _outcome(
    game: Game,
    pairing: Pairing,
    game_id: str,
    attempt: int,
    sides: dict[str, str],
    players: dict[str, Any],
    result: MatchResult,
    elapsed: float,
) -> GameOutcome:
    rescues = {side: 0 for side in sides}
    model_moves = {side: 0 for side in sides}
    assisted = {side: False for side in sides}
    accounts = {side: Accounting() for side in sides}
    decisions: list[tuple[str, MoveDecision]] = result.decisions
    for side, decision in decisions:
        rescues[side] += decision.rescued
        model_moves[side] += decision.model_selected
        assisted[side] = assisted[side] or decision.rescued or decision.assistance is not None
        accounts[side].merge(decision.accounting)
    if result.error is not None and getattr(result.error, "record", None) is not None:
        side = game.player(result.state)
        accounts[side].merge(result.error.record.accounting)
    for side, player in players.items():
        if getattr(player, "level", 0) >= 2:
            assisted[side] = True  # engine-filtered menus
    dump = getattr(game, "dump_state", None)
    final_state = dump(result.state) if callable(dump) else None
    winner_side = result.winner if result.terminal_reason in ("game_over", "forfeit") else None
    terminal = result.terminal_reason
    if isinstance(result.error, BudgetExhausted):
        terminal = "budget_exhausted"  # not final: a later run with more budget replays it
    return GameOutcome(
        game_id,
        pairing,
        attempt,
        terminal,
        winner_side,
        sides,
        list(result.moves),
        rescues,
        model_moves,
        assisted,
        {side: a.to_dict() for side, a in accounts.items()},
        {side: _fp(players[side]) for side in sides},
        forfeited_by=result.forfeited_by,
        error=str(result.error) if result.error else None,
        final_state=final_state,
        elapsed_s=elapsed,
    )


def _fp(player: Any) -> str | None:
    fp = getattr(player, "fingerprint", None)
    return fp().digest if callable(fp) else None


# -- calibration --------------------------------------------------------------------------


@dataclass
class CalibrationStep:
    profile: str
    games: int
    score: float
    interval: tuple[float, float]


@dataclass
class Calibration:
    """An exploratory result: which profile gave an informative score, or why it stayed unresolved."""

    status: str  # "resolved" or "unresolved"
    profile: str | None
    steps: list[CalibrationStep]
    reason: str
    games: int
    target: tuple[float, float]
    note: str = "exploratory: profile scores are not official ratings; confirm with a fixed schedule"


def wilson(score: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a score fraction over ``games`` (draws counted as half points)."""
    if games == 0:
        return 0.0, 1.0
    centre = (score + z * z / (2 * games)) / (1 + z * z / games)
    spread = z * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games)) / (1 + z * z / games)
    return round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4)


def calibrate(
    candidate: Entrant,
    profiles: list[Entrant],
    game_factory: Callable[[], Game],
    *,
    sides: tuple[str, str] | None = None,
    target: tuple[float, float] = (0.35, 0.65),
    games_per_step: int = 4,
    min_games: int = 8,
    max_games: int = 40,
    max_seconds: float | None = None,
    budget: GameBudget | None = None,
    cancel: CancelToken | None = None,
    max_moves: int = 400,
) -> Calibration:
    """Find an opponent profile (ordered weakest first) against which ``candidate`` scores inside ``target``.

    Plays ``games_per_step`` side-balanced games per step, moving to a stronger
    profile above the range and a weaker one below it. A profile is only
    reported once its score is inside the range over at least ``min_games``
    decided games (more steps are played there until then). The search stops
    as ``unresolved`` when it would revisit a profile, runs off either end, or
    hits the game/time/budget ceiling.
    """
    if not profiles:
        raise ValueError("calibration needs at least one opponent profile")
    game = game_factory()
    side_names = game_sides(game, sides)
    started = time.monotonic()
    index = len(profiles) // 2
    seen: dict[int, CalibrationStep] = {}
    steps: list[CalibrationStep] = []
    tally: dict[int, list[float]] = {}  # profile index -> [points, decided games]
    played = 0
    while True:
        if played + games_per_step > max_games:
            return Calibration("unresolved", None, steps, f"game ceiling {max_games} reached", played, target)
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            return Calibration("unresolved", None, steps, f"time ceiling {max_seconds}s reached", played, target)
        if budget is not None and budget.check() is not None:
            return Calibration("unresolved", None, steps, f"budget: {budget.check()}", played, target)
        opponent = profiles[index]
        schedule = pair_schedule(candidate.name, opponent.name, side_names, games=games_per_step)
        report = run_batch(
            game_factory,
            [candidate, opponent],
            schedule,
            budget=budget,
            cancel=cancel,
            max_moves=max_moves,
            design="exploratory",
        )
        decided = [o for o in report.outcomes if o.terminal_reason in ("game_over", "forfeit")]
        played += len(report.outcomes)
        points = sum(1.0 if o.winner == candidate.name else 0.5 if o.winner is None else 0.0 for o in decided)
        total = tally.setdefault(index, [0.0, 0.0])
        total[0] += points
        total[1] += len(decided)
        games = int(total[1])
        score = total[0] / games if games else 0.0
        step = CalibrationStep(opponent.name, games, round(score, 4), wilson(score, games))
        steps.append(step)
        if report.cancelled:
            return Calibration("unresolved", None, steps, "cancelled", played, target)
        if not decided:
            return Calibration("unresolved", None, steps, "no decided games at this profile", played, target)
        if target[0] <= score <= target[1]:
            if games >= min_games:
                return Calibration(
                    "resolved",
                    opponent.name,
                    steps,
                    f"score {score:.2f} over {games} games inside {target}",
                    played,
                    target,
                )
            continue  # informative so far: gather the minimum evidence at this profile
        seen[index] = step
        nxt = index + 1 if score > target[1] else index - 1
        if nxt < 0 or nxt >= len(profiles):
            edge = "strongest" if nxt >= len(profiles) else "weakest"
            return Calibration(
                "unresolved", None, steps, f"score {score:.2f} even against the {edge} profile", played, target
            )
        if nxt in seen:
            return Calibration(
                "unresolved", None, steps, "scores straddle the range between adjacent profiles", played, target
            )
        index = nxt
