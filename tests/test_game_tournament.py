"""Side-balanced schedules, batches with recovery, concurrency and budgets, and calibration."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import Any

import pytest

from prompture.drivers.base import Driver
from prompture.games import Connect4, Connect4Engine, EnginePlayer, GameBudget, GameHarness, Othello, RandomPlayer
from prompture.games.control import CancelToken
from prompture.games.tournament import (
    BatchStore,
    Entrant,
    MatchContext,
    calibrate,
    double_round_robin,
    pair_schedule,
    run_batch,
    standings,
)


class FirstListed(Driver):
    """Plays the first listed move; optionally slow; tracks concurrency across instances."""

    supports_messages = True
    model = "scripted/first"
    active = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self, pause: float = 0.0):
        self.pause = pause
        self.calls = 0

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        with FirstListed.lock:
            FirstListed.active += 1
            FirstListed.peak = max(FirstListed.peak, FirstListed.active)
        try:
            time.sleep(self.pause)
            self.calls += 1
            move = re.search(r"^- ([^:\n]+)", prompt, re.M).group(1).strip()
            text = json.dumps({"reason": "r", "plan": "p", "move": move})
            return {
                "text": text,
                "meta": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10, "cost": 0.001},
            }
        finally:
            with FirstListed.lock:
                FirstListed.active -= 1


def random_entrant(name: str, seed: int) -> Entrant:
    return Entrant(
        name, lambda ctx: RandomPlayer(ctx.game, seed=seed + int(ctx.game_id[-4:])), {"kind": "random", "seed": seed}
    )


def model_entrant(name: str, pause: float = 0.0, provider: str | None = None, **kwargs: Any) -> Entrant:
    def make(ctx: MatchContext) -> GameHarness:
        return GameHarness(ctx.game, driver=FirstListed(pause), budget=ctx.budget, cancel=ctx.cancel, **kwargs)

    return Entrant(name, make, {"kind": "model", "pause": pause, **kwargs}, provider=provider)


class TestSchedules:
    def test_double_round_robin_of_four_is_twelve_balanced_games(self):
        schedule = double_round_robin("abcd", ("Red", "Yellow"))
        assert len(schedule) == 12
        pairs = Counter(frozenset(n for _, n in p.sides) for p in schedule)
        assert set(pairs.values()) == {2} and len(pairs) == 6
        first = Counter(p.entrant_for("Red") for p in schedule)
        assert set(first.values()) == {3}  # each entrant takes the first side 3 times
        assert len(double_round_robin("abcd", ("Red", "Yellow"), starts=[None, {"moves": ["4"]}], repeats=2)) == 48
        with pytest.raises(ValueError, match="unique"):
            double_round_robin("aab", ("Red", "Yellow"))

    def test_pair_schedule_alternates_sides(self):
        schedule = pair_schedule("x", "y", ("Black", "White"), games=4)
        assert [p.entrant_for("Black") for p in schedule] == ["x", "y", "x", "y"]


class TestBatches:
    def test_serial_batch_with_standings_and_eligibility(self):
        entrants = [
            random_entrant("r1", 1),
            random_entrant("r2", 2),
            Entrant("engine", lambda ctx: EnginePlayer(ctx.game, Connect4Engine(2))),
        ]
        report = run_batch(Connect4, entrants, double_round_robin(["r1", "r2", "engine"], ("Red", "Yellow")))
        assert len(report.outcomes) == 6 and all(o.completed for o in report.outcomes)
        standings = report.standings()
        assert sum(row["games"] for row in standings.values()) == 12
        rows = report.export()
        engine_rows = [r for r in rows if "engine" in r["entrants"].values()]
        assert engine_rows and all(not r["eligibility"]["eligible"] for r in engine_rows)
        assert all(r["final_state"]["moves"] == r["moves"] for r in rows)

    def test_restart_skips_finished_games_and_never_duplicates_results(self, tmp_path):
        store = tmp_path / "batch.jsonl"
        entrants = [random_entrant(n, i) for i, n in enumerate("abcd")]
        schedule = double_round_robin("abcd", ("Black", "White"))
        token = CancelToken()
        seen = []

        def stop_after_five(outcome):
            seen.append(outcome.game_id)
            if len(seen) == 5:
                token.cancel("operator stop")

        first = run_batch(Othello, entrants, schedule, store=store, cancel=token, on_outcome=stop_after_five)
        assert first.cancelled and len(first.outcomes) == 5 and len(first.not_started) == 7
        second = run_batch(Othello, entrants, schedule, store=store)
        assert len(second.skipped) == 5 and len(second.outcomes) == 7
        results = [r for r in BatchStore(store).records() if r["type"] == "result"]
        assert len(results) == 12 and len({r["game_id"] for r in results}) == 12
        stored = BatchStore(store).outcomes()
        assert len(stored) == 12 and sum(row["games"] for row in standings(stored).values()) == 24

    def test_interrupted_game_is_reconciled_before_replay(self, tmp_path):
        store = BatchStore(tmp_path / "batch.jsonl")
        entrants = [random_entrant("a", 1), random_entrant("b", 2)]
        schedule = pair_schedule("a", "b", ("Red", "Yellow"), games=2)
        report = run_batch(Connect4, entrants, schedule, store=store, cancel=_cancelled())
        assert report.not_started and not report.outcomes
        game_id = report.not_started[0]
        store.append(
            {"type": "started", "schedule_id": report.schedule_id, "game_id": game_id, "attempt": 1}
        )  # crash mid-game
        resumed = run_batch(Connect4, entrants, schedule, store=store)
        assert resumed.interrupted == [game_id]
        kinds = [(r["type"], r.get("attempt")) for r in store.records() if r.get("game_id") == game_id]
        assert kinds == [("started", 1), ("interrupted", 1), ("started", 2), ("result", 2)]
        interrupted = next(r for r in store.records() if r["type"] == "interrupted")
        assert interrupted["usage"] == "unknown"

    def test_a_store_belongs_to_one_schedule(self, tmp_path):
        store = tmp_path / "batch.jsonl"
        entrants = [random_entrant("a", 1), random_entrant("b", 2)]
        run_batch(Connect4, entrants, pair_schedule("a", "b", ("Red", "Yellow"), games=2), store=store)
        with pytest.raises(ValueError, match="holds schedule"):
            run_batch(Connect4, entrants, pair_schedule("a", "b", ("Red", "Yellow"), games=4), store=store)

    def test_bounded_concurrency_and_provider_admission(self):
        FirstListed.peak = 0
        entrants = [model_entrant(n, pause=0.01, provider="p" if n == "a" else "q") for n in "abcd"]
        schedule = double_round_robin("abcd", ("Red", "Yellow"))
        report = run_batch(Connect4, entrants, schedule, max_concurrency=3, max_moves=6)
        assert len(report.outcomes) == 12 and FirstListed.peak <= 3
        FirstListed.peak = 0
        both_p = [model_entrant(n, pause=0.01, provider="p") for n in "ab"]
        run_batch(
            Connect4,
            both_p,
            pair_schedule("a", "b", ("Red", "Yellow"), games=4),
            max_concurrency=4,
            provider_limits={"p": 1},
            max_moves=6,
        )
        assert FirstListed.peak == 1
        with pytest.raises(ValueError, match="at least 1"):
            run_batch(Connect4, both_p, [], provider_limits={"p": 0})

    def test_shared_budget_stops_spending_and_marks_games_replayable(self):
        budget = GameBudget(max_calls=15)
        entrants = [model_entrant("a"), model_entrant("b")]
        report = run_batch(
            Connect4, entrants, pair_schedule("a", "b", ("Red", "Yellow"), games=4), budget=budget, max_concurrency=2
        )
        assert budget.calls <= 15
        reasons = Counter(o.terminal_reason for o in report.outcomes)
        assert reasons["budget_exhausted"] >= 1 and not any(
            o.completed for o in report.outcomes if o.terminal_reason == "budget_exhausted"
        )
        spent = sum(acc["calls"] for o in report.outcomes for acc in o.accounting.values())
        assert spent == budget.calls  # every call accounted once, none invented
        assert report.budget["calls"] == budget.calls

    def test_cancellation_stops_admission(self):
        report = run_batch(
            Connect4,
            [random_entrant("a", 1), random_entrant("b", 2)],
            pair_schedule("a", "b", ("Red", "Yellow")),
            cancel=_cancelled(),
        )
        assert report.cancelled and not report.outcomes and len(report.not_started) == 2

    def test_a_failing_entrant_does_not_decide_other_games(self):
        class Broken:
            def choose(self, state):
                raise RuntimeError("engine crashed")

        entrants = [random_entrant("a", 1), random_entrant("b", 2), Entrant("broken", lambda ctx: Broken())]
        report = run_batch(Connect4, entrants, double_round_robin(["a", "b", "broken"], ("Red", "Yellow")))
        errors = [o for o in report.outcomes if o.terminal_reason == "error"]
        assert len(errors) == 4 and all(o.winner is None and "engine crashed" in o.error for o in errors)
        assert all(o.terminal_reason == "game_over" for o in report.outcomes if "broken" not in o.entrants.values())
        assert "broken" not in report.standings()
        assert all(row["eligibility"]["failed"] for row in report.export() if row["terminal_reason"] == "error")

    def test_forfeits_and_assistance_are_exported(self):
        class Bad(FirstListed):
            def generate(self, prompt, options):
                return {"text": json.dumps({"reason": "", "plan": "", "move": "9"}), "meta": {"total_tokens": 1}}

        strict = Entrant(
            "strict", lambda ctx: GameHarness(ctx.game, driver=Bad(), max_retries=0, failure_policy="forfeit")
        )
        report = run_batch(
            Connect4, [strict, random_entrant("r", 1)], pair_schedule("strict", "r", ("Red", "Yellow"), games=2)
        )
        assert all(o.terminal_reason == "forfeit" and o.winner == "r" for o in report.outcomes)
        rows = report.export()
        assert all(row["eligibility"]["forfeit"] and not row["eligibility"]["failed"] for row in rows)
        # The forfeit itself is a decided result; the random opponent's moves are what make a row assisted.
        assert all(
            row["eligibility"]["reasons"] == ["assisted moves (rescues or engine help)"] for row in rows if row["moves"]
        )

    def test_changed_configuration_is_flagged(self):
        report = run_batch(
            Connect4,
            [model_entrant("a"), random_entrant("r", 1)],
            pair_schedule("a", "r", ("Red", "Yellow"), games=1),
            max_moves=2,
        )
        row = report.export()[0]
        assert report.entrants["a"] and not row["eligibility"]["incompatible"]
        declared = {"a": "0" * 16, "r": None}
        assert report.outcomes[0].eligibility(declared)["incompatible"] == ["Red"]


class TestCalibration:
    def profiles(self) -> list[Entrant]:
        def engine(depth: int) -> Entrant:
            return Entrant(
                f"engine-d{depth}",
                lambda ctx: EnginePlayer(ctx.game, Connect4Engine(depth)),
                {"engine": "Connect4Engine", "depth": depth},
            )

        return [random_entrant("random", 9), engine(1), engine(3), engine(5)]

    def test_finds_an_informative_profile_or_reports_why_not(self):
        candidate = Entrant("cand", lambda ctx: EnginePlayer(ctx.game, Connect4Engine(3)), {"depth": 3})
        result = calibrate(candidate, self.profiles(), Connect4, games_per_step=4, max_games=16)
        assert result.status in ("resolved", "unresolved") and result.games <= 16
        assert result.steps and all(len(step.interval) == 2 for step in result.steps)
        if result.status == "resolved":
            assert result.steps[-1].games >= 8  # minimum evidence at the reported profile
        assert "not official ratings" in result.note

    def test_ceilings_leave_it_unresolved(self):
        candidate = random_entrant("cand", 3)
        result = calibrate(candidate, self.profiles(), Connect4, games_per_step=4, max_games=3)
        assert (result.status, result.games) == ("unresolved", 0) and "ceiling" in result.reason
        spent = GameBudget(max_calls=1)
        spent.settle(spent.reserve(0), {}, "estimated")
        assert "budget" in calibrate(candidate, self.profiles(), Connect4, budget=spent).reason


def test_generic_batch_without_python_chess():
    code = (
        "import sys; sys.modules['chess'] = None\n"
        "from prompture.games import Othello, RandomPlayer\n"
        "from prompture.games.tournament import Entrant, double_round_robin, run_batch\n"
        "es = [Entrant(n, lambda ctx, i=i: RandomPlayer(ctx.game, seed=i)) for i, n in enumerate('abc')]\n"
        "r = run_batch(Othello, es, double_round_robin('abc', Othello.sides), max_concurrency=2)\n"
        "assert len(r.outcomes) == 6 and all(o.completed for o in r.outcomes)\n"
        "assert 'chess' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def _cancelled() -> CancelToken:
    token = CancelToken()
    token.cancel("stopped before start")
    return token


def test_chess_batch_uses_the_games_side_names_and_starts():
    pytest.importorskip("chess")
    from prompture.games.chess import ChessGame

    start = {"initial": None, "moves": ["e2e4", "e7e5"]}
    schedule = double_round_robin("ab", ChessGame.sides, starts=[None, start])
    report = run_batch(ChessGame, [random_entrant("a", 1), random_entrant("b", 2)], schedule, max_moves=6)
    assert len(report.outcomes) == 4 and {o.terminal_reason for o in report.outcomes} == {"move_cap"}
    assert all(set(o.entrants) == {"White", "Black"} for o in report.outcomes)
    started = [o for o in report.outcomes if o.pairing.start]
    assert all(o.final_state["moves"][:2] == ["e2e4", "e7e5"] for o in started)
    assert report.standings() == {}  # capped games are not decided results
