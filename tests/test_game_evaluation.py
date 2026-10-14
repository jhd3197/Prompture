"""Reproducible evaluation: fixtures, hint checks, scorecards and paired comparisons."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from prompture.drivers.base import Driver
from prompture.games import Connect4, GameHarness, Othello, RandomPlayer, play_match
from prompture.games.evaluation import (
    EvaluationRun,
    Fixture,
    FixtureError,
    Preregistration,
    Sample,
    check_hints,
    compare,
    evaluate,
    fixture_state,
    load_fixtures,
    load_run,
    paired_schedule,
    save_run,
    scorecard,
    select,
    summary_text,
)
from prompture.games.fixtures import builtin_fixtures


class FirstListed(Driver):
    """Answers with the first move on the prompt's menu (or a fixed bad move first)."""

    supports_messages = True
    model = "scripted/first-listed"

    def __init__(self, bad_first: int = 0, meta: dict[str, Any] | None = None):
        self.bad_first = bad_first
        self.calls = 0
        self.meta = (
            meta
            if meta is not None
            else {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25, "cost": 0.001}
        )

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        move = re.search(r"^- ([^:\n]+)", prompt, re.M).group(1).strip()
        if self.calls <= self.bad_first:
            move = "not-a-move"
        return {"text": json.dumps({"reason": "first", "plan": "", "move": move}), "meta": dict(self.meta)}


class TestFixtures:
    def test_builtin_non_chess_hints_pass_without_chess(self):
        for game in (Connect4(), Othello()):
            report = check_hints(game, builtin_fixtures(game.title))
            assert report.summary()["checks"] >= 2 and not report.regressions and not report.known_gaps

    def test_invalid_history_fails_explicitly(self):
        bad = Fixture.from_dict(
            {
                "id": "x",
                "game": "Connect Four",
                "split": "dev",
                "provenance": "t",
                "moves": ["1"] * 7,
                "acceptable": ["2"],
            }
        )
        with pytest.raises(FixtureError, match="illegal Connect Four move"):
            fixture_state(Connect4(), bad)
        with pytest.raises(FixtureError, match="is for Connect Four"):
            fixture_state(Othello(), bad)

    @pytest.mark.parametrize(
        "data,error",
        [
            ({"id": "x", "game": "g", "split": "dev"}, "missing"),
            ({"id": "x", "game": "g", "split": "train", "provenance": "p", "acceptable": ["1"]}, "split"),
            ({"id": "x", "game": "g", "split": "dev", "provenance": "p"}, "no acceptable"),
        ],
    )
    def test_malformed_fixtures_are_rejected(self, data, error):
        with pytest.raises(FixtureError, match=error):
            Fixture.from_dict(data)

    def test_fixture_files_round_trip(self, tmp_path):
        path = tmp_path / "pack.json"
        path.write_text(json.dumps({"fixtures": [f.to_dict() for f in builtin_fixtures("Othello")]}), encoding="utf-8")
        assert [f.id for f in load_fixtures(path)] == [f.id for f in builtin_fixtures("Othello")]

    def test_held_out_split_needs_explicit_request(self):
        fixtures = builtin_fixtures("Connect Four")
        assert {f.split for f in select(fixtures)} == {"dev", "validation"}
        with pytest.raises(ValueError, match="held-out"):
            select(fixtures, ["test"])
        assert [f.id for f in select(fixtures, ["test"], include_test=True)] == ["c4-quiet-opening"]


class TestRuns:
    def make(self, **kwargs: Any) -> Any:
        return lambda: GameHarness(Connect4(), driver=FirstListed(**kwargs), level=1, concepts="all", seed=3)

    def test_runs_are_reproducible_and_reconcile_with_raw_records(self):
        fixtures = builtin_fixtures("Connect Four")
        first = evaluate(self.make(), fixtures, game=Connect4(), repeats=3, seed=11)
        second = evaluate(self.make(), fixtures, game=Connect4(), repeats=3, seed=11)
        strip = lambda run: [(s.fixture, s.repeat, s.order, s.move, s.success) for s in run.samples]  # noqa: E731
        assert strip(first) == strip(second)
        assert len(first.samples) == 2 * 3  # only the two decision fixtures, three repeats each
        card = scorecard(first)
        judged = [s for s in first.samples if s.success is not None]
        assert card["strength"]["unassisted_success"]["denominator"] == len(judged)
        assert card["strength"]["unassisted_success"]["numerator"] == sum(s.success for s in judged)
        assert card["efficiency"]["tokens_per_accepted_action"]["numerator"] == 25 * len(first.samples)
        assert card["strength"]["blunders"] == "not_assessed"
        assert "scripted evidence: not a measure of model quality" in first.notes
        assert first.fingerprint and first.config["game"] == "Connect Four" and first.model == "scripted/first-listed"

    def test_invalid_replies_rescues_and_failures_are_counted(self):
        fixtures = builtin_fixtures("Connect Four")
        run = evaluate(self.make(bad_first=1), fixtures, game=Connect4(), seed=1)
        card = scorecard(run)
        rel = card["reliability"]
        assert rel["first_answer_validity"]["numerator"] == 0 and rel["first_answer_validity"]["denominator"] == 2
        # A fresh player per sample: each sample's first reply is invalid, its retry valid.
        assert rel["invalid_per_100_model_attempts"] == {"value": 50.0, "numerator": 2, "denominator": 4}
        strict = lambda: GameHarness(Connect4(), driver=FirstListed(bad_first=99), max_retries=0, failure_policy="stop")  # noqa: E731
        failed = scorecard(evaluate(strict, fixtures, game=Connect4()))
        assert failed["reliability"]["failed_turns"]["numerator"] == 2
        assert failed["reliability"]["turn_failures"] == {"invalid_output": 2}
        assert failed["strength"]["unassisted_success"]["numerator"] == 0

    def test_unknown_prices_make_cost_a_lower_bound(self):
        player = lambda: GameHarness(Othello(), driver=FirstListed(meta={"total_tokens": 5}))  # noqa: E731
        run = evaluate(player, builtin_fixtures("Othello"), game=Othello())
        card = scorecard(run)
        assert not card["efficiency"]["cost_per_accepted_action"]["complete"]
        assert any(m.startswith("cost:") for m in card["missing"])
        assert "(lower bound)" in summary_text(card)

    def test_a_player_for_another_game_is_rejected(self):
        with pytest.raises(ValueError, match="plays Connect Four"):
            evaluate(self.make(), builtin_fixtures("Othello"), game=Othello())

    def test_judge_supplies_blunder_counts(self):
        run = evaluate(self.make(), builtin_fixtures("Connect Four"), game=Connect4(), judge=lambda g, s, m: m == 0)
        assert scorecard(run)["strength"]["blunders"]["denominator"] == 2

    def test_saved_runs_round_trip_and_versions_are_checked(self, tmp_path):
        run = evaluate(self.make(), builtin_fixtures("Connect Four"), game=Connect4())
        path = save_run(run, tmp_path / "run.json")
        loaded = load_run(path)
        assert scorecard(loaded) == scorecard(run)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["version"] = "0"
        with pytest.raises(ValueError, match="version"):
            EvaluationRun.from_dict(data)

    def test_match_clean_completions(self):
        game = Connect4()
        results = [
            play_match(game, {"Red": RandomPlayer(game, seed=i), "Yellow": RandomPlayer(game, seed=i + 1)})
            for i in range(3)
        ]
        results.append(play_match(game, {"Red": RandomPlayer(game), "Yellow": RandomPlayer(game)}, max_moves=2))
        card = scorecard([], matches=results)
        assert card["reliability"]["clean_completions_per_started_game"] == {
            "value": 0.75,
            "numerator": 3,
            "denominator": 4,
        }
        assert card["reliability"]["terminal_reasons"] == {"game_over": 3, "move_cap": 1}

    def test_paired_schedule_gives_every_config_the_same_jobs(self):
        jobs = paired_schedule(builtin_fixtures("Othello"), ["base", "hints"], repeats=2, seed=4)
        by_config = {c: sorted((f, r) for cc, f, r in jobs if cc == c) for c in ("base", "hints")}
        assert by_config["base"] == by_config["hints"] and len(jobs) == 8
        assert jobs != sorted(jobs)


def synthetic(
    success: list[bool],
    *,
    evidence: str = "live",
    config: dict[str, Any] | None = None,
    model: str = "m",
    invalid: int = 0,
) -> EvaluationRun:
    samples = [
        Sample(
            f"f{i}",
            "dev",
            0,
            i,
            "x",
            ok,
            "acceptable",
            "accepted",
            True,
            False,
            1,
            1,
            invalid if i == 0 else 0,
            True,
            None,
            {},
            {"total_tokens": 10, "cost": 0.01, "cost_complete": True},
            5.0,
        )
        for i, ok in enumerate(success)
    ]
    return EvaluationRun(
        "r",
        "",
        "g",
        evidence,
        0,
        1,
        {f"f{i}": "dev" for i in range(len(success))},
        "fp",
        config or {"concepts": []},
        model,
        samples,
        0,
        0,
    )


class TestComparisons:
    prereg = Preregistration(factor="concepts", min_pairs=20, resamples=500)

    def test_improvement_needs_a_positive_paired_interval(self):
        base = synthetic([False] * 30 + [True] * 10)
        cand = synthetic([True] * 30 + [True] * 10, config={"concepts": ["fork"]})
        result = compare(base, cand, self.prereg)
        assert result.verdict == "improved" and result.interval[0] > 0 and result.pairs == 40

    def test_regression_and_inconclusive(self):
        base = synthetic([True] * 40)
        assert (
            compare(base, synthetic([False] * 30 + [True] * 10, config={"concepts": ["fork"]}), self.prereg).verdict
            == "regression"
        )
        mixed = synthetic([i % 2 == 0 for i in range(40)], config={"concepts": ["fork"]})
        assert compare(synthetic([i % 2 == 1 for i in range(40)]), mixed, self.prereg).verdict == "inconclusive"
        small = compare(synthetic([False] * 5), synthetic([True] * 5, config={"concepts": ["fork"]}), self.prereg)
        assert small.verdict == "inconclusive" and "minimum" in small.reasons[0]

    def test_scripted_unregistered_and_unfair_comparisons_get_no_verdict(self):
        base = synthetic([False] * 40)
        cand = synthetic([True] * 40, config={"concepts": ["fork"]})
        assert compare(synthetic([False] * 40, evidence="scripted"), cand, self.prereg).verdict == "not_evidence"
        assert compare(base, cand, None).verdict == "not_preregistered"
        two_factors = synthetic([True] * 40, config={"concepts": ["fork"], "level": 1})
        unfair = compare(base, two_factors, self.prereg)
        assert unfair.verdict == "not_comparable" and "level" in unfair.reasons[0]
        assert (
            compare(base, synthetic([True] * 40, config={"concepts": ["fork"]}, model="other"), self.prereg).verdict
            == "not_comparable"
        )
        assert (
            compare(base, synthetic([True] * 39, config={"concepts": ["fork"]}), self.prereg).verdict
            == "not_comparable"
        )

    def test_reliability_gate_blocks_an_improvement_claim(self):
        base = synthetic([False] * 30 + [True] * 10)
        cand = synthetic([True] * 40, config={"concepts": ["fork"]}, invalid=5)
        result = compare(base, cand, self.prereg)
        assert result.verdict == "inconclusive" and any("invalid replies rose" in r for r in result.reasons)


chess = pytest.importorskip("chess")


class TestChessFixtures:
    def test_at_least_24_validated_positions_across_categories_and_splits(self):
        from prompture.games.chess import chess_fixtures

        fixtures = chess_fixtures()
        assert len(fixtures) >= 24
        tags = {t for f in fixtures for t in f.tags}
        for needed in (
            "mate",
            "fork",
            "exchange",
            "pin",
            "promotion",
            "en-passant",
            "repetition",
            "quiet-defense",
            "sacrifice",
            "negative",
            "regression",
            "x-ray",
        ):
            assert needed in tags
        assert {f.split for f in fixtures} == {"dev", "validation", "test"}
        assert all(f.provenance for f in fixtures)

    def test_plan_regressions_are_kept_and_valid(self):
        from prompture.games.chess import chess_fixtures

        table = {
            "regression-20-Nd7-allows-d6-fork": (
                "r1b2rk1/2q1bppp/5n2/p1pPp3/PpN1P3/5N1P/1PB2PP1/R2QR1K1 b - - 1 20",
                "Nd7",
                "d6",
            ),
            "regression-21-Qb6-loses-queen": (
                "r1b2rk1/2qnbppp/3P4/p1p1p3/PpN1P3/5N1P/1PB2PP1/R2QR1K1 b - - 0 21",
                "Qb6",
                "Nxb6",
            ),
            "regression-34-c5-loses-pawn": ("7k/6p1/7p/6r1/PrP5/3R1P2/5P1P/5R1K w - - 1 34", "c5", "Rxc5"),
        }
        by_id = {f.id: f for f in chess_fixtures()}
        for fixture_id, (fen, move, reply) in table.items():
            assert by_id[fixture_id].initial == fen
            board = chess.Board(fen)
            board.push_san(move)
            board.parse_san(reply)  # the warned reply is legal

    def test_avoid_mate_sets_match_python_chess(self):
        from prompture.games.chess import chess_fixtures

        for fixture in chess_fixtures():
            if fixture.id not in ("allows-mate-in-one", "quiet-defense-luft"):
                continue
            board = chess.Board(fixture.initial)
            safe = []
            for move in board.legal_moves:
                after = board.copy()
                after.push(move)
                mates = False
                for reply in list(after.legal_moves):
                    after.push(reply)
                    mates = mates or after.is_checkmate()
                    after.pop()
                if not mates:
                    safe.append(board.san(move))
            assert sorted(fixture.acceptable) == sorted(safe)

    def test_hint_report_has_no_regressions(self):
        from prompture.games.chess import chess_hint_report

        report = chess_hint_report()
        assert not report.regressions, [(c.fixture, c.move, c.problems) for c in report.regressions]
        assert not report.closed_gaps, [(c.fixture, c.move) for c in report.closed_gaps]

    def test_checkmate_evaluator_and_scripted_run(self):
        from prompture.games.chess import ChessGame, ChessHarness, chess_fixtures

        fixtures = [f for f in chess_fixtures() if f.decides and f.split != "test"]
        run = evaluate(
            lambda: ChessHarness(driver=FirstListed(), level=1, concepts="all"), fixtures, game=ChessGame(), seed=2
        )
        assert {s.verdict for s in run.samples} <= {"acceptable", "evaluator:checkmate"}
        assert scorecard(run)["strength"]["unassisted_success"]["denominator"] == len(fixtures)

    @pytest.mark.parametrize("level", [0, 1])
    def test_fixture_answers_never_reach_the_prompt(self, level):
        from prompture.games.chess import ChessHarness, chess_fixtures
        from prompture.games.chess.harness import ChessGame

        for fixture in chess_fixtures():
            state = fixture_state(ChessGame(), fixture)
            plain = chess.Board(fixture.initial) if fixture.initial else chess.Board()
            for name in fixture.moves:
                plain.push_san(name)
            harness = ChessHarness(driver=FirstListed(), level=level, concepts="all")
            prompt = harness.prompt(state, harness.candidates(state))
            assert prompt == harness.prompt(plain, harness.candidates(plain))
            assert fixture.id not in prompt and fixture.provenance not in prompt
            assert "acceptable" not in prompt.lower() and (not fixture.notes or fixture.notes not in prompt)

    def test_puzzle_report_uses_the_shared_scorecard(self):
        from prompture.games.chess import ChessHarness, run_puzzles

        puzzle = {"id": "mate", "fen": "7k/5ppp/8/8/8/8/8/RR4K1 b - - 0 1", "moves": ["h8g8", "a1a8"]}
        report = run_puzzles(ChessHarness(driver=FirstListed(bad_first=1)), [puzzle])
        card = report.scorecard()
        assert card["samples"] == 1 and card["reliability"]["invalid_per_100_model_attempts"]["numerator"] == 1


def test_generic_evaluation_runs_without_python_chess():
    """Block the chess import entirely: generic games, fixtures and evaluation still work."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['chess'] = None\n"
        "from prompture.games import Connect4, GameHarness, check_hints\n"
        "from prompture.games.fixtures import builtin_fixtures\n"
        "report = check_hints(Connect4(), builtin_fixtures('Connect Four'))\n"
        "assert not report.regressions\n"
        "try:\n"
        "    import prompture.games.chess\n"
        "except ImportError as exc:\n"
        "    assert 'python-chess' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('chess adapter imported without python-chess')\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
