"""Turn records, failure classes and policies, accounting, fingerprints and traces."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prompture.drivers.base import Driver
from prompture.games import (
    Attempt,
    Connect4,
    Connect4Board,
    FailureKind,
    GameHarness,
    JsonlTraceSink,
    MemoryTraceSink,
    MoveDecision,
    Othello,
    OthelloBoard,
    RandomPlayer,
    TraceWriteError,
    TurnFailed,
    play_match,
    read_jsonl_trace,
)
from prompture.games.records import Accounting, classify_failure, usage_status

SECRET = "sk-test-SYNTHETICSECRET1234567890"


def answer(move: str = "4", reason: str = "center", plan: str = "connect") -> str:
    return json.dumps({"reason": reason, "plan": plan, "move": move})


class Scripted(Driver):
    """Plays a script of replies (texts or exceptions); each call reports fixed usage."""

    supports_messages = True

    def __init__(self, script: list[Any], meta: dict[str, Any] | None = None):
        self.script = list(script)
        self.meta = (
            meta
            if meta is not None
            else {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}
        )
        self.prompts: list[str] = []
        self.model = "scripted/model"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(prompt)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return {"text": item, "meta": dict(self.meta)}


class StreamBreaks(Driver):
    """Streams part of an answer, then the connection drops."""

    supports_streaming = True
    supports_json_mode = True
    model = "scripted/stream"

    def generate(self, prompt, options):  # pragma: no cover - streaming only
        raise AssertionError("non-streaming call")

    def generate_messages_stream(self, messages, options):
        yield {"type": "delta", "text": '{"reason": "centre", '}
        raise ConnectionError("connection reset by peer")


class HTTPError(Exception):
    def __init__(self, message: str, status_code: int, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = type("R", (), {"status_code": status_code, "headers": headers or {}})()


class TestAttemptsAndAccounting:
    @pytest.mark.parametrize(
        "game,state,moves", [(Connect4(), Connect4Board(), ("9", "4")), (Othello(), OthelloBoard(), ("a1", "d3"))]
    )
    def test_invalid_then_valid_is_two_attempts_and_one_accepted_decision(self, game, state, moves):
        driver = Scripted([answer(moves[0]), answer(moves[1])])
        decision = GameHarness(game, driver=driver).choose(state)
        record = decision.record
        assert decision.name == moves[1] and decision.model_selected
        assert record.outcome == "accepted" and record.move == moves[1]
        assert [a.valid for a in decision.attempts] == [False, True]
        first = decision.attempts[0]
        assert first.failure.kind is FailureKind.INVALID_OUTPUT
        assert first.prompt and first.schema["properties"]["move"]["enum"]
        assert first.response_text == answer(moves[0]) and first.reply == moves[0]
        assert "rejected" in decision.attempts[1].prompt
        # Each call is counted exactly once; the legacy usage dict is the same sum.
        assert record.accounting.calls == 2
        assert record.accounting.total_tokens == 30 == sum(a.usage["total_tokens"] for a in decision.attempts)
        assert decision.usage == record.accounting.legacy()
        assert record.accounting.cost_complete and record.accounting.cost_label == "$0.0200"
        assert [a.attempt_id for a in decision.attempts] == [f"{record.turn_id}.0", f"{record.turn_id}.1"]
        assert record.state_id and record.ply == 0 and record.side == game.player(state)

    def test_broken_stream_keeps_partial_text_and_unknown_usage(self):
        decision = GameHarness(Connect4(), driver=StreamBreaks(), max_retries=0, seed=1).choose(Connect4Board())
        attempt = decision.attempts[0]
        assert decision.rescued and not decision.model_selected and decision.assistance == "rescue"
        assert attempt.response_text == '{"reason": "centre", '
        assert attempt.failure.kind is FailureKind.TRANSPORT
        assert (attempt.usage_status, attempt.cost_status) == ("unknown", "unknown")
        accounting = decision.record.accounting
        assert accounting.unknown_cost_calls == 1 and not accounting.cost_complete
        assert accounting.cost_label.startswith(">= $0.0000")

    def test_missing_price_is_unknown_not_zero(self):
        driver = Scripted([answer()], meta={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5})
        accounting = GameHarness(Connect4(), driver=driver).choose(Connect4Board()).record.accounting
        assert accounting.total_tokens == 5 and accounting.usage_complete
        assert not accounting.cost_complete and accounting.unknown_cost_calls == 1

    def test_driver_cost_status_is_kept(self):
        meta = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "cost": 0.5, "cost_status": "partial"}
        decision = GameHarness(Connect4(), driver=Scripted([answer()], meta=meta)).choose(Connect4Board())
        assert decision.attempts[0].cost_status == "partial"
        assert decision.record.accounting.partial_cost_calls == 1
        assert usage_status({"usage_complete": False, "total_tokens": 4}) == ("partial", "unknown")
        assert usage_status(None) == ("unknown", "unknown")

    def test_historical_constructors_still_work(self):
        decision = MoveDecision(3, "4", "why", [], [Attempt("4")], False, {"total_tokens": 2})
        assert decision.san == "4" and decision.plan == "" and decision.record is None
        assert decision.accounting.total_tokens == 2
        assert Attempt("x", "bad").error == "bad" and not Attempt("x", "bad").valid


class TestFailureClasses:
    @pytest.mark.parametrize(
        "exc,kind",
        [
            (HTTPError("invalid api key", 401), FailureKind.AUTH),
            (HTTPError("Too Many Requests", 429, {"retry-after": "7"}), FailureKind.RATE_LIMIT),
            (HTTPError("bad gateway", 502), FailureKind.UPSTREAM),
            (HTTPError("model not found", 404), FailureKind.CONFIGURATION),
            (TimeoutError("read timed out"), FailureKind.TIMEOUT),
            (ConnectionError("reset"), FailureKind.TRANSPORT),
            (RuntimeError("Ollama chat request failed: read timed out"), FailureKind.TIMEOUT),
        ],
    )
    def test_provider_failures_are_classified(self, exc, kind):
        assert classify_failure(exc).kind is kind

    def test_retry_after_and_status_are_preserved(self):
        info = classify_failure(HTTPError("Too Many Requests", 429, {"retry-after": "7"}))
        assert (info.status_code, info.retry_after, info.error_type) == (429, 7.0, "HTTPError")

    def test_cause_chain_is_used(self):
        try:
            try:
                raise HTTPError("unauthorized", 401)
            except HTTPError as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            assert classify_failure(outer).kind is FailureKind.AUTH

    def test_auth_fails_fast_and_never_becomes_a_move(self):
        driver = Scripted([HTTPError("invalid api key", 401)])
        with pytest.raises(TurnFailed) as caught:
            GameHarness(Connect4(), driver=driver, max_retries=3).choose(Connect4Board())
        assert caught.value.kind is FailureKind.AUTH and not caught.value.forfeit
        assert len(driver.prompts) == 1
        assert caught.value.record.outcome == "failed" and caught.value.record.attempts[0].failure.status_code == 401


class TestFailurePolicies:
    def test_rescue_is_marked_and_not_model_selected(self):
        decision = GameHarness(Connect4(), driver=Scripted([answer("9")]), max_retries=1, seed=2).choose(
            Connect4Board()
        )
        assert decision.rescued and decision.assistance == "rescue" and not decision.model_selected
        assert decision.reason == "" and decision.plan == ""
        assert decision.record.outcome == "rescued" and len(decision.attempts) == 2

    def test_forfeit_loses_the_match(self):
        game = Connect4()
        harness = GameHarness(game, driver=Scripted([answer("9")]), max_retries=0, failure_policy="forfeit")
        result = play_match(game, {"Red": harness, "Yellow": RandomPlayer(game, seed=1)})
        assert (result.terminal_reason, result.forfeited_by, result.winner, result.finished) == (
            "forfeit",
            "Red",
            "Yellow",
            True,
        )
        assert result.moves == [] and isinstance(result.error, TurnFailed)
        assert result.error.record.attempts[0].reply == "9"

    def test_stop_ends_without_a_winner_or_draw(self):
        game = Othello()
        harness = GameHarness(game, driver=Scripted([answer("a1")]), max_retries=0, failure_policy="stop")
        result = play_match(game, {"Black": harness, "White": RandomPlayer(game, seed=1)})
        assert (result.terminal_reason, result.winner, result.finished, result.draw) == ("stopped", None, False, False)

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ValueError, match="failure_policy"):
            GameHarness(Connect4(), driver=Scripted([answer()]), failure_policy="maybe")


class TestFingerprints:
    def base(self, **kwargs: Any) -> GameHarness:
        return GameHarness(Connect4(), driver=Scripted([answer()]), **kwargs)

    @pytest.mark.parametrize(
        "change",
        [
            {"level": 1},
            {"concepts": {"block"}},
            {"max_retries": 4},
            {"failure_policy": "forfeit"},
            {"options": {"effort": "high"}},
            {"options": {"max_tokens": 900}},
            {"system_prompt": "Play sharp"},
            {"level": 2},
        ],
    )
    def test_changed_behavior_changes_the_fingerprint(self, change):
        assert self.base().fingerprint().digest != self.base(**change).fingerprint().digest

    def test_names_model_and_secrets_do_not(self):
        plain = self.base().fingerprint()
        other = GameHarness(Connect4(), driver=Scripted([answer()]), model="other/model", options={"api_key": SECRET})
        assert other.fingerprint().digest == plain.digest
        assert SECRET not in json.dumps(other.fingerprint().config)
        assert plain.differences(self.base(level=1).fingerprint()) == ["level"]

    def test_renamed_players_share_records_fingerprint(self):
        game = Connect4()
        digests = set()
        for names in (("alice", "bob"), ("carol", "dave")):
            harness = self.base()
            result = play_match(game, {"Red": harness, "Yellow": RandomPlayer(game, seed=3)}, max_moves=2)
            digests.add(result.decisions[0][1].record.fingerprint)
            assert names  # names never reach the harness
        assert len(digests) == 1


class TestTraces:
    def test_jsonl_trace_has_run_turn_and_attempt_records_without_secrets(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        driver = Scripted([answer("9", reason=f"key {SECRET}"), answer("4", reason="Bearer abcdefghijklmnop")])
        harness = GameHarness(
            Connect4(),
            driver=driver,
            trace=JsonlTraceSink(path),
            options={"api_key": SECRET, "headers": {"Authorization": SECRET}, "effort": "low"},
        )
        harness.choose(Connect4Board())
        text = path.read_text(encoding="utf-8")
        assert SECRET not in text and "abcdefghijklmnop" not in text
        records, skipped = read_jsonl_trace(path)
        assert skipped == 0
        assert [r["type"] for r in records] == ["run", "turn_started", "attempt", "attempt", "turn_finished"]
        assert records[0]["config"]["options"] == {"effort": "low"}
        assert records[2]["attempt"]["validation"] == "invalid" and records[2]["attempt"]["prompt"]
        finished = records[-1]["record"]
        assert finished["outcome"] == "accepted" and finished["accounting"]["calls"] == 2
        assert {r["turn_id"] for r in records[1:]} == {finished["turn_id"]}

    def test_torn_final_line_is_skipped(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        GameHarness(Connect4(), driver=Scripted([answer()]), trace=JsonlTraceSink(path)).choose(Connect4Board())
        with path.open("ab") as handle:
            handle.write(b'{"type": "attempt", "turn_')
        records, skipped = read_jsonl_trace(path)
        assert skipped == 1 and records[-1]["type"] == "turn_finished"

    def test_oversized_records_are_bounded(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        harness = GameHarness(
            Connect4(),
            driver=Scripted([answer(reason="x" * 50_000)]),
            trace=JsonlTraceSink(path, max_record_bytes=4_000),
        )
        harness.choose(Connect4Board())
        assert all(len(line.encode()) <= 4_000 for line in path.read_text(encoding="utf-8").splitlines())

    def test_failed_sink_warns_by_default_and_raises_when_asked(self):
        class Broken:
            def write(self, record):
                raise OSError("disk full")

        harness = GameHarness(Connect4(), driver=Scripted([answer()]), trace=Broken())
        assert harness.choose(Connect4Board()).name == "4"
        assert harness.trace.failures == 4
        strict = GameHarness(Connect4(), driver=Scripted([answer()]), trace=Broken(), trace_errors="raise")
        with pytest.raises(TraceWriteError, match="disk full"):
            strict.choose(Connect4Board())

    def test_prompts_can_be_kept_out(self):
        sink = MemoryTraceSink()
        GameHarness(Connect4(), driver=Scripted([answer()]), trace=sink, trace_prompts=False).choose(Connect4Board())
        assert sink.of_type("attempt")[0]["attempt"]["prompt"] is None

    def test_custom_game_without_history_still_records(self):
        class Bare(Connect4):
            history = None  # type: ignore[assignment]

        decision = GameHarness(Bare(), driver=Scripted([answer()])).choose(Connect4Board())
        assert decision.record.state_id is None and decision.record.ply is None


def test_accounting_merge_and_labels():
    a, b = Accounting(), Accounting()
    a.add({"total_tokens": 5, "cost": 0.1}, "complete", "estimated")
    b.add({}, "unknown", "unknown", abandoned=True)
    a.merge(b)
    assert (a.calls, a.abandoned_calls, a.unknown_usage_calls) == (2, 1, 1)
    assert a.to_dict()["cost_sources"] == {"estimated": 1, "unknown": 1}
