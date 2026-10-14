"""Turn deadlines, cancellation, transient retries, budgets and progress events."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest

from prompture.drivers.base import Driver
from prompture.games import (
    BudgetExhausted,
    Connect4,
    Connect4Board,
    FailureKind,
    GameHarness,
    Othello,
    OthelloBoard,
    RandomPlayer,
    TurnCancelled,
    TurnFailed,
    play_match,
)
from prompture.games.control import CancelToken, EventBuffer, GameBudget, RetryPolicy, TurnEvent
from prompture.infra.callbacks import DriverCallbacks

META = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01}


def answer(move: str = "4") -> str:
    return json.dumps({"reason": "center", "plan": "connect", "move": move})


class Blocking(Driver):
    """A blocking call that ignores every stop signal for ``delay`` seconds."""

    supports_messages = True
    model = "scripted/blocking"

    def __init__(self, delay: float, text: str | None = None, meta: dict[str, Any] | None = None):
        self.delay = delay
        self.text = text or answer()
        self.meta = META if meta is None else meta
        self.calls = 0
        self.finished = threading.Event()

    def generate(self, prompt, options):
        self.calls += 1
        time.sleep(self.delay)
        self.finished.set()
        return {"text": self.text, "meta": dict(self.meta)}


class Script(Driver):
    """Replies or raises from a script, instantly."""

    supports_messages = True
    model = "scripted/script"

    def __init__(self, script: list[Any], meta: dict[str, Any] | None = None):
        self.script = list(script)
        self.meta = META if meta is None else meta
        self.prompts: list[str] = []

    def generate(self, prompt, options):
        self.prompts.append(prompt)
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        return {"text": item, "meta": dict(self.meta)}


class SlowStream(Driver):
    """Streams an answer one character per ``pause`` seconds; records closes, never closes itself."""

    supports_streaming = True
    supports_json_mode = True
    model = "scripted/stream"

    def __init__(self, pause: float = 0.02, text: str | None = None):
        self.pause = pause
        self.text = text or answer()
        self.closed = 0
        self.client_closed = False
        self.callbacks = DriverCallbacks(on_stream_delta=lambda payload: None)
        self.lock = threading.Lock()

    def generate(self, prompt, options):  # pragma: no cover - streaming only
        raise AssertionError("non-streaming call")

    def close(self):  # a shared client: no game may close it
        self.client_closed = True

    def generate_messages_stream(self, messages, options):
        try:
            for ch in self.text:
                time.sleep(self.pause)
                yield {"type": "delta", "text": ch}
            yield {"type": "done", "text": self.text, "meta": dict(META)}
        finally:
            with self.lock:
                self.closed += 1


class HTTPError(Exception):
    def __init__(self, message: str, status_code: int, retry_after: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        headers = {"retry-after": retry_after} if retry_after else {}
        self.response = type("R", (), {"status_code": status_code, "headers": headers})()


class TestDeadlines:
    def test_stalled_call_stops_within_the_bound_and_its_late_reply_is_never_applied(self):
        driver = Blocking(delay=1.0)
        harness = GameHarness(Connect4(), driver=driver, deadline=0.3, cleanup_timeout=0.05, seed=1)
        started = time.monotonic()
        decision = harness.choose(Connect4Board())
        elapsed = time.monotonic() - started
        assert elapsed < 0.3 + 0.05 + 0.25  # deadline + cleanup + polling slack
        attempt = decision.attempts[0]
        assert decision.rescued and decision.assistance == "rescue"
        assert attempt.abandoned and attempt.failure.kind is FailureKind.TIMEOUT
        assert attempt.failure.category == "turn_deadline"
        assert (attempt.usage_status, attempt.cost_status) == ("unknown", "unknown")
        assert decision.record.accounting.abandoned_calls == 1
        assert driver.finished.wait(2) and _eventually(lambda: attempt.late)
        assert driver.calls == 1  # nothing was sent on the abandoned request's behalf

    def test_expired_turn_follows_the_stop_policy(self):
        harness = GameHarness(
            Connect4(), driver=Blocking(delay=0.6), deadline=0.1, cleanup_timeout=0.0, failure_policy="stop"
        )
        with pytest.raises(TurnFailed) as caught:
            harness.choose(Connect4Board())
        assert caught.value.kind is FailureKind.TIMEOUT and caught.value.record.outcome == "failed"

    def test_retries_and_backoff_cannot_extend_the_deadline(self):
        driver = Script([HTTPError("bad gateway", 502)])
        retry = RetryPolicy(transient_retries=50, base_delay=0.2, jitter=0)
        harness = GameHarness(Connect4(), driver=driver, deadline=0.5, retry=retry, failure_policy="stop")
        started = time.monotonic()
        with pytest.raises(TurnFailed) as caught:
            harness.choose(Connect4Board())
        assert time.monotonic() - started < 0.8
        assert caught.value.record.failure.category == "turn_deadline"
        assert 2 <= len(driver.prompts) <= 4

    def test_fast_calls_with_a_deadline_still_succeed(self):
        decision = GameHarness(Connect4(), driver=Script([answer()]), deadline=5).choose(Connect4Board())
        assert decision.model_selected and decision.record.deadline_s == 5


class TestCancellation:
    def test_cancel_during_a_blocking_call(self):
        token = CancelToken()
        driver = Blocking(delay=1.0)
        harness = GameHarness(Connect4(), driver=driver, cancel=token, cleanup_timeout=0.05)
        threading.Timer(0.1, token.cancel).start()
        started = time.monotonic()
        with pytest.raises(TurnCancelled) as caught:
            harness.choose(Connect4Board())
        assert time.monotonic() - started < 0.5
        record = caught.value.record
        assert record.outcome == "cancelled" and record.move is None
        assert record.attempts[0].abandoned and record.attempts[0].failure.kind is FailureKind.CANCELLED
        assert driver.finished.wait(2) and _eventually(lambda: record.attempts[0].late)
        assert driver.calls == 1

    def test_cancelled_before_start_sends_nothing(self):
        token = CancelToken()
        token.cancel()
        driver = Script([answer()])
        with pytest.raises(TurnCancelled):
            GameHarness(Connect4(), driver=driver).choose(Connect4Board(), cancel=token)
        assert driver.prompts == []

    def test_cancelling_a_match_never_plays_a_fallback(self):
        token = CancelToken()
        game = Connect4()
        harness = GameHarness(game, driver=Blocking(delay=0.5), cancel=token, cleanup_timeout=0.0)
        threading.Timer(0.05, token.cancel).start()
        result = play_match(game, {"Red": harness, "Yellow": RandomPlayer(game, seed=1)})
        assert (result.terminal_reason, result.finished, result.winner, result.moves) == ("cancelled", False, None, [])

    def test_stream_is_closed_and_partial_output_kept(self):
        token = CancelToken()
        driver = SlowStream(pause=0.03)
        events: list[TurnEvent] = []
        harness = GameHarness(Connect4(), driver=driver, cancel=token, on_event=events.append, cleanup_timeout=0.5)
        threading.Timer(0.2, token.cancel).start()
        with pytest.raises(TurnCancelled) as caught:
            harness.choose(Connect4Board())
        attempt = caught.value.record.attempts[0]
        assert not attempt.abandoned  # the stream stopped at its next chunk
        assert attempt.response_text and answer().startswith(attempt.response_text)
        assert driver.closed == 1 and not driver.client_closed
        assert any(e.kind == "output_delta" for e in events)
        assert events[-1].kind == "turn_cancelled"

    def test_two_games_sharing_a_driver_are_isolated(self):
        driver = SlowStream(pause=0.01)
        token_a = CancelToken()
        results: dict[str, Any] = {}

        def play(name: str, token: CancelToken) -> None:
            harness = GameHarness(Connect4(), driver=driver, cancel=token)
            try:
                results[name] = harness.choose(Connect4Board())
            except TurnCancelled as exc:
                results[name] = exc

        threads = [
            threading.Thread(target=play, args=("a", token_a)),
            threading.Thread(target=play, args=("b", CancelToken())),
        ]
        for thread in threads:
            thread.start()
        time.sleep(0.1)
        token_a.cancel()
        for thread in threads:
            thread.join(5)
        assert isinstance(results["a"], TurnCancelled)
        assert results["b"].model_selected and results["b"].name == "4"
        assert driver.closed == 2 and not driver.client_closed

    def test_a_completion_racing_a_cancel_is_not_returned(self):
        token = CancelToken()

        def cancel_on_success(event: TurnEvent) -> None:
            if event.kind == "attempt_finished" and event.data["valid"]:
                token.cancel("raced")

        harness = GameHarness(Connect4(), driver=Script([answer()]), cancel=token, on_event=cancel_on_success)
        with pytest.raises(TurnCancelled, match="before its move was returned"):
            harness.choose(Connect4Board())

    def test_parent_token_cancels_children(self):
        batch = CancelToken()
        game_token = batch.child()
        batch.cancel("batch stopped")
        assert game_token.cancelled and game_token.reason == "batch stopped"

    def test_achoose_cancellation_reaches_the_turn(self):
        driver = Blocking(delay=0.6)
        events: list[TurnEvent] = []
        harness = GameHarness(Connect4(), driver=driver, on_event=events.append, cleanup_timeout=0.0)

        async def scenario() -> None:
            task = asyncio.ensure_future(harness.achoose(Connect4Board()))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        assert _eventually(lambda: any(e.kind == "turn_cancelled" for e in events))

    def test_achoose_returns_a_decision(self):
        decision = asyncio.run(GameHarness(Othello(), driver=Script([answer("d3")])).achoose(OthelloBoard()))
        assert decision.name == "d3"


class TestTransientRetries:
    def test_transient_failures_have_their_own_allowance_and_no_feedback(self):
        driver = Script([HTTPError("Too Many Requests", 429, retry_after="0.05"), answer()])
        harness = GameHarness(Connect4(), driver=driver, max_retries=0, retry=RetryPolicy(base_delay=0.01, jitter=0))
        events: list[TurnEvent] = []
        harness.on_event = events.append
        started = time.monotonic()
        decision = harness.choose(Connect4Board())
        assert decision.model_selected and len(decision.attempts) == 2
        assert time.monotonic() - started >= 0.04  # Retry-After honored (Windows clock ticks ~15ms)
        assert "rejected" not in driver.prompts[1]
        retry = next(e for e in events if e.kind == "retry_scheduled")
        assert retry.data == {"delay_s": 0.05, "failure": "rate_limit"}

    def test_exhausted_transient_allowance_uses_the_policy(self):
        driver = Script([HTTPError("bad gateway", 502)])
        harness = GameHarness(
            Connect4(), driver=driver, retry=RetryPolicy(transient_retries=1, base_delay=0.0, jitter=0), seed=1
        )
        decision = harness.choose(Connect4Board())
        assert decision.rescued and len(driver.prompts) == 2
        assert decision.record.failure.kind is FailureKind.UPSTREAM

    def test_without_a_policy_transient_failures_keep_legacy_behavior(self):
        driver = Script([TimeoutError("read timed out"), answer()])
        decision = GameHarness(Connect4(), driver=driver, max_retries=1).choose(Connect4Board())
        assert decision.model_selected and decision.attempts[0].failure.kind is FailureKind.TIMEOUT

    def test_backoff_schedule(self):
        policy = RetryPolicy(base_delay=0.5, max_delay=3, jitter=0, max_retry_after=10)
        assert [policy.delay(n) for n in (1, 2, 3, 4)] == [0.5, 1.0, 2.0, 3]
        assert policy.delay(1, retry_after=7) == 7 and policy.delay(1, retry_after=60) == 10
        jittered = RetryPolicy(base_delay=1, jitter=0.25).delay(1)
        assert 1 <= jittered <= 1.25


class TestBudgets:
    def test_call_budget_prevents_extra_requests(self):
        driver = Script([answer("9")])
        harness = GameHarness(Connect4(), driver=driver, max_retries=5, budget=GameBudget(max_calls=2))
        with pytest.raises(BudgetExhausted) as caught:
            harness.choose(Connect4Board())
        assert len(driver.prompts) == 2 and caught.value.kind is FailureKind.BUDGET
        assert caught.value.record.outcome == "failed" and not caught.value.forfeit

    def test_a_budget_is_shared_across_harnesses(self):
        budget = GameBudget(max_calls=1)
        GameHarness(Connect4(), driver=Script([answer()]), budget=budget).choose(Connect4Board())
        assert (budget.calls, budget.tokens) == (1, 15)
        with pytest.raises(BudgetExhausted, match="call limit"):
            GameHarness(Connect4(), driver=Script([answer()]), budget=budget).choose(Connect4Board())

    def test_token_budget_counts_the_estimate_before_sending(self):
        prompt_tokens = len(GameHarness(Connect4(), driver=Script([answer()])).prompt(Connect4Board(), [])) / 4
        budget = GameBudget(max_tokens=prompt_tokens + 100, output_reserve=4000)
        driver = Script([answer()])
        with pytest.raises(BudgetExhausted, match="token limit"):
            GameHarness(Connect4(), driver=driver, budget=budget).choose(Connect4Board())
        assert driver.prompts == []

    def test_unknown_price_stops_a_cost_only_budget(self):
        driver = Script([answer("9")], meta={"total_tokens": 4})
        harness = GameHarness(Connect4(), driver=driver, max_retries=3, budget=GameBudget(max_cost=1.0))
        with pytest.raises(BudgetExhausted, match="price is unknown"):
            harness.choose(Connect4Board())
        assert len(driver.prompts) == 1

    def test_unknown_price_with_another_bound_continues_unless_strict(self):
        lenient = GameBudget(max_cost=1.0, max_calls=10)
        driver = Script([answer("9"), answer()], meta={"total_tokens": 4})
        assert GameHarness(Connect4(), driver=driver, budget=lenient).choose(Connect4Board()).model_selected
        strict = GameBudget(max_cost=1.0, max_calls=10, strict=True)
        with pytest.raises(BudgetExhausted):
            GameHarness(Connect4(), driver=Script([answer("9")], meta={"total_tokens": 4}), budget=strict).choose(
                Connect4Board()
            )

    def test_reservations_hold_capacity_for_concurrent_work(self):
        budget = GameBudget(max_calls=1)
        held = budget.reserve(10)
        with pytest.raises(BudgetExhausted):
            budget.reserve(10)
        budget.release(held)
        budget.settle(budget.reserve(10), {"total_tokens": 3}, "estimated")
        assert budget.snapshot()["calls"] == 1 and budget.check() is not None

    def test_time_budget(self):
        budget = GameBudget(max_seconds=0.0)
        with pytest.raises(BudgetExhausted, match="time budget"):
            budget.reserve(1)


class TestEvents:
    def test_sequenced_events_compose_with_driver_callbacks(self):
        driver = SlowStream(pause=0)
        seen: list[str] = []
        driver.callbacks = DriverCallbacks(on_stream_delta=lambda payload: seen.append(payload["text"]))
        events: list[TurnEvent] = []
        decision = GameHarness(Connect4(), driver=driver, on_event=events.append).choose(Connect4Board())
        kinds = [e.kind for e in events]
        assert kinds[0] == "turn_started" and kinds[1] == "attempt_started" and kinds[-1] == "turn_completed"
        assert kinds.count("output_delta") == len(answer())
        assert [e.seq for e in events] == sorted(e.seq for e in events)
        assert "".join(seen) == answer()  # the caller's own callbacks still run
        assert events[-1].data["move"] == decision.name and events[-1].data["accounting"]["calls"] == 1

    def test_a_failing_handler_does_not_break_the_turn(self):
        def broken(event):
            raise RuntimeError("consumer bug")

        assert GameHarness(Connect4(), driver=Script([answer()]), on_event=broken).choose(Connect4Board()).name == "4"

    def test_event_buffer_coalesces_and_stays_bounded(self):
        buffer = EventBuffer(maxsize=4)
        harness = GameHarness(Connect4(), driver=SlowStream(pause=0), on_event=buffer.put)
        harness.choose(Connect4Board())
        events = buffer.drain()
        assert len(events) <= 4 and buffer.dropped >= 1
        assert events[-1].kind == "turn_completed"

    def test_event_buffer_merges_deltas_of_one_attempt(self):
        buffer = EventBuffer(maxsize=2)
        for seq, text in enumerate("abcdef"):
            buffer.put(TurnEvent(seq, "output_delta", "r", "g", "t", "Red", 0.0, {"attempt_id": "t.0", "text": text}))
        events = buffer.drain()
        assert len(events) == 2 and "".join(e.data["text"] for e in events) == "abcdef"


def _eventually(check, timeout: float = 2.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.01)
    return False


class TestProviderStreamClose:
    """Real driver stream code: a stopped turn closes the HTTP response it was reading."""

    def test_ollama_stream_is_closed_when_a_turn_is_cancelled(self, monkeypatch):
        from prompture.drivers import ollama_driver

        closed = threading.Event()
        text = answer()

        class Response:
            def raise_for_status(self):
                return None

            def iter_lines(self):
                for ch in text:
                    time.sleep(0.02)
                    yield json.dumps({"message": {"content": ch}}).encode()
                yield json.dumps({"done": True, "prompt_eval_count": 5, "eval_count": 5}).encode()

            def close(self):
                closed.set()

        monkeypatch.setattr(ollama_driver.requests, "post", lambda *a, **k: Response())
        driver = ollama_driver.OllamaDriver(endpoint="http://localhost:11434/api/generate", model="m")
        token = CancelToken()
        threading.Timer(0.15, token.cancel).start()
        with pytest.raises(TurnCancelled):
            GameHarness(Connect4(), driver=driver, cancel=token, cleanup_timeout=1.0).choose(Connect4Board())
        assert closed.is_set()

    def test_openai_chat_stream_is_closed_on_early_exit(self):
        pytest.importorskip("openai")
        from types import SimpleNamespace

        from prompture.drivers.openai_driver import OpenAIDriver

        class Stream:
            closed = False

            def __iter__(self):
                for ch in "abc":
                    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=ch))], usage=None)

            def close(self):
                Stream.closed = True

        driver = OpenAIDriver(api_key="sk-test-not-real", model="gpt-4o-mini")
        driver.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: Stream())))
        chunks = driver.generate_messages_stream([{"role": "user", "content": "hi"}], {})
        assert next(chunks)["text"] == "a"
        chunks.close()
        assert Stream.closed
