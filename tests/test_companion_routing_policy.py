"""Router cost records, presets, compatibility, fallback, escalation and task budgets."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prompture.companion import CompanionServer, LedgerSource, LiveBus, hook
from prompture.companion.calls import CallLog, RoutedCall, Usage, UsageSniffer, settle, summarize_calls
from prompture.companion.router import Router, Routes, normalize_routes, session_of
from prompture.companion.routing_policy import Needs, RoutePolicy, tier_of, tool_outcome
from prompture.companion.tool_routing import ToolRouting
from prompture.infra.capabilities import ProviderCapabilities, clear_overrides, override_capabilities

# ------------------------------------------------------------------ fakes


class _Vendor(BaseHTTPRequestHandler):
    """A vendor that reports usage like Anthropic does, or answers with ``status``."""

    seen: list[dict] = []
    status = 200
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        _Vendor.seen.append(
            {"path": self.path, "body": body, "headers": {k.lower(): v for k, v in self.headers.items()}}
        )
        if _Vendor.status != 200:
            raw = json.dumps({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}).encode()
            self.send_response(_Vendor.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        model = body.get("model")
        if body.get("stream"):
            start = {
                "type": "message_start",
                "message": {
                    "model": model,
                    "usage": {"input_tokens": 100, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0},
                },
            }
            frames = [
                f"event: message_start\ndata: {json.dumps(start)}\n\n",
                'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n',
                'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":50}}\n\n',
            ]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for frame in frames:
                data = frame.encode()
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.write(b"0\r\n\r\n")
            return
        raw = json.dumps(
            {"id": "msg_vendor", "model": model, "usage": {"input_tokens": 1000, "output_tokens": 50}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _Driver:
    """A Prompture driver stand-in; ``fail`` makes its stream raise before any text."""

    supports_streaming = True
    supports_tool_use = True

    def __init__(self, fail: bool = False, cost: float = 0.0):
        self.fail = fail
        self.cost = cost
        self.calls = 0

    def _meta(self):
        return {"prompt_tokens": 1000, "completion_tokens": 50, "cost": self.cost}

    def generate_messages_stream(self, messages, options):
        self.calls += 1
        if self.fail:
            raise ConnectionError("provider down")
        yield {"type": "delta", "text": "routed"}
        yield {"type": "done", "text": "routed", "meta": self._meta()}

    def generate_messages_with_tools_stream(self, messages, tools, options):
        yield from self.generate_messages_stream(messages, options)

    def generate_messages(self, messages, options):
        self.calls += 1
        if self.fail:
            raise ConnectionError("provider down")
        return {"text": "routed", "meta": self._meta()}

    generate_messages_with_tools = None


@pytest.fixture
def vendor():
    _Vendor.seen, _Vendor.status = [], 200
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Vendor)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def stack(tmp_path, vendor):
    """A companion with a router in front of the fake vendor; drivers by model name."""
    drivers: dict[str, _Driver] = {}
    bus = LiveBus()
    routes = Routes(tmp_path / "routes.json")
    projects = {"s-proj": "shop"}
    router = Router(
        bus,
        routes,
        upstream=lambda tool: vendor,
        driver_for=lambda model: drivers.setdefault(model, _Driver()),
        project_for=lambda agent, session: projects.get(session),
    )
    routing = ToolRouting(tmp_path / "prefs.json", claude_root=tmp_path / "claude", codex_root=tmp_path / "codex")
    srv = CompanionServer(
        LedgerSource(tmp_path / "none.db"), token="t0ken", bus=bus, state_path=None, tool_routing=routing, router=router
    )
    srv.start_background()
    yield srv, router, routes, drivers
    srv.shutdown()
    srv.shutdown_companion()
    clear_overrides()


def _post(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **(headers or {})}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.read()


def _get(url):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer t0ken"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _wait_calls(router, n):
    deadline = time.monotonic() + 5
    while len(router.calls.calls()) < n and time.monotonic() < deadline:
        time.sleep(0.02)
    calls = router.calls.calls()
    assert len(calls) >= n, calls
    return calls


PLAN = {"Authorization": "Bearer sk-ant-oat01-secret"}
KEY = {"x-api-key": "sk-ant-api-secret"}


def _msg(model, text="fix the bug", *, session="s1", stream=True, **extra):
    return {
        "model": model,
        "stream": stream,
        "metadata": {"user_id": f"user_abc_account_def_session_{session}"},
        "messages": [{"role": "user", "content": text}],
        **extra,
    }


def _tool_turn(model, output, *, session="s1", is_error=False, command="npm test"):
    return {
        "model": model,
        "stream": True,
        "metadata": {"user_id": f"user_abc_account_def_session_{session}"},
        "messages": [
            {"role": "user", "content": "fix the bug"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": command}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": output, "is_error": is_error}],
            },
        ],
    }


# ------------------------------------------------------------------ usage and cost


def test_usage_is_read_from_anthropic_and_responses_replies():
    sniff = UsageSniffer("anthropic")
    start = {
        "type": "message_start",
        "message": {"model": "claude-sonnet-5", "usage": {"input_tokens": 10, "cache_read_input_tokens": 90}},
    }
    raw = f"event: message_start\ndata: {json.dumps(start)}\n\n".encode()
    sniff.feed(raw[:30], stream=True)  # split across chunks
    sniff.feed(
        raw[30:] + b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":7}}\n\n', stream=True
    )
    usage = sniff.finish()
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens, usage.model) == (
        100,
        90,
        7,
        "claude-sonnet-5",
    )

    sniff = UsageSniffer("openai")
    done = {
        "type": "response.completed",
        "response": {
            "model": "gpt-5.5",
            "usage": {"input_tokens": 50, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 40}},
        },
    }
    sniff.feed(f"event: response.completed\ndata: {json.dumps(done)}\n\n".encode(), stream=True)
    usage = sniff.finish()
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (50, 40, 5)

    sniff = UsageSniffer("anthropic")
    sniff.feed(
        json.dumps({"model": "claude-haiku-4-5", "usage": {"input_tokens": 3, "output_tokens": 4}}).encode(),
        stream=False,
    )
    assert sniff.finish().input_tokens == 3


def _call(route, billing, original, requested="claude/claude-sonnet-5", served="claude/claude-sonnet-5"):
    return RoutedCall(
        id="c",
        ts="2026-09-27T00:00:00+00:00",
        tool="claude-code",
        kind="main",
        endpoint="/v1/messages",
        requested=requested,
        served=served,
        route=route,
        rule={},
        billing=billing,
        original_billing=original,
    )


def test_subscription_traffic_is_kept_apart_from_api_spend():
    usage = Usage(input_tokens=1_000_000, output_tokens=100_000)
    # On an API key, passing through costs what it costs, and saves nothing.
    api = settle(_call("passthrough", "api", "api"), usage, vendor="claude")
    assert api.cost_usd > 0 and api.baseline_usd == api.cost_usd and api.savings_usd == 0
    # On a plan it adds no bill; the API price is only context.
    plan = settle(_call("passthrough", "subscription", "subscription"), usage, vendor="claude")
    assert plan.cost_usd == 0 and plan.baseline_usd == 0 and plan.plan_equivalent_usd == api.cost_usd
    # Routing API traffic to a local model saves the whole estimate.
    local = settle(_call("routed", "local", "api", served="ollama/qwen3"), usage, vendor="claude")
    assert local.cost_usd == 0 and local.savings_usd == local.baseline_usd == api.cost_usd
    # Routing plan traffic to a paid API is new spend, never savings.
    paid = settle(_call("routed", "api", "subscription", served="claude/claude-haiku-4-5"), usage, vendor="claude")
    assert paid.cost_usd > 0 and paid.new_spend_usd == paid.cost_usd and paid.savings_usd == -paid.cost_usd


def test_savings_assume_the_original_path_kept_its_cache():
    usage = Usage(input_tokens=1_000_000, output_tokens=0)
    cold = settle(_call("routed", "local", "api", served="ollama/qwen3"), usage, vendor="claude")
    warm = settle(_call("routed", "local", "api", served="ollama/qwen3"), usage, vendor="claude", cache_ratio=0.9)
    assert warm.baseline_usd < cold.baseline_usd / 3  # cached input is a fraction of the price


def test_calls_are_summed_per_tool_project_and_rule(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")
    a = _call("routed", "local", "api", served="ollama/qwen3")
    a.id, a.project, a.rule = "a", "shop", {"source": "kind", "match": "background"}
    a.baseline_usd, a.savings_usd = 1.0, 1.0
    b = _call("passthrough", "api", "api")
    b.id, b.cost_usd, b.baseline_usd = "b", 2.0, 2.0
    log.add(a)
    log.add(b)
    again = CallLog(tmp_path / "calls.jsonl")  # read back from disk
    summary = summarize_calls(again.calls(), a.when)
    assert summary["total"]["calls"] == 2 and summary["total"]["savings_usd"] == 1.0
    assert {r["rule"] for r in summary["by_rule"]} == {"kind: background", "none"}
    assert next(r for r in summary["by_project"] if r["project"] == "shop")["routed"] == 1


def test_sessions_come_from_what_each_cli_sends():
    from prompture.companion.router import TOOLS

    claude, codex = TOOLS["claude-code"], TOOLS["codex"]
    assert session_of(claude, {}, {"metadata": {"user_id": "user_x_account_y_session_1234abcd-ef"}}) == "1234abcd-ef"
    assert session_of(claude, {}, {"metadata": {"user_id": json.dumps({"session_id": "js"})}}) == "js"
    assert session_of(claude, {"x-claude-code-session-id": "hdr"}, {}) == "hdr"
    assert session_of(codex, {"session_id": "cx"}, {}) == "cx"
    assert session_of(codex, {}, {"prompt_cache_key": "pk"}) == "pk"


def test_passed_through_calls_record_usage_cost_and_billing(stack):
    srv, router, _, _ = stack
    _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-sonnet-5"), KEY)
    (call,) = _wait_calls(router, 1)
    assert (call.route, call.billing, call.session, call.served) == (
        "passthrough",
        "api",
        "s1",
        "claude/claude-sonnet-5",
    )
    assert (call.input_tokens, call.cache_read_tokens, call.output_tokens) == (1000, 900, 50)
    assert call.cost_usd > 0 and call.ttft_ms is not None
    ended = [e for e in srv.bus.replay(0) if e["type"] == "request.ended"][-1]
    assert ended["call_id"] == call.id and ended["route"] == "passthrough"


# ------------------------------------------------------------------ presets and rules


def test_presets_use_the_vendors_own_smaller_models_and_never_upgrade(stack):
    srv, router, routes, drivers = stack
    routes.save({"tools": {"claude-code": {"preset": "economy"}}})
    _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-opus-5-5", session="e1"), PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-sonnet-5"  # one tier down, same login
    assert _Vendor.seen[-1]["headers"]["authorization"] == PLAN["Authorization"]
    _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-haiku-4-5", session="e2"), PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-haiku-4-5"  # never up
    probe = {"model": "claude-opus-5-5", "max_tokens": 1, "messages": [{"role": "user", "content": "quota"}]}
    _post(f"{srv.url}/tools/claude-code/v1/messages", probe, PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-opus-5-5"  # quota checks are the vendor's
    assert drivers == {}

    calls = _wait_calls(router, 3)
    first = calls[0]
    assert first.route == "native" and first.served == "claude/claude-sonnet-5"
    assert first.rule["source"] == "preset" and "Economy" in first.rule["reason"]
    assert first.original_billing == "subscription" and first.cost_usd == 0  # still the plan


def test_project_settings_win_over_the_tools(stack):
    srv, router, routes, _ = stack
    routes.save({"tools": {"claude-code": {"preset": "economy"}}, "projects": {"shop": {"preset": "quality"}}})
    _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-opus-5-5", session="s-proj"), PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-opus-5-5"
    (call,) = _wait_calls(router, 1)
    assert call.project == "shop" and call.rule["preset"] == "quality"


def test_a_task_keeps_its_destination_through_its_tool_loop(stack):
    srv, router, routes, drivers = stack
    url = f"{srv.url}/tools/claude-code/v1/messages"
    routes.save({"tools": {"claude-code": {"kinds": {"main": "ollama/qwen3"}}}})
    _post(url, _msg("claude-sonnet-5", session="t1"), KEY)
    routes.save({"tools": {"claude-code": {"kinds": {"main": "ollama/other"}}}})
    _post(url, _tool_turn("claude-sonnet-5", "ok", session="t1"), KEY)  # same tool loop: same model
    assert drivers["ollama/qwen3"].calls == 2 and "ollama/other" not in drivers
    _post(url, _msg("claude-sonnet-5", "next thing", session="t1"), KEY)  # the next prompt: new rules
    assert drivers["ollama/other"].calls == 1
    calls = _wait_calls(router, 3)
    assert "Kept for this task" in calls[1].rule["reason"]
    assert calls[0].savings_usd > 0 and calls[0].cost_usd == 0  # a local model on an API key


def test_a_pin_never_overrides_a_model_the_user_switched_to(stack):
    srv, _, routes, _ = stack
    url = f"{srv.url}/tools/claude-code/v1/messages"
    routes.save({"tools": {"claude-code": {"preset": "economy"}}})
    _post(url, _msg("claude-opus-5-5", session="sw"), PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-sonnet-5"
    _post(url, _tool_turn("claude-haiku-4-5", "ok", session="sw"), PLAN)  # /model haiku mid-task
    assert _Vendor.seen[-1]["body"]["model"] == "claude-haiku-4-5"


def test_incompatible_models_are_skipped_with_the_reason(stack):
    srv, router, routes, drivers = stack
    override_capabilities("fake/text-only", ProviderCapabilities(tool_use=False))
    routes.save({"tools": {"claude-code": {"kinds": {"main": "fake/text-only"}}}})
    body = _msg("claude-sonnet-5", session="c1", tools=[{"name": "Bash", "input_schema": {"type": "object"}}])
    _post(f"{srv.url}/tools/claude-code/v1/messages", body, KEY)
    (call,) = _wait_calls(router, 1)
    assert call.route == "passthrough" and "can't call tools" in call.rule["reason"]
    assert drivers == {}


def test_chained_responses_stay_on_their_own_backend():
    needs = Needs.of("openai", {"previous_response_id": "resp_1", "input": []})
    policy = RoutePolicy(lambda: {"tools": {"codex": {"kinds": {"main": "ollama/qwen3"}}}})
    decision = policy.decide("codex", "openai", "gpt-5.5", "main", needs=needs)
    assert decision.route == "passthrough" and "previous_response_id" in decision.reason


def test_routes_json_keeps_only_known_settings():
    clean = normalize_routes(
        {
            "preset": "economy",
            "tools": {"codex": {"preset": "nonsense", "kinds": {"main": "ollama/x"}}},
            "projects": {
                "shop": {"preset": "quality", "tools": {"codex": {"preset": "balanced"}}},
                "": {"preset": "x"},
            },
            "fallback": {"models": ["auto/cheap", 3], "allow_paid": "yes"},
            "escalation": {"after_failures": 2, "to": "native:large", "junk": 1},
            "budget": {"task_usd": 1.5, "task_attempts": -1, "on_exceed": "stop"},
        }
    )
    assert clean == {
        "tools": {"codex": {"models": {}, "kinds": {"main": "ollama/x"}}},
        "preset": "economy",
        "projects": {"shop": {"preset": "quality", "tools": {"codex": {"preset": "balanced"}}}},
        "fallback": {"models": ["auto/cheap"], "allow_paid": True},
        "escalation": {"after_failures": 2, "to": "native:large"},
        "budget": {"task_usd": 1.5, "on_exceed": "stop"},
    }


def test_tiers_come_from_model_names():
    assert tier_of("anthropic", "claude-haiku-4-5-20251001") == "small"
    assert tier_of("anthropic", "claude-opus-5-5") == "large"
    assert tier_of("openai", "gpt-5.1-codex-mini") == "small"
    assert tier_of("openai", "gpt-5.5") == "mid"
    assert tier_of("openai", "ollama-thing") is None


# ------------------------------------------------------------------ fallback


def test_a_failed_route_falls_back_before_the_cli_sees_anything(stack):
    srv, router, routes, drivers = stack
    drivers["ollama/down"] = _Driver(fail=True)
    routes.save({"tools": {"claude-code": {"kinds": {"main": "ollama/down"}}}, "fallback": {"models": ["ollama/up"]}})
    status, raw = _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-sonnet-5", session="f1"), KEY)
    assert status == 200 and b"routed" in raw and b"provider down" not in raw
    (call,) = _wait_calls(router, 1)
    assert call.served == "ollama/up" and [a["model"] for a in call.attempts] == ["ollama/down", "ollama/up"]
    assert call.rule["source"] == "fallback" and "ollama/down failed" in call.rule["reason"]


def test_a_rate_limited_vendor_falls_back_only_where_allowed(stack):
    srv, router, routes, _drivers = stack
    _Vendor.status = 429
    routes.save({"fallback": {"models": ["openai/gpt-5.5-mini"]}})
    # Plan traffic isn't moved onto a paid API unless that's allowed.
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-sonnet-5", session="r1"), PLAN)
    assert err.value.code == 429
    routes.save({"fallback": {"models": ["openai/gpt-5.5-mini"], "allow_paid": True}})
    status, raw = _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-sonnet-5", session="r2"), PLAN)
    assert status == 200 and b"routed" in raw
    call = _wait_calls(router, 2)[-1]
    assert call.attempts[0]["error"] == "HTTP 429" and call.served == "openai/gpt-5.5-mini"
    assert call.original_billing == "subscription" and call.billing == "api"


# ------------------------------------------------------------------ escalation and budgets


def test_failing_tool_results_escalate_the_task_back_to_the_requested_model(stack):
    srv, router, routes, _ = stack
    routes.save({"tools": {"claude-code": {"preset": "economy"}}})
    url = f"{srv.url}/tools/claude-code/v1/messages"
    _post(url, _msg("claude-opus-5-5", session="x1"), PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-sonnet-5"
    _post(url, _tool_turn("claude-opus-5-5", "3 failed, 10 passed", session="x1", command="npm test -- a"), PLAN)
    _post(
        url,
        _tool_turn(
            "claude-opus-5-5", "The user doesn't want to proceed with this tool use.", session="x1", is_error=True
        ),
        PLAN,
    )
    _post(url, _tool_turn("claude-opus-5-5", "Error: build broke", session="x1", command="npm run build"), PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-sonnet-5"  # the declined tool reset the count
    _post(url, _tool_turn("claude-opus-5-5", "Error: build broke", session="x1", command="npm run build"), PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-opus-5-5"  # the same failing action twice
    calls = _wait_calls(router, 5)
    call = calls[-1]
    assert call.rule["source"] == "escalation" and "same action failed 2 times" in call.rule["reason"]
    assert call.escalated and [c.tool_failed for c in calls] == [None, True, False, True, True]
    economy = next(r for r in _get(f"{srv.url}/v1/router/savings")["by_preset"] if r["preset"] == "economy")
    assert (economy["tool_results"], economy["tool_failures"], economy["tool_success"]) == (3, 2, 0.3333)
    escalation = next(r for r in _get(f"{srv.url}/v1/router/savings")["by_preset"] if r["preset"] == "none")
    assert escalation["escalations"] == 1
    tasks = _get(f"{srv.url}/v1/router/tasks")
    assert tasks[0]["session"] == "x1" and tasks[0]["attempts"] == 1 and len(tasks[0]["escalations"]) == 1


def test_a_review_can_escalate_a_task(stack):
    srv, _router, routes, _ = stack
    routes.save({"tools": {"claude-code": {"preset": "economy"}}})
    _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-opus-5-5", session="rv"), PLAN)
    req = urllib.request.Request(
        f"{srv.url}/v1/router/sessions/claude-code/rv/escalate",
        data=json.dumps({"reason": "tests still fail"}).encode(),
        headers={"Authorization": "Bearer t0ken", "Content-Type": "application/json"},
    )
    body = json.loads(urllib.request.urlopen(req, timeout=5).read())
    assert body["escalated"] and "tests still fail" in body["task"]["escalations"][0]["reason"]
    _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-opus-5-5", session="rv"), PLAN)
    assert _Vendor.seen[-1]["body"]["model"] == "claude-opus-5-5"


def test_task_budgets_bound_attempts_and_spend(stack):
    srv, router, routes, drivers = stack
    drivers["openai/gpt-paid"] = _Driver(cost=0.6)
    routes.save({"tools": {"claude-code": {"kinds": {"main": "openai/gpt-paid"}}}, "budget": {"task_usd": 1.0}})
    url = f"{srv.url}/tools/claude-code/v1/messages"
    for _ in range(2):
        _post(url, _msg("claude-sonnet-5", session="b1"), KEY)
    _post(url, _msg("claude-sonnet-5", session="b1"), KEY)
    assert drivers["openai/gpt-paid"].calls == 2
    last = _wait_calls(router, 3)[-1]
    assert last.route == "passthrough" and "Task budget reached ($1.20 of $1.00)" in last.rule["reason"]

    routes.save(
        {
            "tools": {"claude-code": {"kinds": {"main": "openai/gpt-paid"}}},
            "budget": {"task_usd": 0.5, "on_exceed": "stop"},
        }
    )
    _post(url, _msg("claude-sonnet-5", session="b2"), KEY)
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(url, _msg("claude-sonnet-5", session="b2"), KEY)
    assert err.value.code == 429 and b"Task budget reached" in err.value.read()


def test_tool_outcomes_tell_failures_from_user_decisions():
    failed = _tool_turn("m", "FAILED tests/test_x.py::test_y")
    assert tool_outcome("anthropic", failed).failed
    declined = _tool_turn("m", "The user doesn't want to proceed with this tool use.", is_error=True)
    assert tool_outcome("anthropic", declined).user_decision
    codex = {
        "input": [
            {"type": "function_call", "call_id": "c", "name": "shell", "arguments": '{"cmd": "pytest"}'},
            {"type": "function_call_output", "call_id": "c", "output": '{"output": "", "metadata": {"exit_code": 1}}'},
        ]
    }
    outcome = tool_outcome("openai", codex)
    assert outcome.failed and outcome.name == "shell"
    assert not tool_outcome("anthropic", _tool_turn("m", "all 12 tests passed")).failed


# ------------------------------------------------------------------ endpoints and hooks


def test_calls_and_savings_endpoints(stack):
    srv, router, routes, _ = stack
    routes.save({"tools": {"claude-code": {"kinds": {"main": "ollama/qwen3"}}}})
    _post(f"{srv.url}/tools/claude-code/v1/messages", _msg("claude-sonnet-5", session="s-proj"), KEY)
    (call,) = _wait_calls(router, 1)
    listed = _get(f"{srv.url}/v1/router/calls?period=day&routed=true")
    assert [c["id"] for c in listed] == [call.id]
    detail = _get(f"{srv.url}/v1/router/calls/{call.id}")
    assert detail["rule"]["source"] == "kind" and detail["task"]["session"] == "s-proj"
    savings = _get(f"{srv.url}/v1/router/savings?period=week")
    assert savings["total"]["routed"] == 1 and savings["by_project"][0]["project"] == "shop"
    state = _get(f"{srv.url}/v1/router")
    assert set(state["presets"]) == {"quality", "balanced", "economy"} and "probe" not in state["background_kinds"]


def test_hooks_carry_the_project_folder_and_mark_permission_waits(tmp_path):
    body = hook.payload(
        "claude", json.dumps({"hook_event_name": "Notification", "session_id": "s", "cwd": "C:\\work\\shop\\"})
    )
    assert body == {"agent": "claude", "event": "Notification", "session": "s", "project": "shop"}
    router = Router(LiveBus(), Routes(None))
    router.policy.task("claude-code", "s")
    router.hook("claude", "Notification", "s", "shop")
    task = router.policy.find("claude-code", "s")
    assert task.waiting and task.project == "shop"
    router.hook("claude", "PostToolUse", "s")
    assert not task.waiting
