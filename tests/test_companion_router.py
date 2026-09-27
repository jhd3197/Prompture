"""The companion's router: CLI traffic passed through or routed, config switches, and hooks."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from prompture.companion import CodingToolSource, CompanionServer, LedgerSource, LiveBus, hook
from prompture.companion.router import Router, Routes, anthropic_kind, responses_kind
from prompture.companion.tool_routing import RoutingError, ToolRouting
from prompture.infra.coding_agent_activity import ActiveTurn, AgentActivity
from prompture.infra.coding_agent_readers import ClaudeCodeReader

# ------------------------------------------------------------------ fakes


class _Upstream(BaseHTTPRequestHandler):
    """A vendor stand-in: records each request and answers with SSE or JSON."""

    seen: list[dict] = []
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _Upstream.seen.append(
            {
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": json.loads(body or b"{}"),
            }
        )
        if json.loads(body or b"{}").get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for frame in (b"event: message_start\ndata: {}\n\n", b"event: content_block_delta\ndata: {}\n\n"):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(frame), frame))
            self.wfile.write(b"0\r\n\r\n")
            return
        raw = json.dumps({"id": "msg_vendor", "content": [{"type": "text", "text": "hi"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST


class _Driver:
    """A Prompture driver stand-in that streams a fixed answer."""

    supports_streaming = True
    supports_tool_use = False

    def __init__(self):
        self.calls: list[list[dict]] = []

    def generate_messages_stream(self, messages, options):
        self.calls.append(messages)
        yield {"type": "delta", "text": "routed "}
        yield {"type": "delta", "text": "answer"}
        yield {"type": "done", "text": "routed answer", "meta": {"prompt_tokens": 5, "completion_tokens": 2}}

    def generate_messages(self, messages, options):
        self.calls.append(messages)
        return {"text": "routed answer", "meta": {"prompt_tokens": 5, "completion_tokens": 2}}


@pytest.fixture
def upstream():
    _Upstream.seen = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def companion(tmp_path, upstream):
    driver = _Driver()
    routes = Routes(tmp_path / "routes.json")
    bus = LiveBus()
    router = Router(bus, routes, upstream=lambda tool: upstream, driver_for=lambda model: driver)
    routing = ToolRouting(tmp_path / "prefs.json", claude_root=tmp_path / "claude", codex_root=tmp_path / "codex")
    srv = CompanionServer(
        LedgerSource(tmp_path / "none.db"),
        token="t0ken",
        bus=bus,
        state_path=None,
        tool_routing=routing,
        router=router,
    )
    srv.start_background()
    yield srv, driver, routes
    srv.shutdown()
    srv.shutdown_companion()


def _calls(router, n=1, timeout=5.0):
    """The router's recorded calls once there are ``n`` (they're written just after the reply)."""
    import time

    deadline = time.monotonic() + timeout
    while len(router.calls.calls()) < n and time.monotonic() < deadline:
        time.sleep(0.02)
    return router.calls.calls()


def _post(url: str, body: dict, headers: dict | None = None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **(headers or {})}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.headers.get("Content-Type"), resp.read()


# ------------------------------------------------------------------ request kinds


def test_request_kinds():
    assert anthropic_kind({"max_tokens": 1, "messages": [{"role": "user", "content": "quota"}]}) == "probe"
    tool_result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}
    assert anthropic_kind({"messages": [tool_result]}) == "tool_result"
    assert (
        anthropic_kind({"system": "Generate a 5-10 word title for it", "messages": [{"role": "user", "content": "x"}]})
        == "title"
    )
    compact = "Your task is to create a detailed summary of the conversation so far"
    assert anthropic_kind({"messages": [{"role": "user", "content": compact}]}) == "compaction"
    assert anthropic_kind({"messages": [{"role": "user", "content": "fix the bug"}]}) == "main"

    assert responses_kind({"input": [{"type": "function_call_output", "output": "ok"}]}) == "tool_result"
    assert responses_kind({"input": [{"type": "message", "role": "user"}]}) == "main"
    assert responses_kind({}, "/v1/responses/compact") == "compaction"


def test_routes_prefer_kind_rules_then_model_patterns(tmp_path):
    routes = Routes(tmp_path / "routes.json")
    assert routes.target("claude-code", "claude-haiku-4-5", "main") is None
    routes.save(
        {
            "tools": {
                "claude-code": {
                    "models": {"claude-haiku-*": "ollama/qwen3", "claude-opus-*": "passthrough", "": "x"},
                    "kinds": {"background": "ollama/tiny", "nonsense": "y"},
                },
                "bad": "rules",
            }
        }
    )
    saved = json.loads((tmp_path / "routes.json").read_text())
    assert saved == {
        "tools": {
            "claude-code": {
                "models": {"claude-haiku-*": "ollama/qwen3", "claude-opus-*": "passthrough"},
                "kinds": {"background": "ollama/tiny"},
            }
        }
    }
    assert routes.target("claude-code", "claude-haiku-4-5", "main") == "ollama/qwen3"
    assert routes.target("claude-code", "claude-opus-5-5", "title") == "ollama/tiny"  # a kind rule wins
    assert routes.target("claude-code", "claude-opus-5-5", "main") is None  # passthrough
    assert routes.target("codex", "gpt-5.5", "main") is None


# ------------------------------------------------------------------ pass through


def test_passthrough_forwards_the_clis_own_login_and_streams_back(companion):
    srv, driver, _ = companion
    status, ctype, raw = _post(
        f"{srv.url}/tools/claude-code/v1/messages?beta=true",
        {"model": "claude-opus-5-5", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
        {"Authorization": "Bearer sk-ant-oat-secret", "anthropic-beta": "oauth-2025-04-20"},
    )
    assert status == 200 and ctype == "text/event-stream"
    assert raw == b"event: message_start\ndata: {}\n\nevent: content_block_delta\ndata: {}\n\n"  # de-chunked
    (seen,) = _Upstream.seen
    assert seen["path"] == "/v1/messages?beta=true"
    assert seen["headers"]["authorization"] == "Bearer sk-ant-oat-secret"
    assert seen["headers"]["anthropic-beta"] == "oauth-2025-04-20"
    assert driver.calls == []

    types = [e["type"] for e in srv.bus.replay(0)]
    assert types == ["request.started", "request.first_token", "request.ended"]
    started = srv.bus.replay(0)[0]
    assert (started["tool"], started["key_name"], started["model"], started["kind"]) == (
        "claude",
        "Claude Code",
        "claude/claude-opus-5-5",
        "main",
    )
    assert srv.bus.running() == []


def test_passthrough_of_non_model_calls_publishes_nothing(companion):
    srv, _, _ = companion
    status, _, raw = _post(f"{srv.url}/tools/claude-code/v1/messages/count_tokens", {"model": "claude-opus-5-5"})
    assert status == 200 and json.loads(raw)["id"] == "msg_vendor"
    assert srv.bus.replay(0) == []


def test_codex_signed_in_with_chatgpt_goes_to_the_chatgpt_backend(tmp_path):
    router = Router(LiveBus(), Routes(None))
    tool = __import__("prompture.companion.router", fromlist=["TOOLS"]).TOOLS["codex"]
    assert router.upstream_url(tool, "/v1/responses", {"chatgpt-account-id": "acc"}) == (
        "https://chatgpt.com/backend-api/codex/responses"
    )
    assert router.upstream_url(tool, "/v1/responses", {}) == "https://api.openai.com/v1/responses"


def test_the_router_refuses_web_pages_and_other_hosts(companion):
    srv, _, _ = companion
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{srv.url}/tools/claude-code/v1/messages", {"model": "m"}, {"Origin": "https://evil.example"})
    assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{srv.url}/tools/claude-code/v1/messages", {"model": "m"}, {"Host": "evil.example"})
    assert err.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{srv.url}/tools/nope/v1/messages", {"model": "m"})
    assert err.value.code == 404
    assert _Upstream.seen == []


# ------------------------------------------------------------------ routed to a Prompture model


def test_a_matching_rule_sends_the_request_to_a_prompture_model(companion):
    srv, driver, routes = companion
    routes.save({"tools": {"claude-code": {"models": {"claude-haiku-*": "ollama/qwen3"}}}})
    status, ctype, raw = _post(
        f"{srv.url}/tools/claude-code/v1/messages",
        {"model": "claude-haiku-4-5", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert status == 200 and ctype == "text/event-stream"
    text = raw.decode()
    assert '"text":"routed "' in text and '"text":"answer"' in text
    assert "msg_prompture_" in text
    assert "event: message_stop" in text
    assert _Upstream.seen == []
    assert driver.calls[0][-1]["content"] == "hi"
    started = srv.bus.replay(0)[0]
    assert started["routed_to"] == "ollama/qwen3"
    assert [e["type"] for e in srv.bus.replay(0)][-1] == "request.ended"

    # Not streamed, and in Codex's Responses format.
    routes.save({"tools": {"codex": {"kinds": {"main": "ollama/qwen3"}}}})
    status, _, raw = _post(
        f"{srv.url}/tools/codex/v1/responses",
        {"model": "gpt-5.5-codex", "input": [{"type": "message", "role": "user", "content": "hello"}]},
    )
    body = json.loads(raw)
    assert body["id"].startswith("resp_prompture_") and body["output"][0]["content"][0]["text"] == "routed answer"


def test_an_unusable_route_falls_back_to_the_vendor(tmp_path, upstream):
    def broken(model):
        raise ValueError("no such model")

    bus = LiveBus()
    routes = Routes(tmp_path / "routes.json")
    routes.save({"tools": {"claude-code": {"models": {"*": "nope/model"}}}})
    router = Router(bus, routes, driver_for=broken, upstream=lambda tool: upstream)
    srv = CompanionServer(LedgerSource(tmp_path / "none.db"), bus=bus, state_path=None, router=router)
    srv.start_background()
    try:
        status, _, raw = _post(f"{srv.url}/tools/claude-code/v1/messages", {"model": "claude-x", "messages": []})
        assert status == 200 and json.loads(raw)["id"] == "msg_vendor"  # the CLI never saw the failure
        (call,) = _calls(router)
        assert call.served == "claude/claude-x" and call.route == "passthrough"
        assert [a["model"] for a in call.attempts] == ["nope/model", "claude/claude-x"]
        assert call.attempts[0]["status"] == "error" and "no such model" in call.attempts[0]["error"]
        assert call.rule["source"] == "fallback" and "fell back to claude-x" in call.rule["reason"]
        assert bus.running() == []
    finally:
        srv.shutdown()
        srv.shutdown_companion()


def test_log_readers_skip_replies_prompture_produced(tmp_path):
    reader = ClaudeCodeReader(tmp_path)
    entry = {
        "type": "assistant",
        "timestamp": "2026-09-26T10:00:00Z",
        "message": {"id": "msg_prompture_abc", "model": "claude-haiku-4-5", "usage": {"output_tokens": 5}},
    }
    assert reader._call(entry) is None
    entry["message"]["id"] = "msg_real"
    assert reader._call(entry) is not None


# ------------------------------------------------------------------ switches


def test_claude_code_routing_keeps_other_settings_and_restores_the_previous_gateway(tmp_path):
    routing = ToolRouting(None, claude_root=tmp_path, codex_root=tmp_path / "codex")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model": "opus", "env": {"ANTHROPIC_BASE_URL": "https://gw.example", "X": "1"}}))

    routing.set_enabled("claude-code", True, "http://127.0.0.1:47811")
    data = json.loads(settings.read_text())
    assert data["env"] == {"ANTHROPIC_BASE_URL": "http://127.0.0.1:47811/tools/claude-code", "X": "1"}
    assert data["model"] == "opus"
    assert routing.routed("claude-code") and routing.enabled() == ["claude-code"]
    assert routing.upstream("claude-code") == "https://gw.example"

    routing.apply_enabled("http://127.0.0.1:50000")  # a new port on the next start
    assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"].endswith(":50000/tools/claude-code")
    assert routing.upstream("claude-code") == "https://gw.example"

    routing.restore_all()  # the companion stopping
    assert json.loads(settings.read_text())["env"]["ANTHROPIC_BASE_URL"] == "https://gw.example"
    assert routing.enabled() == ["claude-code"]

    routing.apply_enabled("http://127.0.0.1:47811")
    routing.set_enabled("claude-code", False, "http://127.0.0.1:47811")
    assert json.loads(settings.read_text()) == {
        "model": "opus",
        "env": {"ANTHROPIC_BASE_URL": "https://gw.example", "X": "1"},
    }
    assert routing.enabled() == []


def test_claude_code_routing_leaves_a_broken_settings_file_alone(tmp_path):
    routing = ToolRouting(None, claude_root=tmp_path, codex_root=tmp_path)
    (tmp_path / "settings.json").write_text("{ not json")
    with pytest.raises(RoutingError):
        routing.set_enabled("claude-code", True, "http://127.0.0.1:1")
    assert (tmp_path / "settings.json").read_text() == "{ not json"


def test_codex_routing_edits_only_its_own_lines(tmp_path):
    codex = tmp_path / "codex"
    codex.mkdir()
    config = codex / "config.toml"
    original = 'model = "gpt-5.5-codex"\nmodel_provider = "azure"\n\n[projects."C:\\\\code"]\ntrust_level = "trusted"\n'
    config.write_text(original)
    routing = ToolRouting(None, claude_root=tmp_path, codex_root=codex)

    routing.set_enabled("codex", True, "http://127.0.0.1:47811")
    text = config.read_text()
    assert text.splitlines()[0] == 'model_provider = "prompture"  # prompture-router'
    assert 'model_provider = "azure"' not in text
    assert 'base_url = "http://127.0.0.1:47811/tools/codex/v1"' in text
    assert "requires_openai_auth = true" in text
    assert 'trust_level = "trusted"' in text
    tomllib = pytest.importorskip("tomllib")
    parsed = tomllib.loads(text)
    assert parsed["model_provider"] == "prompture"
    assert parsed["model_providers"]["prompture"]["wire_api"] == "responses"
    assert routing.routed("codex")

    routing.apply_enabled("http://127.0.0.1:50000")
    assert text.count("prompture-router: begin") == config.read_text().count("prompture-router: begin") == 1

    routing.set_enabled("codex", False, "")
    restored = tomllib.loads(config.read_text())
    assert restored["model_provider"] == "azure" and restored["model"] == "gpt-5.5-codex"
    assert "model_providers" not in restored
    assert not routing.routed("codex")


def test_hooks_install_and_remove_only_prompture_entries(tmp_path):
    routing = ToolRouting(None, claude_root=tmp_path, codex_root=tmp_path, python="C:\\py\\python.exe")
    mine = {"hooks": [{"type": "command", "command": "notify-send done"}]}
    (tmp_path / "settings.json").write_text(json.dumps({"hooks": {"Stop": [mine]}}))

    routing.set_hooks(True)
    hooks = json.loads((tmp_path / "settings.json").read_text())["hooks"]
    assert set(hooks) == {"UserPromptSubmit", "PostToolUse", "Notification", "Stop", "SessionEnd"}
    assert hooks["Stop"][0] == mine
    command = hooks["Stop"][1]["hooks"][0]["command"]
    assert command.startswith('"C:/py/python.exe" "') and command.endswith('/prompture/companion/hook.py" claude')
    assert hooks["PostToolUse"][0]["matcher"] == "*"
    assert routing.hooks_installed()

    routing.set_hooks(True)  # idempotent
    assert len(json.loads((tmp_path / "settings.json").read_text())["hooks"]["Stop"]) == 2

    routing.set_hooks(False)
    assert json.loads((tmp_path / "settings.json").read_text()) == {"hooks": {"Stop": [mine]}}
    assert not routing.hooks_installed()


def test_hooks_from_an_older_prompture_are_brought_up_to_date(tmp_path):
    old = {"hooks": [{"type": "command", "command": '"C:/old/python.exe" -m prompture.companion.hook claude'}]}
    mine = {"hooks": [{"type": "command", "command": "notify-send done"}]}
    (tmp_path / "settings.json").write_text(json.dumps({"hooks": {"Stop": [mine, old], "Notification": [old]}}))
    routing = ToolRouting(None, claude_root=tmp_path, codex_root=tmp_path, python=r"C:\py\python.exe")
    assert routing.hooks_installed() and routing.refresh_hooks()
    hooks = json.loads((tmp_path / "settings.json").read_text())["hooks"]
    assert hooks["Stop"][0] == mine and len(hooks["Stop"]) == 2
    assert all(g["hooks"][0]["command"] == routing.hook_command for e in hooks.values() for g in e if g != mine)
    assert not routing.refresh_hooks()  # already current


def test_router_endpoints_switch_routing_and_save_rules(companion, tmp_path):
    srv, _, _ = companion
    auth = {"Authorization": "Bearer t0ken"}
    info = json.loads(urllib.request.urlopen(f"{srv.url}/v1/companion/info", timeout=5).read())
    assert info["capabilities"]["router"] is True and info["capabilities"]["running_calls"] is True

    _, _, raw = _post(f"{srv.url}/v1/router/tools/claude-code", {"enabled": True}, auth)
    state = json.loads(raw)
    claude = next(t for t in state["tools"] if t["id"] == "claude-code")
    assert claude["enabled"] and claude["routed"] and claude["url"] == f"{srv.url}/tools/claude-code"

    _, _, raw = _post(
        f"{srv.url}/v1/router/routes", {"tools": {"claude-code": {"kinds": {"background": "ollama/qwen3"}}}}, auth
    )
    assert json.loads(raw)["routes"]["tools"]["claude-code"]["kinds"] == {"background": "ollama/qwen3"}

    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{srv.url}/v1/router/tools/claude-code", {"enabled": True})  # no token
    assert err.value.code == 401

    srv.tool_routing.restore_all()  # what the companion does when it stops
    assert not (tmp_path / "claude" / "settings.json").exists()  # it only existed for routing


# ------------------------------------------------------------------ hooks


def test_hook_payload_keeps_only_the_event_session_and_folder():
    raw = json.dumps({"hook_event_name": "Notification", "session_id": "s1", "message": "secret", "cwd": "/home/me/x"})
    assert hook.payload("claude", raw) == {
        "agent": "claude",
        "event": "Notification",
        "session": "s1",
        "project": "x",
        "cwd": "/home/me/x",
    }
    prompt = json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "s1", "prompt": "fix the build"})
    assert hook.payload("claude", prompt)["prompt"] == "fix the build"  # only a prompt event carries it
    assert hook.payload("claude", "not json") is None
    assert hook.payload("claude", json.dumps({"hook_event_name": "Stop"})) is None


def test_hook_main_never_fails_without_a_companion(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "STATE_FILE", tmp_path / "missing.json")
    monkeypatch.setattr("sys.stdin", io.StringIO('{"hook_event_name": "Stop", "session_id": "s"}'))
    assert hook.main(["claude"]) == 0


class _FixedActivity(AgentActivity):
    def __init__(self, turns):
        super().__init__(processes=None)
        self.turns = turns

    def scan(self, now=None):
        return list(self.turns)


def test_a_permission_prompt_shows_the_turn_as_waiting(tmp_path):
    from datetime import datetime, timezone

    turn = ActiveTurn("claude", "s1", "claude/claude-opus-5-5", "alpha", datetime.now(timezone.utc))
    activity = _FixedActivity([turn])
    source = CodingToolSource(tmp_path, tmp_path, prefs_file=None, activity=activity)
    bus = LiveBus()
    source.publish_turns(bus)

    source.hook_event(bus, "claude", "Notification", "s1")
    assert bus.replay(0)[-1]["type"] == "request.activity" and bus.replay(0)[-1]["state"] == "waiting"
    assert bus.running()[0]["state"] == "waiting"

    activity.turns = []  # the log went quiet while the prompt waits: the turn stays
    source.publish_turns(bus)
    assert len(bus.running()) == 1

    source.hook_event(bus, "claude", "PostToolUse", "s1")
    activity.turns = [turn]
    source.publish_turns(bus)
    assert bus.running()[0]["state"] == "working"

    source.hook_event(bus, "claude", "Notification", "other-session")  # no turn there: ignored
    activity.turns = []
    source.hook_event(bus, "claude", "Stop", "s1")
    assert bus.running() == []


def test_hook_endpoint_needs_the_token_and_all_fields(tmp_path):
    source = CodingToolSource(tmp_path, tmp_path, prefs_file=None, activity=_FixedActivity([]))
    srv = CompanionServer(
        LedgerSource(tmp_path / "none.db"), token="t0ken", bus=LiveBus(), state_path=None, coding_tools=source
    )
    srv.start_background()
    try:
        auth = {"Authorization": "Bearer t0ken"}
        assert _post(f"{srv.url}/v1/hooks", {"agent": "claude", "event": "Stop", "session": "s"}, auth)[0] == 200
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(f"{srv.url}/v1/hooks", {"agent": "claude"}, auth)
        assert err.value.code == 422
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(f"{srv.url}/v1/hooks", {"agent": "claude", "event": "Stop", "session": "s"})
        assert err.value.code == 401
    finally:
        srv.shutdown()
        srv.shutdown_companion()


def test_hook_send_talks_only_to_a_local_companion(tmp_path):
    state = tmp_path / "companion.json"
    state.write_text(json.dumps({"url": "https://example.com", "token": "t"}))
    assert hook.send({"agent": "claude", "event": "Stop", "session": "s"}, state) is False
    assert Path(state).exists()


def test_routed_calls_are_written_to_the_usage_ledger(companion):
    srv, driver, routes = companion
    recorded = []
    driver._auto_record_usage = lambda resp, elapsed_ms, status="success", error=None: recorded.append(
        (resp["meta"], status)
    )
    routes.save({"tools": {"claude-code": {"models": {"*": "ollama/qwen3"}}}})
    _post(
        f"{srv.url}/tools/claude-code/v1/messages",
        {"model": "claude-haiku-4-5", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    _post(f"{srv.url}/tools/claude-code/v1/messages", {"model": "claude-haiku-4-5", "messages": []})
    assert recorded == [({"prompt_tokens": 5, "completion_tokens": 2}, "success")] * 2


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("final", [True, False])
def test_codex_routing_restores_the_file_byte_for_byte(tmp_path, newline, final):
    codex = tmp_path / "codex"
    codex.mkdir()
    lines = ['model = "gpt-5.5"', 'model_provider = "azure"', "", "[tui]", "theme = 'dark'", "", ""]
    original = newline.join(lines) + (newline if final else "")
    (codex / "config.toml").write_bytes(original.encode())
    routing = ToolRouting(None, claude_root=tmp_path, codex_root=codex)
    routing.set_enabled("codex", True, "http://127.0.0.1:47811")
    routing.apply_enabled("http://127.0.0.1:50000")
    assert (b"\r\n" in (codex / "config.toml").read_bytes()) == (newline == "\r\n")
    routing.set_enabled("codex", False, "")
    assert (codex / "config.toml").read_bytes() == original.encode()


def test_claude_settings_keep_their_line_ends_and_text(tmp_path):
    settings = tmp_path / "settings.json"
    original = '{\r\n  "statusLine": "café ☕"\r\n}\r\n'
    settings.write_bytes(original.encode())
    routing = ToolRouting(None, claude_root=tmp_path, codex_root=tmp_path)
    routing.set_enabled("claude-code", True, "http://127.0.0.1:47811")
    assert "café ☕" in settings.read_bytes().decode()
    routing.set_enabled("claude-code", False, "")
    assert settings.read_bytes() == original.encode()


# ------------------------------------------------------------------ crash recovery


def test_a_new_companion_repoints_enabled_tools_and_restores_stale_ones(tmp_path):
    """A killed companion leaves configs pointing at a dead port; the next one fixes them."""
    claude = tmp_path / "claude"
    claude.mkdir()
    (claude / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": "https://gw.example"}}))
    routing = ToolRouting(tmp_path / "prefs.json", claude_root=claude, codex_root=tmp_path / "codex")
    routing.set_enabled("claude-code", True, "http://127.0.0.1:5000")
    routing.set_enabled("codex", True, "http://127.0.0.1:5000")

    # The next companion comes up on another port: enabled tools follow it.
    assert routing.apply_enabled("http://127.0.0.1:6000") == []
    env = json.loads((claude / "settings.json").read_text())["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:6000/tools/claude-code"
    assert "127.0.0.1:6000/tools/codex/v1" in (tmp_path / "codex" / "config.toml").read_text()

    # Codex was turned off while its config still pointed at the dead companion.
    prefs = json.loads((tmp_path / "prefs.json").read_text())
    prefs["routed_tools"] = ["claude-code"]
    (tmp_path / "prefs.json").write_text(json.dumps(prefs))
    assert routing.apply_enabled("http://127.0.0.1:6000") == []
    assert not routing.routed("codex")
    assert routing.routed("claude-code")

    # `prompture companion --restore`: everything back, the gateway it replaced included.
    assert routing.restore_all() == ["claude-code"]
    assert json.loads((claude / "settings.json").read_text()) == {"env": {"ANTHROPIC_BASE_URL": "https://gw.example"}}


def test_shutdown_needs_the_token_then_stops_and_puts_configs_back(tmp_path):
    routing = ToolRouting(tmp_path / "prefs.json", claude_root=tmp_path / "claude", codex_root=tmp_path / "codex")
    srv = CompanionServer(
        LedgerSource(tmp_path / "none.db"), token="t0ken", bus=LiveBus(), state_path=None, tool_routing=routing
    )
    done = threading.Event()

    def serve():
        try:
            srv.run()
        finally:
            done.set()

    threading.Thread(target=serve, daemon=True).start()
    info = json.loads(urllib.request.urlopen(f"{srv.url}/v1/companion/info", timeout=5).read())
    assert info["features"]["shutdown"] == "/v1/shutdown"
    routing.set_enabled("claude-code", True, srv.url)
    assert routing.routed("claude-code")

    with pytest.raises(urllib.error.HTTPError) as err:
        _post(f"{srv.url}/v1/shutdown", {})
    assert err.value.code == 401

    status, _, _ = _post(f"{srv.url}/v1/shutdown", {}, {"Authorization": "Bearer t0ken"})
    assert status == 202
    assert done.wait(5)
    assert not routing.routed("claude-code")
    assert routing.enabled() == ["claude-code"]  # the choice survives for the next start


@pytest.mark.parametrize(
    "original", [None, "", "GEMINI_MODEL=pro\n", "CODE_ASSIST_ENDPOINT=https://proxy.example\nX=1", "A=1\r\nB=2\r\n"]
)
def test_gemini_routing_restores_its_env_file_byte_for_byte(tmp_path, original):
    gemini = tmp_path / "gemini"
    gemini.mkdir()
    env = gemini / ".env"
    if original is not None:
        env.write_bytes(original.encode())
    routing = ToolRouting(None, claude_root=tmp_path / "c", codex_root=tmp_path / "x", gemini_root=gemini)
    routing.set_enabled("gemini-cli", True, "http://127.0.0.1:47811")
    text = env.read_text()
    assert "CODE_ASSIST_ENDPOINT=http://127.0.0.1:47811/tools/gemini-cli" in text
    assert "proxy.example" not in text and routing.routed("gemini-cli")
    routing.apply_enabled("http://127.0.0.1:50000")  # a new port: still one block
    assert env.read_text().count("CODE_ASSIST_ENDPOINT") == 1
    routing.set_enabled("gemini-cli", False, "")
    if original is None:
        assert not env.exists()
    else:
        assert env.read_bytes() == original.encode()


def test_a_claude_settings_file_routing_created_is_removed_again(tmp_path):
    routing = ToolRouting(None, claude_root=tmp_path, codex_root=tmp_path / "codex")
    routing.set_enabled("claude-code", True, "http://127.0.0.1:47811")
    assert (tmp_path / "settings.json").exists()
    routing.set_enabled("claude-code", False, "")
    assert not (tmp_path / "settings.json").exists()
