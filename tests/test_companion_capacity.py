"""Plan capacity read from the replies that pass through the router."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prompture.companion import CompanionServer, LedgerSource, LiveBus
from prompture.companion.capacity import PlanCapacity, headroom, pick_agent, windows_from_headers
from prompture.companion.router import Router, Routes

CLAUDE = [
    ("anthropic-ratelimit-unified-5h-utilization", "0.29"),
    ("anthropic-ratelimit-unified-5h-reset", "1790496600"),
    ("anthropic-ratelimit-unified-7d-utilization", "0.16"),
    ("anthropic-ratelimit-unified-7d-reset", "1790982000"),
    ("anthropic-ratelimit-unified-status", "allowed"),
]
CODEX = {
    "x-codex-primary-used-percent": "2",
    "x-codex-primary-window-minutes": "10080",
    "x-codex-primary-reset-at": "1791084760",
    "x-codex-secondary-used-percent": "0",
    "x-codex-secondary-window-minutes": "0",
    "x-codex-plan-type": "prolite",
}


def test_plan_windows_come_from_each_vendors_headers():
    windows, _ = windows_from_headers("claude-code", CLAUDE)
    assert windows == {
        "session_5h": {"limit": 100, "remaining": 71, "resets_at": 1790496600.0},
        "weekly": {"limit": 100, "remaining": 84, "resets_at": 1790982000.0},
    }
    windows, plan = windows_from_headers("codex", CODEX)
    assert windows == {"weekly": {"limit": 100, "remaining": 98, "resets_at": 1791084760.0}} and plan == "prolite"
    assert windows_from_headers("codex", {"content-type": "text/plain"}) == ({}, None)


def test_the_newest_capacity_wins_in_the_companions_limits(tmp_path):
    srv = CompanionServer(LedgerSource(tmp_path / "none.db"), bus=LiveBus(), state_path=None)
    try:
        srv.capacity.observe("claude-code", CLAUDE)
        snap = srv.rate_limits()["claude/claude-code"]
        assert snap["source"] == "headers" and snap["tool_name"] == "Claude Code"
        assert snap["windows"]["session_5h"]["remaining"] == 71
    finally:
        srv.server_close()


def test_the_agent_with_the_most_plan_left_is_picked():
    now = time.time()
    limits = {
        "claude/claude-code": {"windows": {"session_5h": {"limit": 100, "remaining": 20, "resets_at": now + 60}}},
        "openai/codex": {"windows": {"weekly": {"limit": 100, "remaining": 5, "resets_at": now - 60}}},  # reset already
    }
    targets = {"claude": "claude/claude-code", "codex": "openai/codex"}
    assert headroom(limits["openai/codex"], now) == 100.0
    assert pick_agent(limits, ["claude", "codex"], targets) == ("codex", "100% of its plan left")
    assert pick_agent({}, ["claude", "codex"], targets)[1] == "no plan limit known"
    capacity = PlanCapacity()
    capacity.observe("gemini-cli", CLAUDE)  # no plan headers known for it: ignored
    assert capacity.limits() == {}


class _UnlabeledStream(BaseHTTPRequestHandler):
    """Like ChatGPT's Codex backend: a streamed reply with no Content-Type and no length."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        usage = {"input_tokens": 40, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 30}}
        done = {"type": "response.completed", "response": {"model": "gpt-6", "usage": usage}}
        frames = [
            b'event: response.created\ndata: {"type":"response.created"}\n\n',
            f"event: response.completed\ndata: {json.dumps(done)}\n\n".encode(),
        ]
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        for key, value in CODEX.items():
            self.send_header(key, value)
        self.end_headers()
        for data in frames:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")


def test_a_stream_without_a_content_type_still_streams_and_is_counted(tmp_path):
    vendor = ThreadingHTTPServer(("127.0.0.1", 0), _UnlabeledStream)
    threading.Thread(target=vendor.serve_forever, daemon=True).start()
    bus = LiveBus()
    upstream = f"http://127.0.0.1:{vendor.server_address[1]}"
    router = Router(bus, Routes(tmp_path / "routes.json"), upstream=lambda tool: upstream)
    srv = CompanionServer(LedgerSource(tmp_path / "none.db"), bus=bus, state_path=None, router=router)
    srv.start_background()
    try:
        body = {"model": "gpt-6", "stream": True, "input": [{"type": "message", "role": "user", "content": "hi"}]}
        req = urllib.request.Request(
            f"{srv.url}/tools/codex/v1/responses",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.headers.get("Content-Length") is None  # streamed through, not buffered
            assert b"response.completed" in resp.read()
        deadline = time.monotonic() + 5
        while not router.calls.calls() and time.monotonic() < deadline:
            time.sleep(0.02)
        (call,) = router.calls.calls()
        assert (call.input_tokens, call.cache_read_tokens, call.output_tokens) == (40, 30, 5)
        assert call.served == "openai/gpt-6"
        assert srv.rate_limits()["openai/codex"]["windows"]["weekly"]["remaining"] == 98
    finally:
        srv.shutdown()
        srv.shutdown_companion()
        vendor.shutdown()
