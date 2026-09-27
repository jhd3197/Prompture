"""Gemini CLI through the companion's router: pass through, presets, routing and memory."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prompture.companion import CompanionServer, LedgerSource, LiveBus
from prompture.companion.memory import MemoryService, cwd_of, previous_task, prompts_of
from prompture.companion.router import Router, Routes, gemini_kind
from prompture.gateway import gemini_to_driver

SETUP = (
    "This is the Gemini CLI. We are setting up the context for our chat.\n"
    "Today's date is Sunday, September 27, 2026.\nMy operating system is: win32\n"
    "I'm currently working in the directory: C:\\work\\shop\n"
)


class _CodeAssist(BaseHTTPRequestHandler):
    seen: list[tuple[str, dict]] = []
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        _CodeAssist.seen.append((self.path, body))
        if ":streamGenerateContent" in self.path:
            chunks = [
                {
                    "response": {
                        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}}],
                        "modelVersion": body.get("model"),
                    }
                },
                {
                    "response": {
                        "candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}],
                        "usageMetadata": {
                            "promptTokenCount": 1200,
                            "cachedContentTokenCount": 1000,
                            "candidatesTokenCount": 30,
                            "thoughtsTokenCount": 20,
                        },
                        "modelVersion": body.get("model"),
                    }
                },
            ]
            data = b"".join(b"data: " + json.dumps(c).encode() + b"\r\n\r\n" for c in chunks)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        raw = json.dumps({"currentTier": {"id": "free-tier"}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _Driver:
    supports_streaming = True
    supports_tool_use = True

    def generate_messages_with_tools_stream(self, messages, tools, options):
        from prompture.agents.live_events import MessageStop, ToolUseStart, ToolUseStop

        self.messages = messages
        yield ToolUseStart(id="c1", name="read_file")
        yield ToolUseStop(id="c1", name="read_file", input={"path": "a.py"})
        yield MessageStop(stop_reason="tool_use", usage={"prompt_tokens": 100, "completion_tokens": 5})

    def generate_messages(self, messages, options):
        return {"text": "routed", "meta": {"prompt_tokens": 100, "completion_tokens": 5}}


@pytest.fixture
def stack(tmp_path):
    _CodeAssist.seen = []
    vendor = ThreadingHTTPServer(("127.0.0.1", 0), _CodeAssist)
    threading.Thread(target=vendor.serve_forever, daemon=True).start()
    bus, driver = LiveBus(), _Driver()
    routes = Routes(tmp_path / "routes.json")
    memory = MemoryService(directory=tmp_path / "memory", python="py")
    router = Router(
        bus,
        routes,
        upstream=lambda tool: f"http://127.0.0.1:{vendor.server_address[1]}",
        driver_for=lambda m: driver,
        memory=memory,
    )
    srv = CompanionServer(LedgerSource(tmp_path / "none.db"), token="t", bus=bus, state_path=None, router=router)
    srv.start_background()
    yield srv, router, routes, driver, memory
    srv.shutdown()
    srv.shutdown_companion()
    vendor.shutdown()


def _body(model="gemini-2.5-pro", *extra, session="g1", tools=None):
    request = {
        "contents": [
            {"role": "user", "parts": [{"text": SETUP}]},
            {"role": "model", "parts": [{"text": "Got it. Thanks for the context!"}]},
            *extra,
        ],
        "session_id": session,
    }
    if tools:
        request["tools"] = tools
    return {"model": model, "project": "p-123", "user_prompt_id": "u1", "request": request}


def _prompt(text):
    return {"role": "user", "parts": [{"text": text}]}


def _post(url, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer ya29.x"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read()


def _wait(router, n):
    deadline = time.monotonic() + 5
    while len(router.calls.calls()) < n and time.monotonic() < deadline:
        time.sleep(0.02)
    return router.calls.calls()


def test_gemini_cli_passes_through_with_its_login_and_usage_is_recorded(stack):
    srv, router, _, _, _ = stack
    base = f"{srv.url}/tools/gemini-cli/v1internal"
    raw = _post(f"{base}:streamGenerateContent?alt=sse", _body("gemini-2.5-pro", _prompt("fix the build")))
    assert b'"text": "hi"' in raw
    _post(f"{base}:loadCodeAssist", {"metadata": {}})  # sign-in and quota calls are the vendor's, and not recorded
    assert [p for p, _ in _CodeAssist.seen] == [
        "/v1internal:streamGenerateContent?alt=sse",
        "/v1internal:loadCodeAssist",
    ]
    (call,) = _wait(router, 1)
    assert (call.tool, call.session, call.billing, call.served) == (
        "gemini-cli",
        "g1",
        "subscription",
        "google/gemini-2.5-pro",
    )
    assert (call.input_tokens, call.cache_read_tokens, call.output_tokens) == (1200, 1000, 50)
    assert call.project == "shop" and call.plan_equivalent_usd > 0 and call.cost_usd == 0


def test_economy_moves_gemini_to_flash_on_the_same_sign_in(stack):
    srv, _, routes, _, _ = stack
    routes.save({"tools": {"gemini-cli": {"preset": "economy"}}})
    _post(
        f"{srv.url}/tools/gemini-cli/v1internal:streamGenerateContent?alt=sse", _body("gemini-2.5-pro", _prompt("hi"))
    )
    assert _CodeAssist.seen[-1][1]["model"] == "gemini-2.5-flash"


def test_gemini_requests_can_go_to_a_prompture_model(stack):
    srv, router, routes, driver, _ = stack
    routes.save({"tools": {"gemini-cli": {"kinds": {"main": "ollama/qwen3"}}}})
    tools = [
        {
            "functionDeclarations": [
                {"name": "read_file", "parameters": {"type": "OBJECT", "properties": {"path": {"type": "STRING"}}}}
            ]
        }
    ]
    raw = _post(
        f"{srv.url}/tools/gemini-cli/v1internal:streamGenerateContent?alt=sse",
        _body("gemini-2.5-pro", _prompt("read a.py"), tools=tools),
    )
    chunks = [json.loads(line[5:]) for line in raw.decode().splitlines() if line.startswith("data:")]
    parts = [p for c in chunks for p in c["response"]["candidates"][0]["content"]["parts"]]
    assert parts == [{"functionCall": {"name": "read_file", "args": {"path": "a.py"}, "id": "c1"}}]
    assert chunks[-1]["response"]["usageMetadata"]["promptTokenCount"] == 100
    assert _CodeAssist.seen == []
    assert driver.messages[-1] == {"role": "user", "content": "read a.py"}
    (call,) = _wait(router, 1)
    assert call.route == "routed" and call.served == "ollama/qwen3" and call.original_billing == "subscription"

    raw = _post(
        f"{srv.url}/tools/gemini-cli/v1internal:generateContent",
        _body("gemini-2.5-pro", _prompt("hello"), session="g2"),
    )
    assert json.loads(raw)["response"]["candidates"][0]["content"]["parts"] == [{"text": "routed"}]


def test_gemini_requests_translate_calls_and_schemas():
    request = {
        "systemInstruction": {"parts": [{"text": "Be brief."}]},
        "contents": [
            {"role": "user", "parts": [{"text": "list files"}]},
            {"role": "model", "parts": [{"functionCall": {"name": "ls", "args": {"dir": "."}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "ls", "response": {"output": "a.py"}}}]},
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {"name": "ls", "parameters": {"type": "OBJECT", "properties": {"dir": {"type": "STRING"}}}}
                ]
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 100},
    }
    messages, tools, options = gemini_to_driver(request)
    assert messages[0] == {"role": "system", "content": "Be brief."}
    call_id = messages[2]["tool_calls"][0]["id"]
    assert messages[3] == {"role": "tool", "tool_call_id": call_id, "content": "a.py"}
    assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {"dir": {"type": "string"}}}
    assert options == {"temperature": 0, "max_tokens": 100}
    assert gemini_kind({"request": {"contents": request["contents"]}}) == "tool_result"


def test_gemini_sessions_get_project_notes_in_their_first_content(stack):
    srv, _, _, _, memory = stack
    memory.memory.add("shop", "Run the build with pnpm, never npm", kind="convention", verified=True)
    body = _body("gemini-2.5-pro", _prompt("the build is broken"))
    assert cwd_of("gemini", body) == "C:\\work\\shop" and prompts_of("gemini", body) == [2]
    _post(f"{srv.url}/tools/gemini-cli/v1internal:streamGenerateContent?alt=sse", body)
    first = _CodeAssist.seen[-1][1]["request"]["contents"][0]
    assert "pnpm" in first["parts"][0]["text"] and first["parts"][1]["text"] == SETUP
    later = _body(
        "gemini-2.5-pro",
        _prompt("the build is broken"),
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "run_shell_command", "args": {"command": "pnpm build"}}}],
        },
        {"role": "user", "parts": [{"functionResponse": {"name": "run_shell_command", "response": {"output": "ok"}}}]},
    )
    _post(f"{srv.url}/tools/gemini-cli/v1internal:streamGenerateContent?alt=sse", later)
    assert _CodeAssist.seen[-1][1]["request"]["contents"][0] == first  # the same notes, the same place
    finished = previous_task(
        "gemini",
        _body(
            "gemini-2.5-pro",
            *later["request"]["contents"][2:],
            {"role": "model", "parts": [{"text": "Done."}]},
            _prompt("next"),
        ),
    )
    assert finished["steps"] == ["run_shell_command(pnpm build)"] and finished["output"] == "Done."
