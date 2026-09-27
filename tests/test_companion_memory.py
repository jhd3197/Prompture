"""Shared project memory: notes, what each session receives, and mined skills."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from prompture.companion import CodingToolSource, CompanionServer, LedgerSource, LiveBus, hook
from prompture.companion.memory import MemoryService, cwd_of, previous_task, prompts_of
from prompture.companion.router import Router, Routes
from prompture.companion.tool_routing import ToolRouting
from prompture.infra.coding_agent_activity import AgentActivity
from prompture.session_memory import InMemorySessionStore
from prompture.session_memory.project import ProjectMemory

# ------------------------------------------------------------------ project notes


def test_notes_are_kept_per_project_and_can_be_edited_and_deleted():
    mem = ProjectMemory(InMemorySessionStore())
    fix = mem.add(
        "shop",
        "Build needs NODE_OPTIONS=--max-old-space-size=4096",
        kind="fix",
        source="package.json:12",
        verified=True,
        agent="claude",
    )
    mem.add("other", "Unrelated", verified=True)
    assert [f.id for f in mem.notes("shop")] == [fix.id]
    assert {p["project"]: p["verified"] for p in mem.projects()} == {"other": 1, "shop": 1}

    updated = mem.update(
        "shop", fix.id, content="Build needs more heap: NODE_OPTIONS=--max-old-space-size=4096", verified=False
    )
    assert updated.content.startswith("Build needs more heap") and updated.metadata["verified"] is False
    assert updated.metadata["source"] == "package.json:12"  # untouched fields stay
    assert mem.delete("shop", fix.id) and mem.notes("shop") == []
    assert not mem.delete("shop", fix.id)


def test_a_task_gets_only_relevant_verified_notes_within_its_budget():
    mem = ProjectMemory(InMemorySessionStore())
    build = mem.add("shop", "The build breaks unless NODE_OPTIONS raises the heap limit", kind="fix", verified=True)
    mem.add("shop", "Payments use Stripe webhooks, never polling", kind="decision", verified=True)
    mem.add("shop", "The build uses pnpm, not npm", kind="convention", verified=False)  # unverified
    pinned = mem.add("shop", "Run tests with: pnpm test --silent", kind="command", verified=True, pinned=True)
    chosen = mem.select("shop", "the build is failing again, fix it")
    assert [f.id for f in chosen] == [pinned.id, build.id]  # pinned first, then relevant; payments left out
    assert [f.id for f in mem.select("shop", "build", verified_only=False)][-1] != pinned.id
    # A tight budget keeps the best note only.
    tight = mem.select("shop", "the build is failing", budget_tokens=25)
    assert [f.id for f in tight] == [pinned.id]


def test_the_block_teaches_saving_even_before_there_are_notes():
    mem = ProjectMemory(InMemorySessionStore())
    block = mem.render("shop", [], save_hint="To save: run prompture memory add")
    assert block.startswith('<project-memory project="shop">') and "prompture memory add" in block
    assert mem.render("shop", []) == ""


# ------------------------------------------------------------------ reading requests


CODEX_ENV = "<environment_context>\n  <cwd>C:\\work\\shop</cwd>\n</environment_context>"


def _codex(*items):
    return {"model": "gpt-5.5-codex", "stream": False, "prompt_cache_key": "cx-1", "input": list(items)}


def _user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def test_requests_state_their_folder_and_their_tasks():
    claude = {
        "system": [{"type": "text", "text": "You are Claude Code.\nWorking directory: /home/me/shop\nIs git repo: yes"}]
    }
    assert cwd_of("anthropic", claude) == "/home/me/shop"
    codex = _codex(_user(CODEX_ENV), _user("fix the build"))
    assert cwd_of("openai", codex) == "C:\\work\\shop"
    assert prompts_of("openai", codex) == [1]  # the environment context isn't a prompt
    plugins = {"type": "input_text", "text": "<recommended_plugins>\nplugins\n</recommended_plugins>"}
    bundled = _codex(
        {"type": "message", "role": "user", "content": [plugins, {"type": "input_text", "text": CODEX_ENV}]},
        _user("what is the <b>code</b> word?"),
    )
    assert prompts_of("openai", bundled) == [1]  # context wrapped in tags isn't the user's prompt

    history = _codex(
        _user(CODEX_ENV),
        _user("fix the build"),
        {
            "type": "function_call",
            "call_id": "a",
            "name": "shell",
            "arguments": json.dumps({"command": ["bash", "-lc", "cd app && npm ci"]}),
        },
        {"type": "function_call_output", "call_id": "a", "output": "ok"},
        {
            "type": "function_call",
            "call_id": "b",
            "name": "shell",
            "arguments": json.dumps({"command": ["bash", "-lc", "npm run build --prod"]}),
        },
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Fixed."}]},
        _user("now the tests"),
    )
    task = previous_task("openai", history)
    assert task == {"prompt": "fix the build", "steps": ["shell(npm ci)", "shell(npm run)"], "output": "Fixed."}


# ------------------------------------------------------------------ what sessions receive


def test_a_session_gets_its_notes_once_and_the_log_shows_what(tmp_path):
    svc = MemoryService(directory=tmp_path, python="py")
    svc.memory.add("shop", "The build breaks unless NODE_OPTIONS raises the heap", kind="fix", verified=True)
    text = svc.inject("claude", "s1", "shop", "the build fails", via="hook")
    assert "NODE_OPTIONS" in text and '"py" -m prompture memory add' in text
    assert svc.inject("claude", "s1", "shop", "the build fails again", via="hook") is None  # never twice
    (logged,) = svc.injections("shop")
    assert logged["session"] == "s1" and logged["via"] == "hook" and logged["text"] == text and logged["fact_ids"]

    # Through the router every request needs the same text again.
    first = svc.inject("codex", "c1", "shop", "build", via="router")
    assert svc.inject("codex", "c1", "shop", "something else", via="router") == first

    again = MemoryService(directory=tmp_path)  # a restarted companion remembers
    assert again.received("claude", "s1")["text"] == text
    assert again.inject("claude", "s1", "shop", "the build", via="hook") is None


def test_memory_can_be_turned_off_and_budgeted(tmp_path):
    svc = MemoryService(directory=tmp_path)
    svc.memory.add("shop", "The build breaks unless NODE_OPTIONS raises the heap", verified=True)
    svc.save_settings({"enabled": False, "budget_tokens": 7, "junk": 1})
    assert svc.settings() == {"enabled": False, "budget_tokens": 50, "verified_only": True, "teach": True}
    assert svc.inject("claude", "s", "shop", "build", via="hook") is None
    assert MemoryService(directory=tmp_path).settings()["enabled"] is False


# ------------------------------------------------------------------ end to end


class _Vendor(BaseHTTPRequestHandler):
    seen: list[dict] = []
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        _Vendor.seen.append(body)
        raw = json.dumps(
            {
                "id": "resp_vendor",
                "model": body.get("model"),
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def companion(tmp_path):
    _Vendor.seen = []
    vendor = ThreadingHTTPServer(("127.0.0.1", 0), _Vendor)
    threading.Thread(target=vendor.serve_forever, daemon=True).start()
    bus = LiveBus()
    memory = MemoryService(directory=tmp_path / "memory", python="py")
    router = Router(
        bus,
        Routes(tmp_path / "routes.json"),
        upstream=lambda tool: f"http://127.0.0.1:{vendor.server_address[1]}",
        memory=memory,
    )
    routing = ToolRouting(tmp_path / "prefs.json", claude_root=tmp_path / "claude", codex_root=tmp_path / "codex")
    tools = CodingToolSource(
        tmp_path / "claude-logs",
        tmp_path / "codex-logs",
        prefs_file=None,
        activity=AgentActivity(tmp_path / "a", tmp_path / "b"),
    )
    srv = CompanionServer(
        LedgerSource(tmp_path / "none.db"),
        token="t0ken",
        bus=bus,
        state_path=None,
        coding_tools=tools,
        tool_routing=routing,
        router=router,
        memory=memory,
    )
    srv.start_background()
    yield srv, memory
    srv.shutdown()
    srv.shutdown_companion()
    vendor.shutdown()


def _call(url, body=None, method=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": "Bearer t0ken", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def test_a_fix_claude_verified_reaches_codex_without_rediscovery(companion, tmp_path, monkeypatch):
    srv, memory = companion
    # Claude Code verified a fix in the shop project and saved it (what `prompture memory add` does).
    memory.memory.add(
        "shop",
        "npm run build needs NODE_OPTIONS=--max-old-space-size=4096",
        kind="fix",
        source="package.json:12",
        verified=True,
        agent="claude",
    )

    # Codex starts a session in the same folder: its first request carries the note…
    url = f"{srv.url}/tools/codex/v1/responses"
    first = _codex(_user(CODEX_ENV), _user("the build is out of memory, fix it"))
    _call(url, first)
    sent = _Vendor.seen[-1]["input"]
    assert sent[0]["role"] == "developer" and "NODE_OPTIONS" in sent[0]["content"][0]["text"]
    assert sent[1:] == first["input"]  # the rest untouched
    # …and every later request of the session carries the same note in the same place.
    later = _codex(
        _user(CODEX_ENV),
        _user("the build is out of memory, fix it"),
        {"type": "function_call", "call_id": "a", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "a", "output": "ok"},
    )
    _call(url, later)
    assert _Vendor.seen[-1]["input"][0] == sent[0] and len(_Vendor.seen[-1]["input"]) == 5

    # What it received is on record.
    state = _call(f"{srv.url}/v1/memory/projects/shop")
    (received,) = state["injections"]
    assert received["agent"] == "codex" and received["via"] == "router" and "NODE_OPTIONS" in received["text"]
    assert state["folder"] == "C:\\work\\shop"


def test_claude_gets_its_notes_through_the_hook(companion, tmp_path, monkeypatch, capsys):
    srv, memory = companion
    memory.memory.add("shop", "Payments go through Stripe webhooks", kind="decision", verified=True)
    state_file = tmp_path / "companion.json"
    state_file.write_text(json.dumps({"url": srv.url, "token": "t0ken"}))
    real = hook.exchange
    monkeypatch.setattr(hook, "exchange", lambda body: real(body, state_file))
    event = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "cl-1",
        "cwd": "/home/me/shop",
        "prompt": "why do payments use webhooks?",
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    assert hook.main(["claude"]) == 0
    out = json.loads(capsys.readouterr().out)
    context = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit" and "Stripe webhooks" in context

    # The next prompt adds nothing; the router leaves the session's requests alone too.
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({**event, "prompt": "and refunds?"})))
    hook.main(["claude"])
    assert capsys.readouterr().out == ""
    body = {
        "model": "claude-sonnet-5",
        "metadata": {"user_id": "user_x_account_y_session_cl-1"},
        "system": "Working directory: /home/me/shop",
        "messages": [{"role": "user", "content": "why webhooks?"}],
    }
    req = urllib.request.Request(
        f"{srv.url}/tools/claude-code/v1/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-api-key": "k"},
    )
    urllib.request.urlopen(req, timeout=10).read()
    assert _Vendor.seen[-1]["messages"] == body["messages"]


def test_notes_endpoints_add_edit_delete_and_settings(companion):
    srv, _ = companion
    base = f"{srv.url}/v1/memory"
    note = _call(f"{base}/projects/shop/facts", {"content": "Use pnpm", "kind": "convention", "verified": True})
    assert note["agent"] == "you" and note["verified"]
    edited = _call(f"{base}/projects/shop/facts/{note['id']}", {"content": "Use pnpm, never npm", "pinned": True})
    assert edited["content"] == "Use pnpm, never npm" and edited["pinned"]
    overview = _call(base)
    assert overview["settings"]["enabled"] and overview["projects"][0]["project"] == "shop"
    assert _call(f"{base}/settings", {"budget_tokens": 300})["budget_tokens"] == 300
    assert _call(f"{base}/projects/shop/facts/{note['id']}", method="DELETE") == {"deleted": note["id"]}
    with pytest.raises(urllib.error.HTTPError) as err:
        _call(f"{base}/projects/shop/facts/{note['id']}", method="DELETE")
    assert err.value.code == 404
    info = json.loads(urllib.request.urlopen(f"{srv.url}/v1/companion/info", timeout=5).read())
    assert info["capabilities"]["memory"] and info["features"]["memory"] == "/v1/memory"


# ------------------------------------------------------------------ skills


def test_a_recurring_workflow_becomes_a_skill_proposal_that_can_be_saved(tmp_path):
    folder = tmp_path / "shop"
    folder.mkdir()
    svc = MemoryService(directory=tmp_path / "memory")
    task = {"prompt": "release", "steps": ["Bash(npm ci)", "Bash(npm run)", "Bash(npm test)", "Edit"], "output": "done"}
    for i in range(3):
        svc.observe_task("claude", f"s{i}", 1, "shop", str(folder), task)
    svc.observe_task("claude", "s2", 1, "shop", str(folder), task)  # the same task again: counted once
    (proposal,) = svc.skills("shop")
    assert proposal["steps"] == task["steps"] and proposal["occurrences"] == 3 and not proposal["saved"]
    path = svc.save_skill("shop", proposal["name"])
    assert (
        path.endswith("SKILL.md")
        and "npm test" in (folder / ".claude" / "skills" / proposal["name"] / "SKILL.md").read_text()
    )
    assert svc.skills("shop")[0]["saved"]
    # A restarted companion re-reads the tasks it saw.
    assert MemoryService(directory=tmp_path / "memory").skills("shop")[0]["name"] == proposal["name"]


# ------------------------------------------------------------------ CLI


def test_the_cli_adds_lists_and_removes_notes(tmp_path, monkeypatch):
    from prompture.cli.cli import cli
    from prompture.session_memory import project

    monkeypatch.setattr(project, "DEFAULT_DB_PATH", tmp_path / "projects.db")
    folder = tmp_path / "shop"
    folder.mkdir()
    monkeypatch.chdir(folder)
    runner = CliRunner()
    added = runner.invoke(
        cli,
        ["memory", "add", "--kind", "command", "--verified", "--source", "Makefile:3", "make check runs everything"],
    )
    assert added.exit_code == 0 and "for shop, verified" in added.output
    listed = runner.invoke(cli, ["memory", "list", "--json"])
    (note,) = json.loads(listed.output)
    assert note["kind"] == "command" and note["source"] == "Makefile:3" and note["verified"]
    removed = runner.invoke(cli, ["memory", "rm", note["id"][:6]])
    assert removed.exit_code == 0
    assert "No notes for shop" in runner.invoke(cli, ["memory", "list"]).output
