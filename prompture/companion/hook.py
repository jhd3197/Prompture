"""Coding-agent hook: tells the local companion what an agent is doing, the moment it happens.

Claude Code runs ``python <this file> claude`` on the events
:mod:`.tool_routing` installs (a prompt, tools starting and finishing,
notifications, the end of a turn) and passes the event as JSON on stdin. This
forwards the event's name, the session id and the working folder — nothing
else — to the companion named in ``~/.prompture/companion.json``, which turns
them into live state: ``Notification`` mid-turn is a permission prompt
waiting on the user.

On a prompt (``UserPromptSubmit``) it also sends the prompt, so the companion
can pick the project notes the session should start with (see
:mod:`.memory`); when it answers with some, they're printed for Claude Code
to add as context. That happens once per session; the prompt isn't stored.

Otherwise it exits 0 without output, quickly, whether or not a companion
runs, so it never gets in the agent's way.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

STATE_FILE = Path.home() / ".prompture" / "companion.json"
TIMEOUT = 3.0  # Claude Code gives the hook 5 s
#: The prompt is only needed to choose relevant notes; long ones are cut.
PROMPT_CHARS = 4000


def payload(agent: str, raw: str) -> dict[str, Any] | None:
    """What is sent for one hook event: the agent, the event name, the session id and the folder.

    A prompt event also carries the prompt (to choose the notes it gets).
    """
    try:
        data = json.loads(raw or "{}")
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    event, session = data.get("hook_event_name"), data.get("session_id")
    if not isinstance(event, str) or not isinstance(session, str) or not session:
        return None
    body: dict[str, Any] = {"agent": agent, "event": event, "session": session}
    cwd = data.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        name = cwd.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if name:
            body["project"] = name
            body["cwd"] = cwd
    if event == "UserPromptSubmit" and isinstance(data.get("prompt"), str):
        body["prompt"] = data["prompt"][:PROMPT_CHARS]
    return body


def exchange(body: dict[str, Any], state_file: Path = STATE_FILE) -> dict[str, Any] | None:
    """Post *body* to the running companion and return its answer; ``None`` when there is none."""
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        url, token = str(state["url"]), str(state["token"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
        return None  # only ever talk to this machine
    request = urllib.request.Request(
        f"{url}/v1/hooks",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:  # nosec B310 - loopback http only, checked above
            answer = json.loads(resp.read() or b"{}")
    except (OSError, ValueError):
        return None
    return answer if isinstance(answer, dict) else {}


def send(body: dict[str, Any], state_file: Path = STATE_FILE) -> bool:
    """Post *body* to the running companion; ``False`` when there is none."""
    return exchange(body, state_file) is not None


def context_output(event: str, context: str) -> str:
    """What Claude Code reads from a hook's stdout to add *context* to the conversation."""
    return json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": context}})


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    agent = args[0] if args else "claude"
    try:
        body = payload(agent, sys.stdin.read())
        if body:
            answer = exchange(body)
            context = (answer or {}).get("context")
            if isinstance(context, str) and context and body["event"] == "UserPromptSubmit":
                sys.stdout.write(context_output(body["event"], context))
    except Exception:  # a hook must never fail the agent
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
