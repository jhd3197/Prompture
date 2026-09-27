"""Coding-agent hook: tells the local companion what an agent is doing, the moment it happens.

Claude Code runs ``python -m prompture.companion.hook claude`` on the events
:mod:`.tool_routing` installs (a prompt, tools starting and finishing,
notifications, the end of a turn) and passes the event as JSON on stdin. This
forwards the event's name and session id — nothing else — to the companion
named in ``~/.prompture/companion.json``, which turns them into live state:
``Notification`` mid-turn is a permission prompt waiting on the user.

It always exits 0 without output, quickly, whether or not a companion runs, so
it never gets in the agent's way.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

STATE_FILE = Path.home() / ".prompture" / "companion.json"
TIMEOUT = 1.5


def payload(agent: str, raw: str) -> dict[str, Any] | None:
    """What is sent for one hook event: the agent, the event name and the session id."""
    try:
        data = json.loads(raw or "{}")
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    event, session = data.get("hook_event_name"), data.get("session_id")
    if not isinstance(event, str) or not isinstance(session, str) or not session:
        return None
    return {"agent": agent, "event": event, "session": session}


def send(body: dict[str, Any], state_file: Path = STATE_FILE) -> bool:
    """Post *body* to the running companion; ``False`` when there is none."""
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        url, token = str(state["url"]), str(state["token"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
        return False  # only ever talk to this machine
    request = urllib.request.Request(
        f"{url}/v1/hooks",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT):  # nosec B310 - loopback http only, checked above
            return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    agent = args[0] if args else "claude"
    try:
        body = payload(agent, sys.stdin.read())
        if body:
            send(body)
    except Exception:  # a hook must never fail the agent
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
