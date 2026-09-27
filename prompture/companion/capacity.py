"""How much of each coding plan is left, read from the replies that pass through the router.

Vendors report a subscription's windows on every model reply: Claude Code's
plan in ``anthropic-ratelimit-unified-<window>-utilization`` / ``-reset``
headers, Codex's ChatGPT plan in ``x-codex-primary-*`` / ``x-codex-secondary-*``.
:class:`PlanCapacity` keeps the latest of each as the same plan snapshots the
coding-agent log readers produce (``claude/claude-code``, ``openai/codex``),
so limits, alerts and automations read them without any change, and Claude
Code's windows need no access to its login.

:func:`pick_agent` chooses the agent with the most plan left, for work that
can go to either.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any

from ..infra.coding_agent_readers import _percent_window, _plan_snapshot

#: Router tool → the rate-limit target its plan appears under.
TARGETS = {"claude-code": "claude/claude-code", "codex": "openai/codex"}
_AGENT = {"claude-code": ("claude", "Claude Code"), "codex": ("codex", "Codex")}
_CLAUDE_UTIL = re.compile(r"^anthropic-ratelimit-unified-(.+)-utilization$")
_CODEX_MINUTES = {300: "session_5h", 10080: "weekly"}


def _claude_window(name: str) -> str:
    if name == "5h":
        return "session_5h"
    if name == "7d":
        return "weekly"
    return "weekly_" + name[3:] if name.startswith("7d_") else name


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def windows_from_headers(
    tool: str, headers: Iterable[tuple[str, str]] | Mapping[str, str]
) -> tuple[dict[str, Any], str | None]:
    """A reply's plan windows (``{name: {limit, remaining, resets_at}}``) and plan type, if it has any."""
    items = headers.items() if isinstance(headers, Mapping) else headers
    h = {str(k).lower(): str(v) for k, v in items}
    windows: dict[str, Any] = {}
    if tool == "claude-code":
        for key, value in h.items():
            m = _CLAUDE_UTIL.match(key)
            util = _number(value)
            if m and util is not None:
                reset = _number(h.get(f"anthropic-ratelimit-unified-{m.group(1)}-reset"))
                window = _percent_window(util * 100, reset)
                if window:
                    windows[_claude_window(m.group(1))] = window
        return windows, None
    if tool == "codex":
        for key in ("primary", "secondary"):
            used = _number(h.get(f"x-codex-{key}-used-percent"))
            minutes = int(_number(h.get(f"x-codex-{key}-window-minutes")) or 0)
            if used is None or minutes <= 0:
                continue
            window = _percent_window(used, _number(h.get(f"x-codex-{key}-reset-at")))
            if window:
                windows[_CODEX_MINUTES.get(minutes, f"{minutes}m")] = window
        return windows, h.get("x-codex-plan-type")
    return windows, None


class PlanCapacity:
    """The newest plan windows per tool, from reply headers. Thread-safe."""

    def __init__(self) -> None:
        self._snaps: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def observe(self, tool: str, headers: Iterable[tuple[str, str]] | Mapping[str, str]) -> None:
        if tool not in TARGETS:
            return
        windows, plan = windows_from_headers(tool, headers)
        if not windows:
            return
        agent, name = _AGENT[tool]
        snap = _plan_snapshot(agent, name, windows, time.time(), plan)
        snap["source"] = "headers"
        with self._lock:
            self._snaps[TARGETS[tool]] = snap

    def limits(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._snaps)


def headroom(snap: Mapping[str, Any] | None, now: float | None = None) -> float | None:
    """The share of a plan left in its tightest window that hasn't reset (0–100), or ``None`` if unknown."""
    if not snap:
        return None
    now = now or time.time()
    left: list[float] = []
    for window in (snap.get("windows") or {}).values():
        limit, remaining, reset = window.get("limit"), window.get("remaining"), window.get("resets_at")
        if not limit or remaining is None:
            continue
        left.append(100.0 if reset and reset <= now else remaining / limit * 100)
    return min(left) if left else None


def pick_agent(
    limits: Mapping[str, Mapping[str, Any]], agents: Iterable[str], targets: Mapping[str, str]
) -> tuple[str, str]:
    """The agent with the most plan left, and why. Agents with no known plan count as full."""
    scored = []
    for agent in agents:
        room = headroom(limits.get(targets.get(agent, "")))
        scored.append((100.0 if room is None else room, room is not None, agent))
    if not scored:
        raise ValueError("No agents to choose from.")
    scored.sort(key=lambda t: (-t[0], not t[1]))
    best, known, agent = scored[0]
    why = f"{best:.0f}% of its plan left" if known else "no plan limit known"
    return agent, why
