"""Usage from coding agents that don't run through Prompture, for the companion.

The reading happens in the engine — :mod:`prompture.infra.coding_agent_usage`
and one reader per agent in :mod:`prompture.infra.coding_agent_readers`
(Claude Code, Codex, Kimi Code, Gemini CLI, Qwen Code, OpenCode, Cline,
Roo Code, Continue; Cursor and Antigravity are detected only). This module
adapts it to the companion API:

- calls join the ledger's rows in ``/v1/spend`` and stream as
  ``request.finished`` events tagged with the agent (``key_name``, ``tool``);
- plan windows (Codex from its logs; Claude Code only when opted in) join
  ``/v1/limits`` as ``source: "plan"`` targets;
- ``/v1/tools`` lists every agent: installed, runnable by Prompture, and its
  usage for the period;
- a Claude Code or Codex turn in progress shows as a running call: a
  ``request.started`` event when it starts (id ``agent:<agent>:<session>``),
  ``request.activity`` when its model or state changes, and ``request.ended``
  when it's over (see :mod:`prompture.infra.coding_agent_activity`). Its usage
  still arrives as ``request.finished`` events for the calls it made. With
  Claude Code's hooks installed (:mod:`.hook`), a turn stuck on a permission
  prompt shows ``state: "waiting"``.

Costs are what the same tokens would cost on the API; subscriptions don't
bill per token. Only token counts, model names, times and folder names are
read — never prompts, replies or tool output.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..infra.coding_agent_activity import ActiveTurn, AgentActivity
from ..infra.coding_agent_usage import AgentCall, CodingAgentUsage, UsageReader, coding_agents_overview
from .live import LiveBus
from .summary import UsageRow, window_start

logger = logging.getLogger("prompture.companion")

#: Seconds an installed-agents answer is reused (it looks up executables on PATH).
OVERVIEW_TTL = 60.0
#: How far back calls are kept: a year of activity, plus a week of slack.
RETENTION = timedelta(days=372)
#: How long a turn waiting on the user stays listed after its log went quiet.
WAITING_TTL = 6 * 3600.0
#: Hook events that mean the agent is working again (a prompt, a tool starting or done).
WORKING_HOOKS = {"UserPromptSubmit", "PreToolUse", "PostToolUse", "SubagentStop"}
#: Companion choices that outlive a restart (today: whether Claude plan windows are fetched).
PREFS_FILE = Path.home() / ".prompture" / "companion-prefs.json"


def _load_prefs(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def turn_event(turn: ActiveTurn, name: str, state: str = "working") -> dict[str, Any]:
    """A ``request.started`` payload for an agent turn in progress."""
    return {
        "request_id": turn.request_id,
        "ts": turn.since.isoformat(),
        "key_id": None,
        "key_name": name,
        "model": turn.model or turn.agent,
        "project": turn.project,
        "endpoint": "coding-agent",
        "stream": True,
        "state": state,
        "tool": turn.agent,
    }


def call_event(call: AgentCall, name: str) -> dict[str, Any]:
    """A ``request.finished`` payload for an agent call, shaped like the ledger's."""
    return {
        "request_id": call.id,
        "ts": call.ts.isoformat(),
        "key_id": None,
        "key_name": name,
        "model": call.model,
        "served_by": None,
        "project": call.project,
        "status": "ok",
        "error": None,
        "prompt_tokens": call.input_tokens,
        "completion_tokens": call.output_tokens,
        "cost_usd": call.cost_usd,
        "latency_ms": 0,
        "attempts": 1,
        "fallback": False,
        "deprioritized": [],
        "tool": call.agent,
    }


class CodingToolSource:
    """Every local coding agent's usage, as the companion needs it.

    With no arguments every registered reader runs. ``claude_dir`` /
    ``codex_dir`` (tests, custom locations) restrict it to Claude Code and
    Codex at those folders; ``readers`` sets the list explicitly.
    """

    def __init__(
        self,
        claude_dir: str | Path | None = None,
        codex_dir: str | Path | None = None,
        *,
        plan_usage: bool | None = None,
        cache_dir: str | Path | None = None,
        readers: list[UsageReader] | None = None,
        prefs_file: str | Path | None = PREFS_FILE,
        activity: AgentActivity | None = None,
    ) -> None:
        from ..infra.coding_agent_readers import ClaudeCodeReader, CodexReader
        from ..infra.coding_agent_usage import USAGE_READERS

        self.prefs_file = Path(prefs_file) if prefs_file else None
        explicit_dirs = bool(claude_dir or codex_dir)
        if plan_usage is None and not explicit_dirs and "PROMPTURE_CLAUDE_PLAN_USAGE" not in os.environ:
            # The environment wins; otherwise the choice made in a companion app.
            plan_usage = bool(_load_prefs(self.prefs_file).get("claude_plan_usage")) if self.prefs_file else None
        if readers is None:
            claude = ClaudeCodeReader(claude_dir, plan_usage=plan_usage, cache_dir=cache_dir)
            codex = CodexReader(codex_dir)
            if explicit_dirs:
                readers = [claude, codex]
            else:
                others = [cls() for agent, cls in USAGE_READERS.items() if agent not in ("claude", "codex")]
                readers = [claude, codex, *others]
        self.usage = CodingAgentUsage(readers, retention=RETENTION)
        self.names = {r.agent: r.display_name for r in readers}
        if activity is None:
            roots = {r.agent: getattr(r, "root", None) for r in readers}
            if "claude" in roots or "codex" in roots:
                activity = AgentActivity(roots.get("claude"), roots.get("codex"))
        self.activity = activity
        self._turns: dict[str, ActiveTurn] = {}
        self._states: dict[str, str] = {}
        #: State reported by an agent's hooks, per turn: ``(state, monotonic time)``.
        self._hooked: dict[str, tuple[str, float]] = {}
        self._turn_lock = threading.Lock()
        self._overview: tuple[float, list[dict[str, Any]]] | None = None

    @property
    def plan_usage(self) -> bool:
        """Whether Claude Code's plan windows are fetched (opt-in; see ``ClaudeCodeReader``)."""
        return any(getattr(r, "plan_usage", False) for r in self.usage.readers)

    def set_claude_plan_usage(self, enabled: bool) -> None:
        """Turn Claude Code's plan windows on or off, and remember the choice."""
        for reader in self.usage.readers:
            if hasattr(reader, "plan_usage"):
                reader.plan_usage = enabled
                if not enabled:
                    reader._plan = None  # forget the last answer too
        if self.prefs_file:
            prefs = _load_prefs(self.prefs_file)
            prefs["claude_plan_usage"] = enabled
            try:
                self.prefs_file.parent.mkdir(parents=True, exist_ok=True)
                self.prefs_file.write_text(json.dumps(prefs), encoding="utf-8")
            except OSError:
                logger.debug("could not save companion prefs", exc_info=True)

    def refresh(self, *, force: bool = False) -> list[AgentCall]:
        """Read what the logs gained since the last scan; returns the new calls."""
        return self.usage.refresh(force=force)

    def calls(self, period: str = "day", now: datetime | None = None, offset_minutes: int = 0) -> list[AgentCall]:
        return self.usage.calls(window_start(period, now, offset_minutes))

    def rows(self, period: str = "day", now: datetime | None = None, offset_minutes: int = 0) -> list[UsageRow]:
        """Calls in the current ``period`` window, as usage rows."""
        return [
            UsageRow(model=c.model, cost_usd=c.cost_usd, tokens=c.tokens, project=c.project)
            for c in self.calls(period, now, offset_minutes)
        ]

    def project_for(self, agent: str, session: str) -> str | None:
        """The project (folder name) of an agent session: from its running turn, else its logged calls."""
        with self._turn_lock:
            for turn in self._turns.values():
                if turn.agent == agent and turn.session == session and turn.project:
                    return turn.project
        since = datetime.now(timezone.utc) - timedelta(days=2)
        for call in reversed(self.usage.calls(since)):
            if call.agent == agent and call.session == session and call.project:
                return call.project
        return None

    def rate_limits(self) -> dict[str, dict[str, Any]]:
        """Plan windows keyed like provider targets: ``openai/codex``, ``claude/claude-code``."""
        self.usage.refresh()
        return self.usage.plan_limits()

    def event(self, call: AgentCall) -> dict[str, Any]:
        return call_event(call, self.names.get(call.agent, call.agent))

    def tools(self, period: str = "day", now: datetime | None = None, offset_minutes: int = 0) -> dict[str, Any]:
        """``/v1/tools``: each agent's usage for the period, plus what is installed."""
        if self._overview is None or time.monotonic() - self._overview[0] > OVERVIEW_TTL:
            try:
                self._overview = (time.monotonic(), coding_agents_overview())
            except Exception:
                logger.debug("coding agent overview failed", exc_info=True)
                self._overview = (time.monotonic(), [])
        start = window_start(period, now, offset_minutes)
        return {
            "period": period,
            "start": start.isoformat(),
            "agents": self.usage.summary(start),
            "installed": [a for a in self._overview[1] if a["installed"]],
            "claude_plan_usage": self.plan_usage,
        }

    def events_since(self, since: datetime, limit: int = 500) -> list[dict[str, Any]]:
        """``request.finished`` payloads for calls at or after ``since``, newest last."""
        return [self.event(c) for c in self.usage.calls(since)[-limit:]]

    def publish_turns(self, bus: LiveBus) -> None:
        """Publish how the agents' running turns changed since the last call."""
        if self.activity is None:
            return
        with self._turn_lock:
            turns = {t.request_id: t for t in self.activity.scan()}
            now = time.monotonic()
            for rid, (state, at) in self._hooked.items():
                # A permission prompt writes nothing to the log; the turn is still on.
                if state == "waiting" and rid not in turns and rid in self._turns and now - at < WAITING_TTL:
                    turns[rid] = self._turns[rid]
            for rid, turn in turns.items():
                before = self._turns.get(rid)
                state = self._hooked.get(rid, ("working", 0.0))[0]
                if before is None:
                    bus.publish("request.started", turn_event(turn, self.names.get(turn.agent, turn.agent), state))
                elif (before.model != turn.model and turn.model) or self._states.get(rid) != state:
                    bus.publish(
                        "request.activity", {"request_id": rid, "model": turn.model or turn.agent, "state": state}
                    )
                self._states[rid] = state
            for rid in self._turns.keys() - turns.keys():
                gone = self._turns[rid]
                self._states.pop(rid, None)
                self._hooked.pop(rid, None)
                bus.publish(
                    "request.ended", {"request_id": rid, "tool": gone.agent, "key_name": self.names.get(gone.agent)}
                )
            self._turns = turns

    def hook_event(self, bus: LiveBus, agent: str, event: str, session: str) -> None:
        """Apply one event an agent's hooks reported (see :mod:`.hook`), then publish what changed.

        ``Notification`` during a turn means it waits on the user (a permission
        prompt); a prompt or a tool starting or finishing means it works again;
        ``Stop`` / ``SessionEnd`` end it.
        """
        rid = f"agent:{agent}:{session}"
        with self._turn_lock:
            if event in ("Stop", "SessionEnd"):
                self._hooked.pop(rid, None)
            elif event == "Notification":
                if rid in self._turns:
                    self._hooked[rid] = ("waiting", time.monotonic())
            elif event in WORKING_HOOKS:
                self._hooked[rid] = ("working", time.monotonic())
        self.publish_turns(bus)

    def tail(self, bus: LiveBus, stop: threading.Event, interval: float = 2.0) -> None:
        """Publish agent calls as they're logged, and agent turns as they start and end."""
        first = True
        while True:
            try:
                self.publish_turns(bus)
            except Exception:
                logger.debug("coding agent activity scan failed", exc_info=True)
            if first:
                # The first read takes in the whole history (seconds); turns in progress show before it.
                first = False
                try:
                    self.refresh(force=True)
                except Exception:
                    logger.debug("coding tool scan failed", exc_info=True)
            if stop.wait(interval):
                break
            try:
                for call in sorted(self.refresh(force=True), key=lambda c: c.ts):
                    if datetime.now(timezone.utc) - call.ts < timedelta(minutes=30):
                        bus.publish("request.finished", self.event(call))
            except Exception:  # a bad log line must never stop the companion
                logger.debug("coding tool scan failed", exc_info=True)
