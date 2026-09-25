"""Built-in :class:`~prompture.infra.coding_agent_usage.UsageReader` for each coding agent.

Where each agent keeps per-call usage, and how it is counted once:

==============  =====================================================  ==========================
Agent           Files                                                  One call =
==============  =====================================================  ==========================
Claude Code     ``~/.claude/projects/**/*.jsonl``                      ``message.id`` (last line
                                                                       wins; resumed sessions
                                                                       copy messages)
Codex           ``~/.codex/sessions/**/rollout-*.jsonl``               ``token_usage_record``
                                                                       ``response_id``; older
                                                                       logs: a ``token_count``
                                                                       whose running total moved
Kimi Code       ``~/.kimi-code/sessions/*/*/agents/*/wire.jsonl``      ``usage.record`` with
                                                                       ``usageScope: "turn"``
Gemini CLI      ``~/.gemini/tmp/*/chats/session-*.json``               ``messages[].id``
Qwen Code       ``~/.qwen/projects/*/chats/*.jsonl``                   ``api_response``
                                                                       ``response_id``
OpenCode        ``~/.local/share/opencode/opencode.db`` (``message``)  message ``id``
Cline, Roo      ``<editor>/User/globalStorage/<ext>/tasks/*/``         ``api_req_started`` per
                ``ui_messages.json``                                   task + timestamp
Continue        ``~/.continue/dev_data/0.2.0/tokensGenerated.jsonl``   one line
==============  =====================================================  ==========================

Cursor and Antigravity are detected but keep usage on their servers, so they
report installed-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from .coding_agent_usage import (
    AgentCall,
    ChangedFiles,
    JsonlCursor,
    UsageReader,
    as_int,
    editor_storage_roots,
    env_path,
    estimate_cost,
    home,
    parse_ts,
    project_name,
    recent_files,
    register_usage_reader,
)

logger = logging.getLogger("prompture.coding_agents")


def _loads(line: str) -> Any:
    try:
        return json.loads(line)
    except ValueError:
        return None


def _plan_snapshot(
    agent: str, name: str, windows: dict[str, Any], observed_at: float, plan: str | None
) -> dict[str, Any]:
    from .rate_limits import LimitSnapshot

    snap = LimitSnapshot.from_dict({"windows": windows, "observed_at": observed_at, "source": "plan"}).to_dict()
    return {**snap, "tool": agent, "tool_name": name, "plan": plan}


def _percent_window(used: Any, resets_at: float | None) -> dict[str, Any] | None:
    if not isinstance(used, (int, float)):
        return None
    used = max(0.0, min(100.0, float(used)))
    return {"limit": 100, "remaining": round(100 - used), "resets_at": resets_at}


# ------------------------------------------------------------------ Claude Code

PLAN_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
PLAN_USAGE_AGENT = "claude-code/2.1.0"
PLAN_USAGE_INTERVAL = 300.0
PLAN_USAGE_BACKOFF = 600.0
_CLAUDE_WINDOWS = {"five_hour": "session_5h", "seven_day": "weekly"}


@register_usage_reader
class ClaudeCodeReader(UsageReader):
    """Claude Code session transcripts.

    Plan windows (5-hour session, weekly) are not written to disk; Claude Code
    reads them from an undocumented endpoint with its own OAuth token. Reading
    that token is using another app's credential, so it is **off unless**
    ``plan_usage=True`` or ``PROMPTURE_CLAUDE_PLAN_USAGE=1``. When on, the
    token is only read (never refreshed), only sent to ``api.anthropic.com``,
    at most every :data:`PLAN_USAGE_INTERVAL` seconds, and the last answer is
    cached in ``~/.prompture/cache/claude_plan_usage.json``.
    """

    agent = "claude"
    display_name = "Claude Code"

    def __init__(
        self, root: str | Path | None = None, *, plan_usage: bool | None = None, cache_dir: str | Path | None = None
    ) -> None:
        self.root = Path(root) if root else env_path("CLAUDE_CONFIG_DIR", home() / ".claude")
        if plan_usage is None:
            plan_usage = os.environ.get("PROMPTURE_CLAUDE_PLAN_USAGE", "").lower() in ("1", "true", "yes")
        self.plan_usage = plan_usage
        self.cache_file = Path(cache_dir or home() / ".prompture" / "cache") / "claude_plan_usage.json"
        self.cursor = JsonlCursor()
        self._plan: dict[str, Any] | None = None
        self._plan_next = 0.0

    def paths(self) -> list[Path]:
        return [self.root / "projects"]

    def read(self, since: datetime) -> Iterator[AgentCall]:
        for path in recent_files(self.root / "projects", "**/*.jsonl", since):
            for _, line in self.cursor.new_lines(path):
                if '"usage"' not in line or '"assistant"' not in line:
                    continue
                call = self._call(_loads(line))
                if call:
                    yield call

    def _call(self, entry: Any) -> AgentCall | None:
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            return None
        message = entry.get("message")
        if not isinstance(message, dict):
            return None
        usage, model, ts = message.get("usage"), message.get("model"), parse_ts(entry.get("timestamp"))
        if not isinstance(usage, dict) or not isinstance(model, str) or model.startswith("<") or ts is None:
            return None
        fresh, read = as_int(usage.get("input_tokens")), as_int(usage.get("cache_read_input_tokens"))
        write, output = as_int(usage.get("cache_creation_input_tokens")), as_int(usage.get("output_tokens"))
        if not (fresh or read or write or output):
            return None
        details = usage.get("output_tokens_details")
        cost, source = estimate_cost("claude", model, fresh_in=fresh, cache_read=read, cache_write=write, output=output)
        return AgentCall(
            id=f"claude:{message.get('id') or entry.get('uuid')}:{entry.get('requestId') or ''}",
            agent=self.agent,
            ts=ts,
            model=f"claude/{model}",
            input_tokens=fresh + read + write,
            output_tokens=output,
            cache_read_tokens=read,
            cache_write_tokens=write,
            reasoning_tokens=as_int(details.get("thinking_tokens")) if isinstance(details, dict) else 0,
            cost_usd=cost,
            cost_source=source,
            project=project_name(entry.get("cwd")),
            session=entry.get("sessionId"),
        )

    # -- plan windows (opt-in) ----------------------------------------------

    def _token(self) -> tuple[str, str | None] | None:
        try:
            data = json.loads((self.root / ".credentials.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        oauth = data.get("claudeAiOauth") if isinstance(data, dict) else None
        if not isinstance(oauth, dict) or not oauth.get("accessToken"):
            return None
        expires = oauth.get("expiresAt")
        if isinstance(expires, (int, float)) and expires / 1000 <= time.time():
            return None  # Claude Code refreshes it on its next run; we never do
        return str(oauth["accessToken"]), oauth.get("subscriptionType")

    def _load_cache(self) -> None:
        try:
            cached = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(cached, dict):
            self._plan = cached.get("usage") if isinstance(cached.get("usage"), dict) else None
            self._plan_next = float(cached.get("next") or 0.0)

    def _save_cache(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps({"usage": self._plan, "next": self._plan_next}), encoding="utf-8")
        except OSError:
            pass

    def _fetch_plan(self) -> None:
        creds = self._token()
        if creds is None:
            self._plan_next = time.time() + PLAN_USAGE_INTERVAL
            return
        token, plan = creds
        request = urllib.request.Request(
            PLAN_USAGE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "anthropic-beta": "oauth-2025-04-20",
                "Accept": "application/json",
                "User-Agent": PLAN_USAGE_AGENT,  # the endpoint only answers Claude Code, whose token this is
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:  # nosec B310 - fixed https URL
                body = json.loads(resp.read().decode())
            self._plan = {"observed_at": time.time(), "plan": plan, "body": body}
            self._plan_next = time.time() + PLAN_USAGE_INTERVAL
        except urllib.error.HTTPError as exc:
            retry = exc.headers.get("Retry-After") if exc.headers else None
            wait = float(retry) if retry and retry.isdigit() else PLAN_USAGE_BACKOFF
            self._plan_next = time.time() + max(wait, PLAN_USAGE_INTERVAL)
            logger.debug("Claude plan usage: HTTP %s", exc.code)
        except (OSError, ValueError) as exc:
            self._plan_next = time.time() + PLAN_USAGE_INTERVAL
            logger.debug("Claude plan usage failed: %s", exc)
        self._save_cache()

    def plan_limits(self) -> dict[str, dict[str, Any]]:
        if not self.plan_usage:
            return {}
        if self._plan is None and not self._plan_next:
            self._load_cache()
        if time.time() >= self._plan_next:
            self._fetch_plan()
        body = (self._plan or {}).get("body")
        if not isinstance(body, dict):
            return {}
        windows = {}
        for key, value in body.items():
            if not isinstance(value, dict) or not (key in _CLAUDE_WINDOWS or key.startswith("seven_day_")):
                continue
            resets = parse_ts(value.get("resets_at"))
            name = _CLAUDE_WINDOWS.get(key) or "weekly_" + key[len("seven_day_") :]
            window = _percent_window(value.get("utilization"), resets.timestamp() if resets else None)
            if window:
                windows[name] = window
        if not windows or self._plan is None:
            return {}
        snap = _plan_snapshot(
            self.agent, self.display_name, windows, float(self._plan.get("observed_at") or 0), self._plan.get("plan")
        )
        return {"claude/claude-code": snap}


# ------------------------------------------------------------------ Codex


class _CodexFile:
    __slots__ = ("cwd", "model", "records", "session")

    def __init__(self) -> None:
        self.model: str | None = None
        self.cwd: str | None = None
        self.session: str | None = None
        self.records = False  # newer logs: per-call token_usage_record lines


@register_usage_reader
class CodexReader(UsageReader):
    """Codex rollout logs. Plan windows come from its ``token_count`` events — no network."""

    agent = "codex"
    display_name = "Codex"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else env_path("CODEX_HOME", home() / ".codex")
        self.cursor = JsonlCursor()
        self.files: dict[Path, _CodexFile] = {}
        self.limits: tuple[datetime, dict[str, Any]] | None = None

    def paths(self) -> list[Path]:
        return [self.root / "sessions", self.root / "archived_sessions"]

    def read(self, since: datetime) -> Iterator[AgentCall]:
        for root in self.paths():
            for path in recent_files(root, "**/*.jsonl", since):
                state = self.files.setdefault(path, _CodexFile())
                lines = list(self.cursor.new_lines(path))
                if not state.records and any('"token_usage_record"' in line for _, line in lines):
                    state.records = True
                for _, line in lines:
                    call = self._line(line, state)
                    if call:
                        yield call

    def _line(self, line: str, state: _CodexFile) -> AgentCall | None:
        if '"session_meta"' in line or '"turn_context"' in line:
            entry = _loads(line)
            payload = entry.get("payload") if isinstance(entry, dict) else None
            if isinstance(payload, dict):
                if entry.get("type") == "session_meta":
                    state.session = payload.get("id") or state.session
                    state.cwd = payload.get("cwd") or state.cwd
                elif entry.get("type") == "turn_context":
                    state.model = payload.get("model") or state.model
                    state.cwd = payload.get("cwd") or state.cwd
            return None
        if '"token_usage_record"' in line:
            entry = _loads(line)
            payload = entry.get("payload") if isinstance(entry, dict) else None
            ts = parse_ts(entry.get("timestamp")) if isinstance(entry, dict) else None
            if not isinstance(payload, dict) or ts is None or not isinstance(payload.get("usage"), dict):
                return None
            rid = payload.get("response_id") or f"{ts.isoformat()}:{payload.get('turn_id')}"
            return self._call(f"codex:{rid}", ts, payload["usage"], state)
        if '"token_count"' not in line:
            return None
        entry = _loads(line)
        payload = entry.get("payload") if isinstance(entry, dict) else None
        ts = parse_ts(entry.get("timestamp")) if isinstance(entry, dict) else None
        if not isinstance(payload, dict) or payload.get("type") != "token_count" or ts is None:
            return None
        limits = payload.get("rate_limits")
        if isinstance(limits, dict) and (self.limits is None or ts >= self.limits[0]):
            self.limits = (ts, limits)
        info = payload.get("info")
        if state.records or not isinstance(info, dict) or not isinstance(info.get("last_token_usage"), dict):
            return None
        total = info.get("total_token_usage")
        # Codex repeats a token_count without a new call, and forked sessions
        # replay their parent's events: the running total at a moment is unique.
        running = as_int(total.get("total_tokens")) if isinstance(total, dict) else 0
        return self._call(f"codex:{ts.isoformat()}:{running}", ts, info["last_token_usage"], state)

    def _call(self, call_id: str, ts: datetime, usage: dict[str, Any], state: _CodexFile) -> AgentCall | None:
        prompt, cached = as_int(usage.get("input_tokens")), as_int(usage.get("cached_input_tokens"))
        written, output = as_int(usage.get("cache_write_input_tokens")), as_int(usage.get("output_tokens"))
        if not (prompt or output):
            return None
        model = state.model or "codex"
        cost, source = estimate_cost(
            "openai",
            model,
            fresh_in=max(0, prompt - cached - written),
            cache_read=cached,
            cache_write=written,
            output=output,
        )
        return AgentCall(
            id=call_id,
            agent=self.agent,
            ts=ts,
            model=f"openai/{model}",
            input_tokens=prompt,  # cached tokens are already part of input_tokens
            output_tokens=output,
            cache_read_tokens=cached,
            cache_write_tokens=written,
            reasoning_tokens=as_int(usage.get("reasoning_output_tokens")),
            cost_usd=cost,
            cost_source=source,
            project=project_name(state.cwd),
            session=state.session,
        )

    def plan_limits(self) -> dict[str, dict[str, Any]]:
        if not self.limits:
            return {}
        ts, limits = self.limits
        windows = {}
        for key in ("primary", "secondary"):
            w = limits.get(key)
            if not isinstance(w, dict):
                continue
            minutes = as_int(w.get("window_minutes"))
            name = {300: "session_5h", 10080: "weekly"}.get(minutes, f"{minutes}m")
            resets = w.get("resets_at")
            window = _percent_window(w.get("used_percent"), float(resets) if isinstance(resets, (int, float)) else None)
            if window:
                windows[name] = window
        if not windows:
            return {}
        return {
            "openai/codex": _plan_snapshot(
                self.agent, self.display_name, windows, ts.timestamp(), limits.get("plan_type")
            )
        }


# ------------------------------------------------------------------ Kimi Code

#: "You've reached your 5-hour usage limit" / "... weekly usage limit" in a failed turn.
_KIMI_LIMIT = re.compile(r"reached your (5-hour|weekly|daily|monthly) usage limit", re.IGNORECASE)
_KIMI_WINDOWS = {
    "5-hour": ("session_5h", 5 * 3600),
    "weekly": ("weekly", 7 * 86400),
    "daily": ("daily", 86400),
    "monthly": ("monthly", 30 * 86400),
}


@register_usage_reader
class KimiCodeReader(UsageReader):
    """Kimi Code session wire logs (main agent and subagents each have their own file).

    Kimi Code doesn't log how much of its plan is left, but it does log the
    error when a window runs out ("You've reached your 5-hour usage limit").
    That window is reported as used up until the latest time it can reset —
    the hit time plus the window length, since the window's start isn't logged.
    """

    agent = "kimi"
    display_name = "Kimi Code"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else env_path("KIMI_CODE_HOME", home() / ".kimi-code")
        self.cursor = JsonlCursor()
        self.limit_hits: dict[str, float] = {}  # window name -> latest hit (epoch seconds)

    def _limit_hit(self, line: str) -> None:
        match = _KIMI_LIMIT.search(line)
        entry = _loads(line) if match else None
        ts = parse_ts(entry.get("time")) if isinstance(entry, dict) else None
        if match and ts is not None:
            name = _KIMI_WINDOWS[match.group(1).lower()][0]
            self.limit_hits[name] = max(self.limit_hits.get(name, 0.0), ts.timestamp())

    def plan_limits(self) -> dict[str, dict[str, Any]]:
        now = time.time()
        windows = {}
        latest = 0.0
        for name, seconds in _KIMI_WINDOWS.values():
            hit = self.limit_hits.get(name)
            if hit and hit + seconds > now:
                windows[name] = {"limit": 100, "remaining": 0, "resets_at": hit + seconds}
                latest = max(latest, hit)
        if not windows:
            return {}
        snap = _plan_snapshot(self.agent, self.display_name, windows, latest, None)
        return {"moonshot/kimi-code": {**snap, "estimated_reset": True}}

    def paths(self) -> list[Path]:
        return [self.root / "sessions"]

    def read(self, since: datetime) -> Iterator[AgentCall]:
        sessions = self.root / "sessions"
        for path in recent_files(sessions, "*/*/agents/*/wire.jsonl", since):
            # sessions/wd_<folder>_<hash12>/session_<uuid>/agents/<agent>/wire.jsonl
            work, session = path.parents[3].name, path.parents[2].name
            project = work[3:].rsplit("_", 1)[0] if work.startswith("wd_") else None
            rel = f"{session}/{path.parent.name}"
            for offset, line in self.cursor.new_lines(path):
                if "usage limit" in line:
                    self._limit_hit(line)
                    continue
                if '"usage.record"' not in line:
                    continue
                entry = _loads(line)
                if not isinstance(entry, dict) or entry.get("usageScope") not in (None, "turn"):
                    continue  # "session" records are compaction/summary totals, not calls
                usage, ts = entry.get("usage"), parse_ts(entry.get("time"))
                if not isinstance(usage, dict) or ts is None:
                    continue
                fresh, read = as_int(usage.get("inputOther")), as_int(usage.get("inputCacheRead"))
                write, output = as_int(usage.get("inputCacheCreation")), as_int(usage.get("output"))
                if not (fresh or read or write or output):
                    continue
                raw_model = str(entry.get("model") or "kimi")
                model = raw_model.split("/", 1)[-1]
                cost, source = estimate_cost(
                    "moonshot", model, fresh_in=fresh, cache_read=read, cache_write=write, output=output
                )
                yield AgentCall(
                    id=f"kimi:{rel}:{offset}",
                    agent=self.agent,
                    ts=ts,
                    model=f"moonshot/{model}",
                    input_tokens=fresh + read + write,
                    output_tokens=output,
                    cache_read_tokens=read,
                    cache_write_tokens=write,
                    cost_usd=cost,
                    cost_source=source,
                    project=project,
                    session=session.removeprefix("session_"),
                )


# ------------------------------------------------------------------ Gemini CLI


@register_usage_reader
class GeminiCliReader(UsageReader):
    """Gemini CLI chat files (one JSON document per session, rewritten as it grows)."""

    agent = "gemini"
    display_name = "Gemini CLI"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else home() / ".gemini"
        self.files = ChangedFiles()
        self._projects: dict[str, str] | None = None

    def paths(self) -> list[Path]:
        return [self.root / "tmp"]

    def _project(self, folder: str) -> str | None:
        """Folder names are project names, or (older layout) a SHA-256 of the project path."""
        if len(folder) != 64 or any(ch not in "0123456789abcdef" for ch in folder):
            return folder
        if self._projects is None:
            self._projects = {}
            try:
                known = json.loads((self.root / "projects.json").read_text(encoding="utf-8")).get("projects") or {}
                for path, name in known.items():
                    self._projects[hashlib.sha256(path.encode()).hexdigest()] = name
            except (OSError, ValueError, AttributeError):
                pass
        return self._projects.get(folder)

    def read(self, since: datetime) -> Iterator[AgentCall]:
        for path in recent_files(self.root / "tmp", "*/chats/*.json", since):
            if not self.files.changed(path):
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.files.seen.pop(path, None)  # caught mid-write; try again next scan
                continue
            project = self._project(path.parent.parent.name)
            messages = doc.get("messages") if isinstance(doc, dict) else None
            for msg in messages if isinstance(messages, list) else []:
                if not isinstance(msg, dict) or msg.get("type") != "gemini" or not isinstance(msg.get("tokens"), dict):
                    continue
                t, ts, model = msg["tokens"], parse_ts(msg.get("timestamp")), msg.get("model")
                if ts is None or not isinstance(model, str) or not msg.get("id"):
                    continue
                prompt, cached = as_int(t.get("input")), as_int(t.get("cached"))
                thoughts, output = as_int(t.get("thoughts")), as_int(t.get("output"))
                if not (prompt or output):
                    continue
                cost, source = estimate_cost(
                    "google", model, fresh_in=max(0, prompt - cached), cache_read=cached, output=output + thoughts
                )
                yield AgentCall(
                    id=f"gemini:{msg['id']}",
                    agent=self.agent,
                    ts=ts,
                    model=f"google/{model}",
                    input_tokens=prompt,
                    output_tokens=output + thoughts,
                    cache_read_tokens=cached,
                    reasoning_tokens=thoughts,
                    cost_usd=cost,
                    cost_source=source,
                    project=project,
                    session=doc.get("sessionId") if isinstance(doc, dict) else None,
                )


# ------------------------------------------------------------------ Qwen Code


@register_usage_reader
class QwenCodeReader(UsageReader):
    """Qwen Code chat logs; each model response is logged as an ``api_response`` telemetry event."""

    agent = "qwen"
    display_name = "Qwen Code"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else home() / ".qwen"
        self.cursor = JsonlCursor()

    def paths(self) -> list[Path]:
        return [self.root / "projects"]

    def read(self, since: datetime) -> Iterator[AgentCall]:
        for path in recent_files(self.root / "projects", "*/chats/*.jsonl", since):
            for _, line in self.cursor.new_lines(path):
                if "api_response" not in line:
                    continue
                entry = _loads(line)
                event = (entry.get("systemPayload") or {}).get("uiEvent") if isinstance(entry, dict) else None
                if not isinstance(event, dict) or event.get("event.name") != "qwen-code.api_response":
                    continue
                if event.get("status_code") not in (None, 200):
                    continue
                ts = parse_ts(event.get("event.timestamp")) or parse_ts(entry.get("timestamp"))
                prompt, cached = as_int(event.get("input_token_count")), as_int(event.get("cached_content_token_count"))
                thoughts, output = as_int(event.get("thoughts_token_count")), as_int(event.get("output_token_count"))
                if ts is None or not event.get("response_id") or not (prompt or output):
                    continue
                model = str(event.get("model") or "qwen")
                cost, source = estimate_cost(
                    "qwen", model, fresh_in=max(0, prompt - cached), cache_read=cached, output=output + thoughts
                )
                yield AgentCall(
                    id=f"qwen:{event['response_id']}",
                    agent=self.agent,
                    ts=ts,
                    model=f"qwen/{model}",
                    input_tokens=prompt,
                    output_tokens=output + thoughts,
                    cache_read_tokens=cached,
                    reasoning_tokens=thoughts,
                    cost_usd=cost,
                    cost_source=source,
                    project=project_name(entry.get("cwd")),
                    session=entry.get("sessionId"),
                )


# ------------------------------------------------------------------ OpenCode


@register_usage_reader
class OpenCodeReader(UsageReader):
    """OpenCode's SQLite store (``message`` rows), plus the JSON files older versions wrote."""

    agent = "opencode"
    display_name = "OpenCode"

    def __init__(self, root: str | Path | None = None) -> None:
        base = Path(os.environ.get("XDG_DATA_HOME") or home() / ".local" / "share")
        self.root = Path(root) if root else base / "opencode"
        self.watermark = 0  # message.time_updated (ms) already read
        self.files = ChangedFiles()

    def paths(self) -> list[Path]:
        return [self.root / "opencode.db", self.root / "storage" / "message"]

    def read(self, since: datetime) -> Iterator[AgentCall]:
        yield from self._db(since)
        for path in recent_files(self.root / "storage" / "message", "*/*.json", since):
            if self.files.changed(path):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                call = self._call(data.get("id"), data.get("sessionID"), data)
                if call:
                    yield call

    def _db(self, since: datetime) -> Iterator[AgentCall]:
        db = self.root / "opencode.db"
        if not db.is_file():
            return
        start = max(self.watermark, int(since.timestamp() * 1000))
        try:
            # mode=ro (not immutable) so rows still in the WAL are visible.
            con = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=2)
            try:
                rows = con.execute(
                    "SELECT id, session_id, time_updated, data FROM message WHERE time_updated >= ? ORDER BY time_updated",
                    (start,),
                ).fetchall()
            finally:
                con.close()
        except sqlite3.Error as exc:
            logger.debug("OpenCode database not readable: %s", exc)
            return
        for msg_id, session_id, updated, data in rows:
            self.watermark = max(self.watermark, as_int(updated))
            call = self._call(msg_id, session_id, _loads(data) if isinstance(data, str) else None)
            if call:
                yield call

    def _call(self, msg_id: Any, session_id: Any, data: Any) -> AgentCall | None:
        if not isinstance(data, dict) or data.get("role") != "assistant" or not msg_id:
            return None
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            return None
        cache = tokens.get("cache") if isinstance(tokens.get("cache"), dict) else {}
        fresh, output = as_int(tokens.get("input")), as_int(tokens.get("output"))
        read, write, reasoning = as_int(cache.get("read")), as_int(cache.get("write")), as_int(tokens.get("reasoning"))
        if not (fresh or read or write or output):
            return None
        times = data.get("time") if isinstance(data.get("time"), dict) else {}
        ts = parse_ts(times.get("completed") or times.get("created"))
        if ts is None:
            return None
        provider, model = str(data.get("providerID") or "opencode"), str(data.get("modelID") or "unknown")
        reported = data.get("cost")
        if isinstance(reported, (int, float)) and reported > 0:
            cost, source = float(reported), "reported"
        else:
            cost, source = estimate_cost(
                provider, model, fresh_in=fresh, cache_read=read, cache_write=write, output=output + reasoning
            )
        path = data.get("path") if isinstance(data.get("path"), dict) else {}
        return AgentCall(
            id=f"opencode:{msg_id}",
            agent=self.agent,
            ts=ts,
            model=f"{provider}/{model}",
            input_tokens=fresh + read + write,
            output_tokens=output + reasoning,
            cache_read_tokens=read,
            cache_write_tokens=write,
            reasoning_tokens=reasoning,
            cost_usd=cost,
            cost_source=source,
            project=project_name(path.get("root") or path.get("cwd")),
            session=str(session_id) if session_id else None,
        )


# ------------------------------------------------------------------ Cline and Roo Code


@register_usage_reader
class ClineReader(UsageReader):
    """Cline tasks in any VS Code-family editor (and the Cline CLI's data folder)."""

    agent = "cline"
    display_name = "Cline"
    extension = "saoudrizwan.claude-dev"

    def __init__(self, roots: list[Path] | None = None) -> None:
        self.roots = roots if roots is not None else self._default_roots()
        self.files = ChangedFiles()

    def _default_roots(self) -> list[Path]:
        editors = [r / "User" / "globalStorage" / self.extension for r in editor_storage_roots()]
        return [*editors, home() / ".cline" / "data"]

    def paths(self) -> list[Path]:
        return [r / "tasks" for r in self.roots]

    def _model(self, task: Path) -> str:
        """Newer versions record the model per task; older ones don't."""
        try:
            meta = json.loads((task / "task_metadata.json").read_text(encoding="utf-8"))
            usage = meta.get("model_usage") or []
            last = usage[-1] if usage else {}
            if last.get("model_id"):
                return f"{last.get('model_provider_id') or 'unknown'}/{last['model_id']}"
        except (OSError, ValueError, AttributeError, IndexError, TypeError):
            pass
        return f"{self.agent}/unknown"

    def read(self, since: datetime) -> Iterator[AgentCall]:
        for tasks in self.paths():
            for path in recent_files(tasks, "*/ui_messages.json", since):
                if not self.files.changed(path):
                    continue
                try:
                    messages = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self.files.seen.pop(path, None)
                    continue
                task = path.parent
                model = self._model(task)
                for msg in messages if isinstance(messages, list) else []:
                    if not isinstance(msg, dict) or msg.get("say") != "api_req_started":
                        continue
                    info, ts = _loads(msg.get("text") or ""), parse_ts(msg.get("ts"))
                    if not isinstance(info, dict) or ts is None:
                        continue
                    fresh, output = as_int(info.get("tokensIn")), as_int(info.get("tokensOut"))
                    read, write = as_int(info.get("cacheReads")), as_int(info.get("cacheWrites"))
                    if not (fresh or read or write or output):
                        continue
                    reported = info.get("cost")
                    has_cost = isinstance(reported, (int, float))
                    yield AgentCall(
                        id=f"{self.agent}:{task.name}:{msg.get('ts')}",
                        agent=self.agent,
                        ts=ts,
                        model=model,
                        input_tokens=fresh + read + write,
                        output_tokens=output,
                        cache_read_tokens=read,
                        cache_write_tokens=write,
                        cost_usd=float(reported) if has_cost else 0.0,
                        cost_source="reported" if has_cost else "unknown",
                        session=task.name,
                    )


@register_usage_reader
class RooCodeReader(ClineReader):
    """Roo Code (a Cline fork) keeps tasks in the same layout."""

    agent = "roo-code"
    display_name = "Roo Code"
    extension = "rooveterinaryinc.roo-cline"

    def _default_roots(self) -> list[Path]:
        return [r / "User" / "globalStorage" / self.extension for r in editor_storage_roots()]


# ------------------------------------------------------------------ Continue


@register_usage_reader
class ContinueReader(UsageReader):
    """Continue's local dev-data log of generated tokens."""

    agent = "continue"
    display_name = "Continue"

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else env_path("CONTINUE_GLOBAL_DIR", home() / ".continue")
        self.cursor = JsonlCursor()

    def paths(self) -> list[Path]:
        return [self.root / "dev_data"]

    def read(self, since: datetime) -> Iterator[AgentCall]:
        for path in recent_files(self.root / "dev_data", "*/tokensGenerated.jsonl", since):
            for offset, line in self.cursor.new_lines(path):
                entry = _loads(line)
                if not isinstance(entry, dict):
                    continue
                ts = parse_ts(entry.get("timestamp"))
                prompt, output = as_int(entry.get("promptTokens")), as_int(entry.get("generatedTokens"))
                if ts is None or not (prompt or output):
                    continue
                provider, model = str(entry.get("provider") or "unknown"), str(entry.get("model") or "unknown")
                cost, source = estimate_cost(provider, model, fresh_in=prompt, output=output)
                yield AgentCall(
                    id=f"continue:{path.parent.name}:{offset}",
                    agent=self.agent,
                    ts=ts,
                    model=f"{provider}/{model}",
                    input_tokens=prompt,
                    output_tokens=output,
                    cost_usd=cost,
                    cost_source=source,
                )


# ------------------------------------------------------------------ detected only


@register_usage_reader
class CursorReader(UsageReader):
    """Cursor keeps billed usage on its servers; only its presence is detected."""

    agent = "cursor"
    display_name = "Cursor"
    has_local_usage = False

    def paths(self) -> list[Path]:
        return [r for r in editor_storage_roots() if r.name == "Cursor"] + [home() / ".cursor"]


@register_usage_reader
class AntigravityReader(UsageReader):
    """Antigravity stores conversations encrypted; only its presence is detected."""

    agent = "antigravity"
    display_name = "Antigravity"
    has_local_usage = False

    def paths(self) -> list[Path]:
        return [r for r in editor_storage_roots() if r.name == "Antigravity"] + [home() / ".gemini" / "antigravity"]
