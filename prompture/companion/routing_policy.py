"""How the router decides where a coding CLI's request goes, and when that changes.

:class:`RoutePolicy` turns ``routes.json`` plus what it has seen of a session
into a :class:`Decision` for each request. In order:

1. **Native-only requests** (quota checks, anything that isn't a model turn)
   always go to the vendor unchanged.
2. **The task's destination** is kept: once a session's main turns go
   somewhere, its later turns and tool follow-ups go there too, so the prompt
   cache keeps working and the model doesn't change mid-task. Escalation and
   task budgets change it on purpose (below); editing rules takes effect on
   the next task.
3. **Explicit rules**: the project's rules for the tool, then the tool's own
   (a request *kind* first, then a requested-model pattern).
4. **Presets**: the project's preset for the tool, the project's, the tool's,
   then the default. Built in: ``quality`` (nothing changes), ``balanced``
   (titles on the vendor's small model, summaries on its mid one) and
   ``economy`` (background work on the small model, everything else one tier
   down), all on the CLI's own login. See :data:`PRESETS`.
5. Otherwise the request passes through.

A destination is either a **native** model (the CLI's own vendor and login,
another model: ``native:small``, ``native:mid``, ``native:large`` or
``native:<model id>``) or any **Prompture model** (``ollama/qwen3``,
``auto/cheap``, ``combo/…``). Native models never move a request *up* a tier
by preset, and a Prompture model is only used when it can take the request:
tools, images, structured output, context size, and protocol (a Responses
request chained with ``previous_response_id`` only works on its own backend).

**Fallback** (failure recovery) and **escalation** (quality) are separate:

- A routed request that fails before anything reached the CLI (provider
  down, rate limit, incompatible model) is retried on ``fallback.models``
  and finally on the vendor's own path, which is always allowed.
- A task whose tool results keep failing (errors, failed tests, the same
  failing action repeated) is moved to a stronger model: back to the model
  the CLI asked for, or to ``escalation.to``. So is one a reviewer rejects
  (``POST /v1/router/sessions/<tool>/<session>/escalate``). A permission
  prompt, or the user declining a tool, is the user deciding, not the model
  failing, and never counts.

Both are bounded per task by ``budget``: ``task_attempts`` (fallback attempts
plus escalations) and ``task_usd`` (routed spend). Past either, the task stays
on the vendor's own path (or, with ``on_exceed: "stop"``, its routed requests
are refused).
"""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from typing import Any

#: Request kinds the router never changes: a quota or connectivity check must reach the vendor.
NATIVE_ONLY_KINDS = {"probe"}
#: Kinds that are part of the task itself; they share the task's destination.
TASK_KINDS = {"main", "tool_result"}
BACKGROUND_KINDS = {"title", "probe", "compaction"}

#: Built-in presets: request kind → destination. ``native:<tier>`` keeps the CLI's login.
PRESETS: dict[str, dict[str, str]] = {
    "quality": {},
    "balanced": {"title": "native:small", "compaction": "native:mid"},
    "economy": {
        "title": "native:small",
        "compaction": "native:small",
        "main": "native:mid",
        "tool_result": "native:mid",
    },
}
PRESET_NAMES = tuple(PRESETS)

TIERS = ("small", "mid", "large")
#: Native models when none of a tier has been seen yet (Anthropic's aliases are stable).
NATIVE_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {"small": "claude-haiku-4-5", "mid": "claude-sonnet-5", "large": "claude-opus-5-5"},
    "openai": {},
}

DEFAULT_ESCALATION = {"enabled": True, "after_failures": 3, "after_repeats": 2, "to": "native"}
DEFAULT_BUDGET = {"task_usd": None, "task_attempts": 6, "on_exceed": "native"}
#: Most attempts one request makes, whatever the task budget allows.
MAX_ATTEMPTS_PER_REQUEST = 3
#: A task (session) nobody has touched this long is forgotten.
SESSION_TTL = 6 * 3600.0


def tier_of(dialect: str, model: str) -> str | None:
    """Which tier a vendor model is in, from its name."""
    m = model.lower()
    if dialect == "anthropic":
        for tier, word in (("small", "haiku"), ("mid", "sonnet"), ("large", "opus")):
            if word in m:
                return tier
        return None
    if not m.startswith(("gpt", "o1", "o3", "o4", "codex")):
        return None
    return "small" if ("mini" in m or "nano" in m) else "large" if ("pro" in m.split("-")) else "mid"


def _rank(tier: str | None) -> int:
    return TIERS.index(tier) if tier in TIERS else -1


# ------------------------------------------------------------------ decisions


@dataclasses.dataclass(frozen=True)
class Decision:
    """Where one request goes, and why.

    ``target`` is a Prompture model, ``native`` a model on the CLI's own
    vendor; both ``None`` means unchanged.
    """

    source: str = "none"  # none | native_only | sticky | kind | model | preset | fallback | escalation | budget
    reason: str = "Passed through unchanged."
    match: str | None = None
    target: str | None = None
    native: str | None = None
    preset: str | None = None
    stop: bool = False  # refuse the request (task budget exceeded, on_exceed=stop)

    @property
    def route(self) -> str:
        return "routed" if self.target else "native" if self.native else "passthrough"

    def rule(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "match": self.match,
            "target": self.target or (f"native:{self.native}" if self.native else None),
            "preset": self.preset,
            "reason": self.reason,
        }

    def with_reason(self, reason: str, **changes: Any) -> Decision:
        return dataclasses.replace(self, reason=reason, **changes)


PASS = Decision()


@dataclasses.dataclass
class Needs:
    """What a request needs from the model that answers it (for compatibility checks)."""

    tools: bool = False
    images: bool = False
    structured: bool = False
    tokens: int = 0
    native_only: str | None = None  # why only the vendor's own backend can answer

    @classmethod
    def of(cls, dialect: str, body: dict[str, Any]) -> Needs:
        raw = json.dumps(body, default=str)
        needs = cls(tools=bool(body.get("tools")), tokens=len(raw) // 4)
        needs.images = '"type": "image"' in raw or '"input_image"' in raw or '"type":"image"' in raw
        if dialect == "openai":
            fmt = (body.get("text") or {}).get("format") if isinstance(body.get("text"), dict) else None
            needs.structured = isinstance(fmt, dict) and fmt.get("type") == "json_schema"
            if body.get("previous_response_id"):
                needs.native_only = "it continues a stored response (previous_response_id)"
        else:
            needs.structured = bool(body.get("output_format"))
        return needs


def compatible(target: str, needs: Needs) -> str | None:
    """Why *target* can't take a request with *needs*, or ``None`` when it can (or nobody knows)."""
    if needs.native_only:
        return f"only the vendor can answer: {needs.native_only}"
    if "/" not in target or target.split("/", 1)[0] in ("auto", "combo", "fusion"):
        return None  # virtual models fall back across their own targets
    try:
        from ..infra.capabilities import get_capabilities
        from ..infra.model_rates import get_model_capabilities

        caps = get_capabilities(target)
        provider, model_id = target.split("/", 1)
        info = get_model_capabilities(provider, model_id)
    except Exception:
        return None
    if needs.tools and caps.tool_use is False:
        return f"{target} can't call tools"
    if needs.images and caps.vision is False:
        return f"{target} can't read images"
    if needs.structured and caps.json_schema is False:
        return f"{target} can't return structured output"
    window = info.context_window if info else None
    if window and needs.tokens > window * 0.9:
        return f"the request (~{needs.tokens:,} tokens) doesn't fit {target}'s {window:,}-token context"
    return None


# ------------------------------------------------------------------ outcomes of tool calls

_FAILED_OUTPUT = re.compile(
    r"(exit code:? *[1-9]|exited with code [1-9]|\"exit_code\": *[1-9]|\b\d+ failed\b|\bFAILED\b|"
    r"Traceback \(most recent call last\)|\berror(\[E\d+\])?:|npm ERR!|BUILD FAILED|compilation failed)",
    re.IGNORECASE,
)
#: The user said no to a tool, or a permission prompt is pending: not the model failing.
_USER_DECISION = re.compile(
    r"(doesn't want to proceed|does not want to proceed|user (rejected|denied|declined)|"
    r"permission (was )?denied by the user|request interrupted by user|waiting for (your|user) (approval|permission))",
    re.IGNORECASE,
)


@dataclasses.dataclass
class ToolOutcome:
    """The newest tool result in a request: failed or not, and which action produced it."""

    failed: bool
    user_decision: bool
    action: str | None  # a hash of the tool name and input
    name: str | None


def _block_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_block_text(b.get("text") if isinstance(b, dict) else b) for b in content)
    return str(content or "")


def _action_hash(name: str, arguments: Any) -> str:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha1(f"{name}\0{raw}".encode()).hexdigest()[:16]


def tool_outcome(dialect: str, body: dict[str, Any]) -> ToolOutcome | None:
    """The outcome of the tool call a ``tool_result`` request reports, or ``None``."""
    if dialect == "anthropic":
        raw_messages = body.get("messages")
        messages: list[Any] = raw_messages if isinstance(raw_messages, list) else []
        if len(messages) < 2 or not isinstance(messages[-1], dict):
            return None
        results = [
            b for b in messages[-1].get("content") or [] if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        if not results:
            return None
        uses = {
            b.get("id"): b
            for b in (messages[-2].get("content") or [] if isinstance(messages[-2], dict) else [])
            if isinstance(b, dict) and b.get("type") == "tool_use"
        }
        texts = [_block_text(r.get("content")) for r in results]
        failed = any(r.get("is_error") for r in results) or any(_FAILED_OUTPUT.search(t) for t in texts)
        use = uses.get(results[-1].get("tool_use_id")) or {}
        name = use.get("name")
        return ToolOutcome(
            failed=failed,
            user_decision=any(_USER_DECISION.search(t) for t in texts),
            action=_action_hash(str(name), use.get("input")) if name else None,
            name=name,
        )
    raw_items = body.get("input")
    items: list[Any] = raw_items if isinstance(raw_items, list) else []
    outputs = []
    for item in reversed(items):
        if isinstance(item, dict) and item.get("type") in ("function_call_output", "custom_tool_call_output"):
            outputs.append(item)
        else:
            break
    if not outputs:
        return None
    calls = {
        i.get("call_id"): i
        for i in items
        if isinstance(i, dict) and i.get("type") in ("function_call", "custom_tool_call", "local_shell_call")
    }
    texts = [_block_text(o.get("output")) for o in outputs]
    call = calls.get(outputs[0].get("call_id")) or {}
    name = call.get("name") or call.get("type")
    return ToolOutcome(
        failed=any(_FAILED_OUTPUT.search(t) for t in texts),
        user_decision=any(_USER_DECISION.search(t) for t in texts),
        action=_action_hash(str(name), call.get("arguments") or call.get("input") or call.get("action"))
        if name
        else None,
        name=name,
    )


# ------------------------------------------------------------------ sessions


@dataclasses.dataclass
class Task:
    """What the router remembers about one CLI session (a task)."""

    tool: str
    session: str
    project: str | None = None
    decision: Decision | None = None  # the task's destination, kept across its turns
    pinned_for: str | None = None  # the model the CLI asked for when it was chosen
    rules_version: str | None = None  # the routes.json it was chosen under
    served: str | None = None  # model that last answered a task turn (to spot switches)
    failures: int = 0  # tool results failing in a row
    last_action: str | None = None
    repeats: int = 0  # the same failing action in a row
    attempts: int = 0  # fallback attempts + escalations
    spent_usd: float = 0.0  # new API spend of its routed calls
    escalations: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    waiting: bool = False  # a permission prompt is open
    cache_ratio: float | None = None  # prompt share the original path read from cache
    seen: float = dataclasses.field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "session": self.session,
            "project": self.project,
            "destination": self.decision.rule() if self.decision else None,
            "served": self.served,
            "failures": self.failures,
            "repeats": self.repeats,
            "attempts": self.attempts,
            "spent_usd": round(self.spent_usd, 6),
            "escalations": self.escalations,
            "waiting": self.waiting,
        }


# ------------------------------------------------------------------ policy


class RoutePolicy:
    """Decides each request's destination; see the module docstring. Thread-safe.

    ``rules()`` returns the current ``routes.json`` (normalized).
    """

    def __init__(self, rules: Callable[[], dict[str, Any]], *, clock: Callable[[], float] = time.monotonic) -> None:
        self.rules = rules
        self.clock = clock
        self._tasks: dict[tuple[str, str], Task] = {}
        self._seen_models: dict[str, dict[str, str]] = {}  # dialect -> tier -> model
        self._lock = threading.Lock()

    # -- tasks ----------------------------------------------------------------

    def task(self, tool: str, session: str | None, project: str | None = None) -> Task | None:
        if not session:
            return None
        now = self.clock()
        with self._lock:
            for key in [k for k, t in self._tasks.items() if now - t.seen > SESSION_TTL]:
                del self._tasks[key]
            task = self._tasks.get((tool, session))
            if task is None:
                task = self._tasks[(tool, session)] = Task(tool, session, project)
            task.seen = now
            if project and not task.project:
                task.project = project
            return task

    def tasks(self) -> list[Task]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: -t.seen)

    def find(self, tool: str, session: str) -> Task | None:
        with self._lock:
            return self._tasks.get((tool, session))

    def set_waiting(self, tool: str, session: str, waiting: bool) -> None:
        """A permission prompt opened (or was answered): the task waits on the user, not the model."""
        with self._lock:
            task = self._tasks.get((tool, session))
            if task is not None:
                task.waiting = waiting

    # -- native models ---------------------------------------------------------

    def observe_model(self, dialect: str, model: str) -> None:
        """Remember a model the CLI itself asked for, as the current one of its tier."""
        tier = tier_of(dialect, model)
        if tier:
            with self._lock:
                self._seen_models.setdefault(dialect, {})[tier] = model

    def native_model(self, dialect: str, tier: str) -> str | None:
        with self._lock:
            seen = self._seen_models.get(dialect, {}).get(tier)
        return seen or NATIVE_DEFAULTS.get(dialect, {}).get(tier)

    def _native(self, spec: str, dialect: str, requested: str) -> tuple[str | None, str | None]:
        """``native:<tier|model>`` → ``(model, why not)``; a tier is never above the requested model."""
        want = spec.split(":", 1)[1] if ":" in spec else ""
        if want not in TIERS:
            return (want or None), None
        have = tier_of(dialect, requested)
        if have is not None and _rank(want) >= _rank(have):
            return None, f"{requested} is already {have}-tier"
        model = self.native_model(dialect, want)
        if model is None:
            return None, f"no {want}-tier model seen yet"
        return (None, f"{requested} is already {model}") if model == requested else (model, None)

    # -- the decision ---------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        data = self.rules()
        return {
            "fallback": {"models": [], "allow_paid": False, **(data.get("fallback") or {})},
            "escalation": {**DEFAULT_ESCALATION, **(data.get("escalation") or {})},
            "budget": {**DEFAULT_BUDGET, **(data.get("budget") or {})},
        }

    def decide(
        self,
        tool: str,
        dialect: str,
        model: str,
        kind: str,
        *,
        project: str | None = None,
        session: str | None = None,
        needs: Needs | None = None,
        original_billing: str = "api",
    ) -> Decision:
        """Where one request goes. See the module docstring for the order."""
        needs = needs or Needs()
        if kind in NATIVE_ONLY_KINDS:
            return Decision(source="native_only", reason="Quota and connectivity checks always go to the vendor.")
        self.observe_model(dialect, model)
        task = self.task(tool, session, project) if kind in TASK_KINDS else None
        if task and task.decision is not None:
            pinned = task.decision
            # New rules take effect at the next prompt, never mid tool loop; a model the
            # user switched to is decided afresh (a pin must never move it up).
            fresh = kind == "main" and task.rules_version != self.version()
            if pinned.source == "escalation" or (task.pinned_for == model and not fresh):
                return self._finish(pinned, tool, dialect, model, needs, task, original_billing, kept=True)
            task.decision = None
        decision = self._rules_decision(tool, dialect, model, kind, project)
        return self._finish(decision, tool, dialect, model, needs, task, original_billing)

    def version(self) -> str:
        """A fingerprint of the current rules, to tell when they changed."""
        raw = json.dumps(self.rules(), sort_keys=True, default=str)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def _finish(
        self,
        decision: Decision,
        tool: str,
        dialect: str,
        model: str,
        needs: Needs,
        task: Task | None,
        original_billing: str,
        kept: bool = False,
    ) -> Decision:
        if decision.target and (why := compatible(decision.target, needs)):
            decision = decision.with_reason(f"Kept on {model}: {why}.", target=None, native=None)
        if task is not None:
            over = self._over_budget(task)
            if over and decision.route != "passthrough":
                if self.settings()["budget"]["on_exceed"] == "stop" and decision.target:
                    return Decision(source="budget", reason=f"Task budget reached ({over}).", stop=True)
                decision = Decision(source="budget", reason=f"Task budget reached ({over}); back on {model}.")
            if task.decision is None:
                task.decision, task.pinned_for, task.rules_version = decision, model, self.version()
            elif kept and decision is task.decision and decision.source not in ("escalation", "budget"):
                decision = decision.with_reason(f"{decision.reason} Kept for this task.")
        return decision

    def _rules_decision(self, tool: str, dialect: str, model: str, kind: str, project: str | None) -> Decision:
        data = self.rules()
        tool_rules = (data.get("tools") or {}).get(tool) or {}
        proj = (data.get("projects") or {}).get(project or "") or {}
        proj_tool = (proj.get("tools") or {}).get(tool) or {}
        where = f" in {project}" if proj_tool else ""
        for rules, scope in ((proj_tool, where), (tool_rules, "")):
            kinds = rules.get("kinds") or {}
            kind_key = (
                kind if kind in kinds else "background" if kind in BACKGROUND_KINDS and "background" in kinds else None
            )
            if kind_key:
                return self._to(
                    kinds[kind_key], dialect, model, "kind", kind_key, f"{_KIND_WORDS.get(kind, kind)}{scope}"
                )
            for pattern, to in (rules.get("models") or {}).items():
                if fnmatch.fnmatchcase(model.lower(), pattern.lower()):
                    return self._to(to, dialect, model, "model", pattern, f"{model} matches {pattern}{scope}")
        for name, scope in (
            (proj_tool.get("preset"), f"{project}'s preset for this tool"),
            (proj.get("preset"), f"{project}'s preset"),
            (tool_rules.get("preset"), "this tool's preset"),
            (data.get("preset"), "the default preset"),
        ):
            if name in PRESETS:
                target = PRESETS[name].get(kind)
                if not target:
                    return Decision(
                        source="preset",
                        match=name,
                        preset=name,
                        reason=f"{name.capitalize()} ({scope}) keeps {_KIND_WORDS.get(kind, kind)} on {model}.",
                    )
                return self._to(
                    target,
                    dialect,
                    model,
                    "preset",
                    name,
                    f"{name.capitalize()} ({scope}): {_KIND_WORDS.get(kind, kind)}",
                    preset=name,
                )
        return PASS

    def _to(
        self, to: str, dialect: str, model: str, source: str, match: str, why: str, preset: str | None = None
    ) -> Decision:
        if to == "passthrough" or to == "native":
            return Decision(source=source, match=match, preset=preset, reason=f"{why} → unchanged.")
        if to.startswith("native:"):
            native, blocked = self._native(to, dialect, model)
            if native is None:
                return Decision(
                    source=source, match=match, preset=preset, reason=f"{why}: kept on {model} ({blocked})."
                )
            return Decision(source=source, match=match, preset=preset, native=native, reason=f"{why} → {native}.")
        return Decision(source=source, match=match, preset=preset, target=to, reason=f"{why} → {to}.")

    # -- fallback, escalation, budget ------------------------------------------

    def _over_budget(self, task: Task) -> str | None:
        budget = self.settings()["budget"]
        limit_usd = budget.get("task_usd")
        if isinstance(limit_usd, (int, float)) and limit_usd > 0 and task.spent_usd >= limit_usd:
            return f"${task.spent_usd:.2f} of ${limit_usd:.2f}"
        attempts = budget.get("task_attempts")
        if isinstance(attempts, int) and attempts > 0 and task.attempts >= attempts:
            return f"{task.attempts} of {attempts} attempts"
        return None

    def fallbacks(self, decision: Decision, needs: Needs, task: Task | None, original_billing: str) -> list[str]:
        """Prompture models to try, in order, after *decision*'s target fails (the vendor path comes after them)."""
        if task is not None and self._over_budget(task):
            return []
        conf = self.settings()["fallback"]
        out: list[str] = []
        for model in conf.get("models") or []:
            if not isinstance(model, str) or model == decision.target or model in out:
                continue
            from .calls import billing_of

            if original_billing == "subscription" and billing_of(model) == "api" and not conf.get("allow_paid"):
                continue  # plan traffic onto a paid API only when allowed
            if compatible(model, needs) is None:
                out.append(model)
        room = MAX_ATTEMPTS_PER_REQUEST - 2  # the first try and the vendor's own path
        return out[: max(0, room)]

    def count_attempt(self, task: Task | None) -> None:
        if task is not None:
            with self._lock:
                task.attempts += 1

    def add_spend(self, task: Task | None, usd: float) -> None:
        if task is not None and usd:
            with self._lock:
                task.spent_usd += usd

    def observe(
        self, tool: str, dialect: str, body: dict[str, Any], task: Task | None, requested: str
    ) -> Decision | None:
        """Read a ``tool_result`` request's outcome; returns the escalation it triggers, if any."""
        if task is None:
            return None
        outcome = tool_outcome(dialect, body)
        if outcome is None:
            return None
        with self._lock:
            task.waiting = False
            if outcome.user_decision or not outcome.failed:
                task.failures, task.repeats, task.last_action = 0, 0, outcome.action
                return None
            task.failures += 1
            task.repeats = task.repeats + 1 if outcome.action and outcome.action == task.last_action else 1
            task.last_action = outcome.action
            conf = self.settings()["escalation"]
            if not conf.get("enabled"):
                return None
            what = f" ({outcome.name})" if outcome.name else ""
            if task.repeats >= int(conf.get("after_repeats") or 0) > 0 and task.repeats > 1:
                why = f"the same action failed {task.repeats} times in a row{what}"
            elif task.failures >= int(conf.get("after_failures") or 0) > 0:
                why = f"{task.failures} tool results failed in a row{what}"
            else:
                return None
        return self.escalate(tool, dialect, task, requested, why)

    def escalate(self, tool: str, dialect: str, task: Task, requested: str, why: str) -> Decision | None:
        """Move *task* to a stronger model; ``None`` when there is nowhere higher or the budget is spent."""
        if over := self._over_budget(task):
            with self._lock:
                task.escalations.append({"at": time.time(), "reason": why, "to": None, "blocked": over})
            return None
        current = task.decision or PASS
        conf = self.settings()["escalation"]
        if current.route != "passthrough":
            to = Decision(source="escalation", reason=f"Escalated to {requested}: {why}.")
        else:
            spec = str(conf.get("to") or "native")
            if spec == "native":
                return None
            if spec.startswith("native:"):
                want = spec.split(":", 1)[1]
                model = self.native_model(dialect, want) if want in TIERS else want
                if (
                    not model
                    or model == requested
                    or (want in TIERS and _rank(want) <= _rank(tier_of(dialect, requested)))
                ):
                    return None
                to = Decision(source="escalation", native=model, reason=f"Escalated to {model}: {why}.")
            else:
                to = Decision(source="escalation", target=spec, reason=f"Escalated to {spec}: {why}.")
        with self._lock:
            task.decision = to
            task.failures, task.repeats = 0, 0
            task.attempts += 1
            task.escalations.append({"at": time.time(), "reason": why, "to": to.rule()["target"] or requested})
        return to


_KIND_WORDS = {
    "main": "main turns",
    "tool_result": "tool follow-ups",
    "title": "titles",
    "probe": "quota checks",
    "compaction": "conversation summaries",
    "background": "background calls",
}
