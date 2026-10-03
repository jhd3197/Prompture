"""Multi-source research with citations: :class:`ResearchAgent`.

A run is a fixed pipeline rather than an open-ended tool loop, so it stays
predictable and inside its budget:

1. **Plan** — the model splits the question into sub-questions with search
   phrasings, relevant platforms (GitHub, Hacker News, arXiv, YouTube) and
   domain packs (falls back to a keyword plan when the call fails).
2. **Search** — every phrasing, platform search and pack run fans out in
   parallel threads.
3. **Rank + read** — hits are de-duplicated by normalized URL, fused across
   phrasings and spread across domains; the top pages per sub-question are
   opened with ``read_url`` (routed readers, ``web_fetch`` fallback), media
   pages are transcribed when transcription is available.
4. **Synthesize** — only opened pages reach the model, numbered for
   ``[n]`` citations; markers to anything else are stripped. The report
   lists opened sources, conflicting claims and coverage gaps.
5. **Budget** — fetch, search, token, cost and wall-clock limits are
   enforced across threads; when one hits, gathering stops and synthesis
   uses what was already read (or an extractive answer when the LLM budget
   is gone).

Example::

    from prompture.research import ResearchAgent, ResearchBudget

    agent = ResearchAgent("openai/gpt-4o-mini", depth="quick", budget=ResearchBudget(max_cost=0.05))
    report = agent.run("What changed in the EU AI Act timeline in 2025?")
    print(report.to_markdown())
"""

from __future__ import annotations

import asyncio
import concurrent.futures as cf
import contextlib
import os
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any
from urllib.parse import urlsplit

from ..exceptions import ConfigurationError
from .budget import BudgetTracker, DepthPreset, ResearchBudget, get_depth_preset
from .events import ResearchEvent
from .planner import ResearchPlan, heuristic_plan, plan_research
from .ranking import Candidate, CandidatePool, select_diverse
from .report import ReportSource, ResearchReport, SubQuestionResult
from .synthesis import (
    OpenedSource,
    build_source_bundle,
    build_synthesis_prompt,
    extractive_answer,
    fit_chars_per_source,
    parse_citations,
    parse_synthesis,
    restrict_citations,
)
from .tools import ResearchTools, ResearchToolUnavailable

#: Cheap default model per configured provider, in preference order.
DEFAULT_MODEL_CANDIDATES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("openai_api_key",), "openai/gpt-4o-mini"),
    (("claude_api_key",), "claude/claude-haiku-4-5"),
    (("google_api_key",), "google/gemini-2.5-flash"),
    (("groq_api_key",), "groq/llama-3.3-70b-versatile"),
    (("deepseek_api_key",), "deepseek/deepseek-chat"),
    (("grok_api_key", "xai_api_key"), "grok/grok-3-mini"),
    (("mistral_api_key",), "mistral/mistral-small-latest"),
    (("openrouter_api_key",), "openrouter/openai/gpt-4o-mini"),
)

_MEDIA_KINDS = {"video", "audio", "podcast", "media"}
_MEDIA_EXTS = (".mp3", ".m4a", ".mp4", ".wav", ".ogg", ".opus", ".flac", ".aac", ".webm", ".mov", ".mkv")
_MIN_CONTENT_CHARS = 40
_THIN_MEDIA_CHARS = 600


def default_research_model() -> str | None:
    """Pick a model for research when none is given.

    Order: ``PROMPTURE_RESEARCH_MODEL``, ``PROMPTURE_DEFAULT_MODEL``, then the
    cheap model of the first configured provider in
    :data:`DEFAULT_MODEL_CANDIDATES`. ``None`` when nothing is configured.
    """
    from ..infra.credentials import get_config_value

    for var in ("PROMPTURE_RESEARCH_MODEL", "PROMPTURE_DEFAULT_MODEL"):
        # Env first, then the credential store, where `prompture setup` saves the default model.
        value = (get_config_value(var) or "").strip()
        if value:
            return value
    try:
        from ..infra.settings import settings
    except Exception:  # pragma: no cover - settings import failure is environmental
        settings = None
    for attrs, model in DEFAULT_MODEL_CANDIDATES:
        for attr in attrs:
            if os.environ.get(attr.upper(), "").strip() or (settings is not None and getattr(settings, attr, None)):
                return model
    return None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _short_error(exc: BaseException) -> str:
    try:
        from ..security.redaction import scrub_secrets

        text = scrub_secrets(str(exc))
    except Exception:  # pragma: no cover - redaction is best effort
        text = str(exc)
    text = " ".join(text.split())
    if len(text) > 200:
        text = text[:197] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _call_with_deadline(fn: Callable[[], Any], timeout: float | None) -> Any:
    """Run *fn* in a worker thread; raise :class:`TimeoutError` after *timeout* seconds."""
    if timeout is None:
        return fn()
    if timeout <= 0:
        raise TimeoutError("no time left")
    ex = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-llm")
    try:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=timeout)
        except cf.TimeoutError:
            raise TimeoutError(f"timed out after {timeout:.1f}s") from None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _usage_from_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    meta = meta or {}
    return {
        "prompt_tokens": meta.get("prompt_tokens", 0) or 0,
        "completion_tokens": meta.get("completion_tokens", 0) or 0,
        "total_tokens": meta.get("total_tokens", 0) or 0,
        "cost": meta.get("cost", 0.0) or 0.0,
    }


def _is_media_url(url: str) -> bool:
    try:
        path = urlsplit(url).path.lower()
    except ValueError:
        return False
    return path.endswith(_MEDIA_EXTS)


@dataclass
class _Fetched:
    candidate: Candidate
    ok: bool
    title: str = ""
    content: str = ""
    reader: str | None = None
    kind: str | None = None
    route: dict[str, Any] = field(default_factory=dict)
    transcribed: bool = False
    error: str | None = None
    skipped: bool = False


class ResearchAgent:
    """Plan, search in parallel, read the best pages and write a cited report.

    Args:
        model: ``"provider/model"`` used for planning, pack runs and synthesis.
            ``None`` picks :func:`default_research_model`.
        depth: ``"quick"``, ``"standard"`` or ``"deep"`` (see
            :data:`~prompture.research.DEPTH_PRESETS`).
        budget: Limits; unset fields come from the depth preset.
        tools: Gathering functions (search/read/transcribe/packs); defaults
            bind lazily to :mod:`prompture.tools.web` and friends.
        on_event: Called with each :class:`ResearchEvent` (from worker threads,
            serialized). Exceptions it raises are ignored.
        driver: An already-built driver, instead of resolving *model*.
        compress_sources: Send the source bundle to synthesis as a TOON table.
        transcribe_media: Transcribe video/audio sources whose reader returned
            little text, when transcription is available.
        max_per_domain: Pages opened per domain before others get a turn.
        max_workers: Thread pool size for the fan-out.
        options: Extra driver options for every LLM call.
        today: Date shown to the planner (defaults to today).
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        depth: str = "standard",
        budget: ResearchBudget | None = None,
        tools: ResearchTools | None = None,
        on_event: Callable[[ResearchEvent], Any] | None = None,
        driver: Any = None,
        compress_sources: bool = False,
        transcribe_media: bool = True,
        max_per_domain: int = 2,
        max_workers: int = 8,
        options: dict[str, Any] | None = None,
        today: date | None = None,
    ) -> None:
        self.preset: DepthPreset = get_depth_preset(depth)
        self.depth = depth.lower()
        resolved_model = model or (getattr(driver, "model", None) if driver is not None else None)
        if not resolved_model and driver is None:
            resolved_model = default_research_model()
            if not resolved_model:
                raise ConfigurationError(
                    "ResearchAgent needs a model: pass model='provider/model', set PROMPTURE_RESEARCH_MODEL, "
                    "or configure a provider API key."
                )
        self.model: str = resolved_model or ""
        self.budget = budget or ResearchBudget()
        self.tools = tools or ResearchTools()
        self.on_event = on_event
        self.compress_sources = compress_sources
        self.transcribe_media = transcribe_media
        self.max_per_domain = max(1, int(max_per_domain))
        self.max_workers = max(1, int(max_workers))
        self.options = dict(options or {})
        self.today = today
        self._driver = driver
        self._driver_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def run(self, question: str) -> ResearchReport:
        """Research *question* and return a :class:`ResearchReport`."""
        question = " ".join((question or "").split())
        if not question:
            raise ValueError("question must be a non-empty string")
        return _ResearchRun(self, question).execute()

    async def arun(self, question: str) -> ResearchReport:
        """Async wrapper: runs :meth:`run` in a worker thread."""
        return await asyncio.to_thread(self.run, question)

    def run_live(self, question: str) -> Iterator[ResearchEvent]:
        """Yield :class:`ResearchEvent` objects as the run progresses.

        The final event is ``done`` with ``data["report"]``. Errors raised by
        the run are re-raised from the iterator.
        """
        q: queue.Queue[Any] = queue.Queue()
        sentinel = object()
        user_cb = self.on_event

        def _forward(ev: ResearchEvent) -> None:
            q.put(ev)
            if user_cb is not None:
                user_cb(ev)

        def _worker() -> None:
            clone = self._clone(on_event=_forward)
            try:
                clone.run(question)
            except BaseException as exc:
                q.put(exc)
            finally:
                q.put(sentinel)

        threading.Thread(target=_worker, name="research-live", daemon=True).start()
        while True:
            item = q.get()
            if item is sentinel:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------

    def _clone(self, **overrides: Any) -> ResearchAgent:
        clone = object.__new__(ResearchAgent)
        clone.__dict__.update(self.__dict__)
        clone._driver_lock = threading.Lock()
        clone.__dict__.update(overrides)
        return clone

    def _get_driver(self) -> Any:
        with self._driver_lock:
            if self._driver is None:
                from ..drivers import get_driver_for_model

                self._driver = get_driver_for_model(self.model)
            return self._driver

    def _ask_json(self, prompt: str, schema: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        from ..extraction.core import ask_for_json

        result = ask_for_json(
            self._get_driver(),
            prompt,
            schema,
            ai_cleanup=False,
            model_name=self.model,
            options=dict(self.options),
            cache=False,
        )
        return result.get("json_object"), dict(result.get("usage") or {})

    def _generate(self, prompt: str, *, max_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
        options = dict(self.options)
        if max_tokens and "max_tokens" not in options:
            options["max_tokens"] = max_tokens
        resp = self._get_driver().generate(prompt, options)
        return str(_get(resp, "text", "") or ""), _usage_from_meta(_get(resp, "meta", {}))

    def _run_pack(self, pack: str, question: str) -> tuple[str, dict[str, Any]]:
        if self.tools.pack_runner is not None:
            return self.tools.pack_runner(pack, question)
        defs = self.tools.do_pack_tools(pack)
        if not defs:
            raise ResearchToolUnavailable(f"pack {pack!r} has no live tools")
        from ..agents.agent import Agent
        from ..agents.tools_schema import ToolRegistry

        registry = ToolRegistry()
        for td in defs:
            registry.add(td)
        agent = Agent(
            self.model,
            driver=self._get_driver() if self._driver is not None else None,
            tools=registry,
            system_prompt=(
                "Use the available tools to gather concrete, current facts (numbers, dates, names) relevant "
                "to the question. Report the facts plainly and say which tool returned each one. Do not "
                "speculate beyond the tool output."
            ),
            max_iterations=6,
        )
        result = agent.run(question)
        usage = dict(result.run_usage or result.usage or {})
        return result.output_text or "", usage


class _ResearchRun:
    """State for one :meth:`ResearchAgent.run` call."""

    def __init__(self, agent: ResearchAgent, question: str) -> None:
        self.agent = agent
        self.question = question
        self.preset = agent.preset
        self.tracker = BudgetTracker(agent.budget.resolve(agent.preset))
        self.pool = CandidatePool()
        self.routes: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.pack_sources: list[tuple[str, str]] = []  # (pack, text)
        self.fetched: dict[str, _Fetched] = {}
        self._lock = threading.Lock()
        self._emit_lock = threading.Lock()
        # Packs run in parallel, but under a token/cost budget their LLM calls
        # take turns: checking the budget and recording usage happen under this
        # lock, so concurrent calls can't all pass the check and overshoot.
        self._llm_budget_lock = threading.Lock()
        self.search_ok = 0
        self.search_failed = 0

    # -- events -------------------------------------------------------------

    def emit(self, event_type: str, message: str, **data: Any) -> None:
        cb = self.agent.on_event
        if cb is None:
            return
        ev = ResearchEvent(event_type, message, data, self.tracker.elapsed())  # type: ignore[arg-type]
        with self._emit_lock, contextlib.suppress(Exception):
            cb(ev)

    def warn(self, message: str) -> None:
        with self._lock:
            if message not in self.warnings:
                self.warnings.append(message)
        self.emit("warning", message)

    # -- pipeline -----------------------------------------------------------

    def execute(self) -> ResearchReport:
        plan = self._plan()
        self._search(plan)
        self._fetch(plan)
        opened = self._number_sources(plan)
        answer, conflicts, gaps, synthesis = self._synthesize(plan, opened)
        report = self._build_report(plan, opened, answer, conflicts, gaps, synthesis)
        self.emit(
            "done",
            f"Done: {len(report.opened_sources)} sources opened, {len(report.cited_sources)} cited",
            report=report,
            budget_used=report.budget_used,
        )
        return report

    # 1. plan ---------------------------------------------------------------

    def _plan(self) -> ResearchPlan:
        agent = self.agent
        if not self.tracker.llm_budget_left():
            plan = heuristic_plan(self.question, self.preset, error="LLM budget exhausted before planning")
        else:

            def _ask(prompt: str, schema: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
                result = _call_with_deadline(lambda: agent._ask_json(prompt, schema), self.tracker.gather_remaining())
                return result  # type: ignore[no-any-return]

            plan = plan_research(self.question, self.preset, _ask, today=agent.today)
        if plan.usage:
            self.tracker.record_usage(plan.usage)
        if plan.error:
            self.warn(f"Used keyword plan: {plan.error}")
        self.emit(
            "plan",
            f"Planned {len(plan.sub_questions)} sub-question(s)",
            sub_questions=[sq.to_dict() for sq in plan.sub_questions],
            packs=list(plan.packs),
            source=plan.source,
        )
        return plan

    # 2. search ---------------------------------------------------------------

    def _search(self, plan: ResearchPlan) -> None:
        tasks: list[tuple[str, int, str, str]] = []  # (kind, sub_idx, query, platform/pack)
        for i, sq in enumerate(plan.sub_questions):
            for q in sq.queries:
                tasks.append(("web", i, q, ""))
            if self.agent.tools.enable_platforms:
                for p in sq.platforms:
                    tasks.append(("platform", i, sq.queries[0] if sq.queries else sq.question, p))
        if self.agent.tools.enable_packs:
            for pack in plan.packs:
                tasks.append(("pack", -1, self.question, pack))
        self._run_parallel(tasks, self._search_one, self.tracker.gather_remaining)
        if self.search_ok == 0 and self.search_failed:
            self.warn("Every search failed; check `prompture doctor` for web tool health")

    def _search_one(self, task: tuple[str, int, str, str]) -> None:
        kind, idx, query, target = task
        if not self.tracker.try_search():
            return
        if kind == "pack":
            self._pack_one(target)
            return
        tools = self.agent.tools
        try:
            if kind == "web":
                resp = tools.do_search(query, max_results=self.preset.results_per_query)
            else:
                resp = tools.do_search_platform(target, query, max_results=min(self.preset.results_per_query, 6))
        except Exception as exc:
            with self._lock:
                self.search_failed += 1
                self.routes.append(
                    {"op": kind, "target": query, "platform": target or None, "ok": False, "error": _short_error(exc)}
                )
            self.emit("search", f"Search failed: {query}", query=query, platform=target or None, ok=False)
            return
        results = list(_get(resp, "results", None) or (resp if isinstance(resp, list) else []))
        served_by = _get(resp, "served_by", None) or (target if kind == "platform" else None)
        origin = "web" if kind == "web" else target
        with self._lock:
            self.search_ok += 1
            for rank, r in enumerate(results):
                self.pool.add(r, rank=rank, sub_question=idx, query=query, origin=origin)
            self.routes.append(
                {
                    "op": kind,
                    "target": query,
                    "platform": target or None,
                    "ok": True,
                    "results": len(results),
                    "served_by": served_by,
                    "route": _get(resp, "route", None) or {},
                }
            )
        label = f"{target}: {query}" if kind == "platform" else query
        self.emit(
            "search",
            f"{len(results)} result(s) for {label}",
            query=query,
            platform=target or None,
            results=len(results),
            served_by=served_by,
            ok=True,
        )

    def _pack_one(self, pack: str) -> None:
        budgeted = self.tracker.budget.max_tokens is not None or self.tracker.budget.max_cost is not None
        guard = self._llm_budget_lock if budgeted else contextlib.nullcontext()
        with guard:
            if not self.tracker.llm_budget_left():
                return
            try:
                text, usage = _call_with_deadline(
                    lambda: self.agent._run_pack(pack, self.question), self.tracker.gather_remaining()
                )
            except Exception as exc:
                with self._lock:
                    self.routes.append({"op": "pack", "target": pack, "ok": False, "error": _short_error(exc)})
                self.emit("search", f"Pack {pack} unavailable", pack=pack, ok=False)
                return
            self.tracker.record_usage(usage)
        text = (text or "").strip()
        with self._lock:
            self.routes.append({"op": "pack", "target": pack, "ok": bool(text), "served_by": f"pack:{pack}"})
            if len(text) >= _MIN_CONTENT_CHARS:
                self.pack_sources.append((pack, text))
        self.emit("search", f"Pack {pack} returned {len(text)} chars", pack=pack, ok=bool(text))

    # 3. fetch ----------------------------------------------------------------

    def _fetch(self, plan: ResearchPlan) -> None:
        per_q = self.preset.fetches_per_question
        domain_counts: dict[str, int] = {}
        successes = [0] * len(plan.sub_questions)
        for _round in range(3):
            if not self.tracker.gather_time_left():
                break
            batch: list[Candidate] = []
            batch_keys: set[str] = set()
            for i in range(len(plan.sub_questions)):
                need = per_q - successes[i]
                if need <= 0:
                    continue
                exclude = set(self.fetched) | batch_keys
                picks = select_diverse(
                    self.pool.for_sub_question(i),
                    need,
                    max_per_domain=self.agent.max_per_domain,
                    exclude=exclude,
                    domain_counts=domain_counts,
                )
                for c in picks:
                    batch.append(c)
                    batch_keys.add(c.key)
            if not batch:
                break
            self._run_parallel(batch, self._fetch_one, self.tracker.gather_remaining)
            new_success = 0
            for c in batch:
                f = self.fetched.get(c.key)
                if f is not None and f.ok:
                    new_success += 1
                    for i in c.sub_questions:
                        if i < len(successes):
                            successes[i] += 1
            if self.tracker.limits_hit and any(lim in self.tracker.limits_hit for lim in ("max_fetches", "timeout")):
                break
            if new_success == len(batch):
                break  # everything worked: no backfill needed

    def _fetch_one(self, cand: Candidate) -> None:
        if not self.tracker.try_fetch():
            with self._lock:
                self.fetched.setdefault(cand.key, _Fetched(cand, ok=False, skipped=True, error="budget"))
            return
        result = self._open(cand)
        with self._lock:
            self.fetched[cand.key] = result
            self.routes.append(
                {
                    "op": "fetch",
                    "target": cand.url,
                    "ok": result.ok,
                    "served_by": result.reader,
                    "route": result.route,
                    **({"error": result.error} if result.error else {}),
                }
            )
        if result.ok:
            msg = f"Read {cand.url} ({len(result.content)} chars{', transcript' if result.transcribed else ''})"
        else:
            msg = f"Could not read {cand.url}: {result.error}"
        self.emit(
            "fetch",
            msg,
            url=cand.url,
            ok=result.ok,
            reader=result.reader,
            chars=len(result.content),
            transcribed=result.transcribed,
        )

    def _open(self, cand: Candidate) -> _Fetched:
        tools = self.agent.tools
        title, content, reader, kind, route, error = cand.title, "", None, None, {}, None
        try:
            res = tools.do_read(cand.url)
            content = str(_get(res, "content", "") or "")
            title = str(_get(res, "title", "") or "") or cand.title
            reader = _get(res, "reader", None) or _get(res, "served_by", None)
            kind = _get(res, "kind", None)
            route = _get(res, "route", None) or {}
        except Exception as exc:
            error = _short_error(exc)
            # The reader failed: try the plain fetcher once — unless the read
            # already *was* the fetcher, or only a custom reader was supplied.
            if (tools.read is None) == (tools.fetch is None):
                try:
                    res = tools.do_fetch(cand.url, max_chars=self.preset.source_chars * 2)
                    content = str(_get(res, "content", "") or "")
                    title = str(_get(res, "title", "") or "") or cand.title
                    reader = _get(res, "served_by", None) or "web_fetch"
                    route = _get(res, "route", None) or {}
                    error = None
                except Exception as exc2:
                    error = _short_error(exc2)

        transcribed = False
        is_media = (kind or "").lower() in _MEDIA_KINDS or _is_media_url(cand.url)
        if (
            self.agent.transcribe_media
            and is_media
            and len(content.strip()) < _THIN_MEDIA_CHARS
            and tools.can_transcribe()
        ):
            try:
                tr = tools.do_transcribe(cand.url)
                to_md = _get(tr, "to_markdown", None)
                text = to_md() if callable(to_md) else str(_get(tr, "text", "") or "")
                if text and len(text.strip()) > len(content.strip()):
                    content = text
                    transcribed = True
                    reader = f"{reader}+transcript" if reader else "transcript"
                    error = None
            except Exception as exc:
                self.warn(f"Transcription failed for {cand.url}: {_short_error(exc)}")

        if len(content.strip()) < _MIN_CONTENT_CHARS:
            return _Fetched(
                cand,
                ok=False,
                title=title,
                reader=reader,
                kind=kind,
                route=route,
                error=error or "no readable content",
            )
        return _Fetched(
            cand,
            ok=True,
            title=title,
            content=content,
            reader=reader,
            kind=kind,
            route=route,
            transcribed=transcribed,
        )

    # -- parallel helper --------------------------------------------------------

    def _run_parallel(self, items: list[Any], fn: Callable[[Any], None], time_left: Callable[[], float | None]) -> None:
        if not items:
            return
        ex = cf.ThreadPoolExecutor(max_workers=min(self.agent.max_workers, len(items)), thread_name_prefix="research")
        try:
            futures = [ex.submit(fn, item) for item in items]
            timeout = time_left()
            done, pending = cf.wait(futures, timeout=timeout)
            if pending:
                self.tracker.hit("timeout")
                self.warn(f"Wall-clock limit reached; {len(pending)} task(s) abandoned")
            for fut in done:
                exc = fut.exception()
                if exc is not None:
                    self.warn(f"Research task failed: {_short_error(exc)}")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    # 4. number + synthesize ---------------------------------------------------

    def _number_sources(self, plan: ResearchPlan) -> list[OpenedSource]:
        with self._lock:
            ok = [f for f in self.fetched.values() if f.ok]
            packs = list(self.pack_sources)
        ok.sort(key=lambda f: (min(f.candidate.sub_questions or {0}), -f.candidate.score, f.candidate.key))
        opened: list[OpenedSource] = []
        for f in ok:
            opened.append(
                OpenedSource(
                    n=len(opened) + 1,
                    url=f.candidate.url,
                    title=f.title or f.candidate.title,
                    content=f.content,
                    reader=f.reader,
                    kind=f.kind,
                    sub_questions=sorted(f.candidate.sub_questions),
                )
            )
        for pack, text in packs:
            opened.append(
                OpenedSource(
                    n=len(opened) + 1,
                    url=f"pack:{pack}",
                    title=f"{pack} data tools",
                    content=text,
                    reader=f"pack:{pack}",
                    kind="data",
                    sub_questions=list(range(len(plan.sub_questions))),
                )
            )
        return opened

    def _synthesize(self, plan: ResearchPlan, opened: list[OpenedSource]) -> tuple[str, list[str], list[str], str]:
        if not opened:
            self.emit("synthesize", "Nothing was opened; skipping synthesis", reason="no sources", sources=0)
            return extractive_answer([], "nothing could be read"), [], [], "none"

        def _fallback(reason: str) -> tuple[str, list[str], list[str], str]:
            self.emit("synthesize", f"Extractive answer ({reason})", reason=reason, sources=len(opened))
            return extractive_answer(opened, reason), [], [], "extractive"

        if not self.tracker.llm_budget_left():
            return _fallback("LLM budget exhausted")
        chars = fit_chars_per_source(
            default_chars=self.preset.source_chars,
            n_sources=len(opened),
            tokens_remaining=self.tracker.tokens_remaining(),
            answer_tokens=self.preset.answer_tokens,
        )
        if chars < 300:
            self.tracker.hit("max_tokens")
            return _fallback("token budget too small for synthesis")
        bundle = build_source_bundle(opened, chars_per_source=chars, compress=self.agent.compress_sources)
        prompt = build_synthesis_prompt(self.question, [sq.question for sq in plan.sub_questions], bundle)

        max_cost = self.tracker.budget.max_cost
        if max_cost is not None and self.agent.model:
            try:
                from ..infra.budget import estimate_cost

                est = estimate_cost(self.agent.model, len(prompt) // 4, self.preset.answer_tokens)
            except Exception:
                est = 0.0
            if est and self.tracker.cost + est > max_cost:
                self.tracker.hit("max_cost")
                return _fallback(f"estimated synthesis cost ${est:.4f} exceeds the remaining cost budget")

        self.emit(
            "synthesize",
            f"Synthesizing from {len(opened)} source(s)",
            sources=len(opened),
            bundle=bundle.to_dict(),
        )
        try:
            text, usage = _call_with_deadline(
                lambda: self.agent._generate(prompt, max_tokens=self.preset.answer_tokens), self.tracker.remaining()
            )
        except TimeoutError:
            self.tracker.hit("timeout")
            return _fallback("wall-clock limit reached during synthesis")
        except Exception as exc:
            self.warn(f"Synthesis failed: {_short_error(exc)}")
            return _fallback("synthesis call failed")
        self.tracker.record_usage(usage)
        self.tracker.llm_budget_left()  # records max_cost / max_tokens if synthesis crossed them
        if not text.strip():
            return _fallback("model returned an empty answer")

        answer, conflicts, gaps = parse_synthesis(text)
        allowed = {str(s.n) for s in opened}
        answer, dropped = restrict_citations(answer, allowed)
        conflicts = [restrict_citations(c, allowed)[0].strip() for c in conflicts]
        gaps = [restrict_citations(g, allowed)[0].strip() for g in gaps]
        if dropped:
            self.warn(f"Removed citations to sources that were not opened: {', '.join(sorted(set(dropped)))}")
        return answer, [c for c in conflicts if c], [g for g in gaps if g], "llm"

    # 5. report -----------------------------------------------------------------

    def _build_report(
        self,
        plan: ResearchPlan,
        opened: list[OpenedSource],
        answer: str,
        conflicts: list[str],
        gaps: list[str],
        synthesis: str,
    ) -> ResearchReport:
        cited_ids = parse_citations(answer, [s.to_citation_source() for s in opened]).cited_source_ids
        for c in conflicts:
            cited_ids |= parse_citations(c, [s.to_citation_source() for s in opened]).cited_source_ids

        with self._lock:
            fetched = dict(self.fetched)
        by_url = {s.url: s for s in opened}
        sources: list[ReportSource] = []
        for s in opened:
            f = next((x for x in fetched.values() if x.ok and x.candidate.url == s.url), None)
            sources.append(
                ReportSource(
                    n=s.n,
                    url=s.url,
                    title=s.title,
                    opened=True,
                    reader=s.reader,
                    kind=s.kind,
                    cited=str(s.n) in cited_ids,
                    sub_questions=list(s.sub_questions),
                    origins=list(f.candidate.origins) if f else ["pack"],
                    transcribed=bool(f and f.transcribed),
                    chars=len(s.content),
                    snippet=(f.candidate.snippet if f else s.content[:200]),
                    route=dict(f.route) if f else {},
                )
            )
        for cand in self.pool.all():
            if cand.url in by_url:
                continue
            f = fetched.get(cand.key)
            sources.append(
                ReportSource(
                    n=None,
                    url=cand.url,
                    title=cand.title,
                    opened=False,
                    reader=f.reader if f else None,
                    sub_questions=sorted(cand.sub_questions),
                    origins=list(cand.origins),
                    snippet=cand.snippet,
                    error=(f.error if f else None),
                )
            )

        sub_results: list[SubQuestionResult] = []
        for i, sq in enumerate(plan.sub_questions):
            sub_results.append(
                SubQuestionResult(
                    question=sq.question,
                    queries=list(sq.queries),
                    platforms=list(sq.platforms),
                    sources=[s.n for s in opened if i in s.sub_questions],
                    candidates=len(self.pool.for_sub_question(i)),
                )
            )

        all_gaps = list(gaps)
        for sr in sub_results:
            if not sr.answered:
                all_gaps.append(f"No source could be opened for: {sr.question}")
        limits = self.tracker.limits_hit
        if limits:
            unread = sum(1 for s in sources if not s.opened)
            all_gaps.append(f"Research stopped early ({', '.join(limits)}); {unread} found page(s) were not read.")
        if self.search_ok == 0 and not self.pack_sources:
            all_gaps.append("No search backend returned results.")
        all_gaps = list(dict.fromkeys(g for g in all_gaps if g))

        budget_used = self.tracker.snapshot()
        usage = self.tracker.usage()
        return ResearchReport(
            question=self.question,
            answer=answer,
            sources=sources,
            sub_questions=sub_results,
            gaps=all_gaps,
            conflicts=list(dict.fromkeys(conflicts)),
            budget_used=budget_used,
            cost=usage["cost"],
            usage=usage,
            routes=list(self.routes),
            model=self.agent.model,
            depth=self.agent.depth,
            plan_source=plan.source,
            synthesis=synthesis,
            warnings=list(self.warnings),
            elapsed_s=budget_used["elapsed_s"],
        )


def research_tool(
    model: str | None = None,
    *,
    depth: str = "quick",
    name: str = "research",
    **agent_kwargs: Any,
) -> Any:
    """Expose research as a tool other agents can call.

    The tool takes ``question`` (and optionally ``depth``) and returns the
    report as Markdown. It never raises: configuration or runtime errors
    come back as a short ``Research failed: ...`` message the calling model
    can read. The :class:`ResearchAgent` is built per call, so a missing
    model only surfaces when the tool is used.

    Args:
        model: Model for the research run (``None``: :func:`default_research_model`).
        depth: Default depth when the caller doesn't pass one.
        name: Tool name shown to the model.
        **agent_kwargs: Forwarded to :class:`ResearchAgent` (``budget``, ``tools``, ...).

    Returns:
        A :class:`~prompture.agents.tools_schema.ToolDefinition`.
    """
    from ..agents.tools_schema import ToolDefinition

    default_depth = depth

    def research(question: str, depth: str | None = None) -> str:
        try:
            agent = ResearchAgent(model, depth=(depth or default_depth), **agent_kwargs)
            return agent.run(question).to_markdown()
        except Exception as exc:
            return f"Research failed: {_short_error(exc)}"

    return ToolDefinition(
        name=name,
        description=(
            "Research a question on the web: plans sub-questions, searches several sources in parallel, "
            "reads the best pages and returns a Markdown report with numbered citations to the pages it "
            "actually opened, plus conflicting claims and coverage gaps. Slower than a single search; use "
            "for questions that need several sources."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to research."},
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "deep"],
                    "description": f"How thorough to be (default {default_depth}).",
                },
            },
            "required": ["question"],
        },
        function=research,
        metadata={"category": "research", "is_write": False},
    )


__all__ = ["DEFAULT_MODEL_CANDIDATES", "ResearchAgent", "default_research_model", "research_tool"]
