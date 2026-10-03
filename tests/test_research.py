"""Tests for the research agent (no network, no real LLM)."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import pytest
from click.testing import CliRunner

from prompture.drivers.base import Driver
from prompture.exceptions import ConfigurationError
from prompture.research import (
    CandidatePool,
    ResearchAgent,
    ResearchBudget,
    ResearchReport,
    ResearchTools,
    default_research_model,
    get_depth_preset,
    heuristic_plan,
    normalize_url,
    plan_research,
    research_tool,
    select_diverse,
    strip_tracking,
)
from prompture.research.planner import normalize_plan, suggest_packs, suggest_platforms
from prompture.research.synthesis import (
    OpenedSource,
    build_source_bundle,
    extractive_answer,
    fit_chars_per_source,
    parse_synthesis,
    restrict_citations,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

META = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.001, "raw_response": {}}

DEFAULT_PLAN = {
    "sub_questions": [
        {"question": "What is Widget?", "queries": ["widget definition", "widget overview"], "platforms": ["github"]},
        {"question": "Why use Widget?", "queries": ["widget benefits"], "platforms": []},
    ],
    "packs": [],
}

DEFAULT_SYNTHESIS = (
    "## Answer\n"
    "Widget is a toolkit [1]. It is fast [2, 7]. Unknown claim [9].\n\n"
    "## Conflicting claims\n"
    "- [1] says it is free while [3] says it is paid [8].\n\n"
    "## Coverage gaps\n"
    "- Pricing history is not covered.\n"
)


class FakeDriver(Driver):
    """Answers the planning prompt with a JSON plan and everything else with *synthesis*."""

    def __init__(self, plan: Any = None, synthesis: str = DEFAULT_SYNTHESIS, *, fail_plan: bool = False):
        self.model = "fake/model"
        self.plan = DEFAULT_PLAN if plan is None else plan
        self.synthesis = synthesis
        self.fail_plan = fail_plan
        self.prompts: list[str] = []
        self.options: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self.prompts.append(prompt)
            self.options.append(dict(options))
        if "You plan web research" in prompt:
            if self.fail_plan:
                raise RuntimeError("planner exploded")
            text = self.plan if isinstance(self.plan, str) else json.dumps(self.plan)
            return {"text": text, "meta": dict(META)}
        if isinstance(self.synthesis, Exception):
            raise self.synthesis
        return {"text": self.synthesis, "meta": dict(META)}

    @property
    def synthesis_prompts(self) -> list[str]:
        return [p for p in self.prompts if "You plan web research" not in p]


class FakeToon:
    """Stands in for python-toon (an optional extra) with a recognisable encoding."""

    @staticmethod
    def encode(rows: list[dict[str, Any]]) -> str:
        header = ",".join(rows[0])
        body = "\n".join(",".join(str(v) for v in r.values()) for r in rows)
        return f"TOON[{len(rows)}]{{{header}}}:\n{body}"


class Hit:
    def __init__(self, url: str, title: str, snippet: str = "", score: float | None = None):
        self.url = url
        self.title = title
        self.snippet = snippet or f"snippet for {title}"
        self.score = score
        self.extra: dict[str, Any] = {}


class Resp:
    def __init__(self, query: str, results: list[Hit]):
        self.query = query
        self.results = results
        self.served_by = "fake_search"
        self.route = {"served_by": "fake_search", "fallback": False, "attempts": []}


class Page:
    def __init__(self, url: str, content: str | None = None, kind: str = "html", reader: str = "fake_reader"):
        self.url = url
        self.title = f"Title of {url}"
        self.content = content if content is not None else f"Readable content for {url}. " * 5
        self.reader = reader
        self.kind = kind
        self.meta: dict[str, Any] = {}
        self.route = {"served_by": reader}


class Recorder:
    """Search/read fakes that log calls and track concurrency."""

    def __init__(self, *, delay: float = 0.0, per_query: int = 3):
        self.delay = delay
        self.per_query = per_query
        self.searches: list[str] = []
        self.platform_calls: list[tuple[str, str]] = []
        self.reads: list[str] = []
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def _enter(self) -> None:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

    def _exit(self) -> None:
        with self._lock:
            self.active -= 1

    def search(self, query: str, **kw: Any) -> Resp:
        self._enter()
        try:
            if self.delay:
                time.sleep(self.delay)
            with self._lock:
                self.searches.append(query)
            slug = query.replace(" ", "-")
            hits = [
                Hit(f"https://site{i}.example/{slug}?utm_source=feed", f"{query} #{i}") for i in range(self.per_query)
            ]
            return Resp(query, hits)
        finally:
            self._exit()

    def search_platform(self, platform: str, query: str, **kw: Any) -> list[Hit]:
        with self._lock:
            self.platform_calls.append((platform, query))
        return [Hit(f"https://github.com/acme/{query.replace(' ', '-')}", f"acme/{query}")]

    def read(self, url: str) -> Page:
        with self._lock:
            self.reads.append(url)
        return Page(url)


def make_tools(rec: Recorder, **kw: Any) -> ResearchTools:
    kw.setdefault("enable_transcription", False)
    kw.setdefault("enable_packs", False)
    return ResearchTools(search=rec.search, search_platform=rec.search_platform, read=rec.read, **kw)


def make_agent(rec: Recorder | None = None, driver: FakeDriver | None = None, **kw: Any) -> ResearchAgent:
    rec = rec or Recorder()
    tools = kw.pop("tools", None) or make_tools(rec)
    return ResearchAgent(driver=driver or FakeDriver(), depth=kw.pop("depth", "quick"), tools=tools, **kw)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class TestNormalizeUrl:
    def test_drops_tracking_fragment_www_and_trailing_slash(self):
        a = normalize_url("http://www.Example.com/Path/?utm_source=x&b=2&a=1&fbclid=zz#section")
        assert a == "https://example.com/Path?a=1&b=2"

    def test_equivalent_forms_collide(self):
        assert normalize_url("https://example.com/a/") == normalize_url("https://EXAMPLE.com:443/a?gclid=1")

    def test_youtube_forms(self):
        expected = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        assert normalize_url("https://youtu.be/dQw4w9WgXcQ?si=abc") == expected
        assert normalize_url("https://www.youtube.com/shorts/dQw4w9WgXcQ") == expected
        assert normalize_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ&feature=share&t=10") == expected

    def test_non_http_untouched(self):
        assert normalize_url("pack:finance") == "pack:finance"

    def test_strip_tracking_keeps_scheme_and_host(self):
        assert strip_tracking("http://www.a.com/x?utm_medium=y&q=1#f") == "http://www.a.com/x?q=1"


class TestCandidatePool:
    def test_dedupes_and_fuses(self):
        pool = CandidatePool()
        pool.add(Hit("https://a.com/x?utm_source=1", "A"), rank=0, sub_question=0, query="q1")
        pool.add(
            Hit("https://www.a.com/x/", "A again", snippet="a much longer snippet here"),
            rank=2,
            sub_question=1,
            query="q2",
        )
        pool.add(Hit("https://b.com/y", "B"), rank=0, sub_question=0, query="q1")
        assert len(pool) == 2
        top = pool.all()[0]
        assert top.key == "https://a.com/x"
        assert top.hits == 2
        assert top.sub_questions == {0, 1}
        assert top.snippet == "a much longer snippet here"
        assert top.url == "https://a.com/x"  # tracking removed, otherwise as found

    def test_ignores_non_http(self):
        pool = CandidatePool()
        assert pool.add({"url": "ftp://x", "title": "x"}, rank=0, sub_question=0, query="q") is None
        assert len(pool) == 0

    def test_select_diverse_caps_domain_then_fills(self):
        pool = CandidatePool()
        for i in range(4):
            pool.add(Hit(f"https://big.com/{i}", f"big{i}"), rank=i, sub_question=0, query="q")
        pool.add(Hit("https://small.org/1", "small"), rank=5, sub_question=0, query="q")
        chosen = select_diverse(pool.all(), 3, max_per_domain=1)
        domains = [c.domain for c in chosen]
        assert domains[:2] == ["big.com", "small.org"]
        assert len(chosen) == 3  # overflow fills the remaining slot

    def test_select_diverse_respects_exclude(self):
        pool = CandidatePool()
        pool.add(Hit("https://a.com/1", "a"), rank=0, sub_question=0, query="q")
        pool.add(Hit("https://b.com/1", "b"), rank=1, sub_question=0, query="q")
        chosen = select_diverse(pool.all(), 5, exclude={"https://a.com/1"})
        assert [c.key for c in chosen] == ["https://b.com/1"]


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


class TestPlanner:
    def test_normalize_plan_clamps_and_filters(self):
        preset = get_depth_preset("quick")  # 2 sub-questions, 2 phrasings
        raw = {
            "sub_questions": [
                {"question": "A?", "queries": ["a1", "a2", "a3"], "platforms": ["github", "myspace", "Hacker News"]},
                "B?",
                {"question": "A?", "queries": ["dup"]},
                {"question": "C?", "queries": ["c"]},
            ],
            "packs": ["finance", "bogus"],
        }
        plan = normalize_plan(raw, "stock price of acme", preset)
        assert plan is not None
        assert [sq.question for sq in plan.sub_questions] == ["A?", "B?"]
        assert plan.sub_questions[0].queries == ["a1", "a2"]
        assert plan.sub_questions[0].platforms == ["github", "hackernews"]
        assert plan.sub_questions[1].queries == ["B?"]
        assert plan.packs == ["finance"]

    def test_normalize_plan_rejects_garbage(self):
        preset = get_depth_preset("quick")
        assert normalize_plan("nope", "q", preset) is None
        assert normalize_plan({"sub_questions": []}, "q", preset) is None

    def test_plan_research_uses_llm(self):
        preset = get_depth_preset("standard")
        plan = plan_research("q?", preset, lambda prompt, schema: (DEFAULT_PLAN, {"total_tokens": 5}))
        assert plan.source == "llm"
        assert len(plan.sub_questions) == 2
        assert plan.usage == {"total_tokens": 5}

    def test_plan_research_falls_back_on_error(self):
        def boom(prompt, schema):
            raise RuntimeError("down")

        plan = plan_research("How do transformers work?", get_depth_preset("quick"), boom)
        assert plan.source == "heuristic"
        assert "down" in (plan.error or "")
        assert plan.sub_questions[0].queries[0] == "How do transformers work?"

    def test_heuristic_plan_variants_and_hints(self):
        plan = heuristic_plan("What is the best Python library for PDFs?", get_depth_preset("standard"))
        assert len(plan.sub_questions[0].queries) == 3
        assert "github" in plan.sub_questions[0].platforms

    def test_suggestions(self):
        assert "arxiv" in suggest_platforms("recent papers on retrieval")
        assert "youtube" in suggest_platforms("best keynote talk about rust")
        assert "finance" in suggest_packs("NVDA share price after earnings")
        assert suggest_packs("how do plants grow") == []


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------


class TestSynthesisHelpers:
    def test_parse_sections(self):
        answer, conflicts, gaps = parse_synthesis(DEFAULT_SYNTHESIS)
        assert answer.startswith("Widget is a toolkit [1].")
        assert conflicts == ["[1] says it is free while [3] says it is paid [8]."]
        assert gaps == ["Pricing history is not covered."]

    def test_parse_none_bullets_and_missing_headings(self):
        _, conflicts, gaps = parse_synthesis(
            "## Answer\nX [1].\n## Conflicting claims\nNone found.\n## Coverage gaps\n- None."
        )
        assert conflicts == [] and gaps == []
        answer, conflicts, gaps = parse_synthesis("Just text [1].")
        assert answer == "Just text [1]." and conflicts == [] and gaps == []

    def test_restrict_citations(self):
        text, dropped = restrict_citations("A [1]. B [2, 7]. C [9].", {"1", "2"})
        assert text == "A [1]. B [2]. C."
        assert sorted(dropped) == ["7", "9"]

    def test_extractive_answer_cites_each_source(self):
        srcs = [OpenedSource(n=1, url="https://a", title="A", content="alpha " * 200)]
        out = extractive_answer(srcs, "budget")
        assert "[1]" in out and "budget" in out
        assert "no cited answer" in extractive_answer([], "x")

    def test_fit_chars(self):
        assert fit_chars_per_source(default_chars=5000, n_sources=4, tokens_remaining=None, answer_tokens=1000) == 5000
        assert fit_chars_per_source(default_chars=5000, n_sources=4, tokens_remaining=3000, answer_tokens=1000) == 1100
        assert fit_chars_per_source(default_chars=5000, n_sources=4, tokens_remaining=500, answer_tokens=1000) == 0

    def test_bundle_trims_head_and_tail(self):
        srcs = [OpenedSource(n=1, url="https://a", title="A", content="H" * 3000 + "T" * 3000)]
        bundle = build_source_bundle(srcs, chars_per_source=1000, compress=False)
        assert bundle.format == "markdown"
        assert bundle.text.startswith("[1] A")
        assert "omitted" in bundle.text and "T" in bundle.text[-50:]

    def test_bundle_toon(self, monkeypatch):
        monkeypatch.setattr("prompture.extraction.core.toon", FakeToon())
        srcs = [
            OpenedSource(n=1, url="https://a", title="A", content="alpha text"),
            OpenedSource(n=2, url="https://b", title="B", content="beta text"),
        ]
        bundle = build_source_bundle(srcs, chars_per_source=1000, compress=True)
        assert bundle.format == "toon"
        assert "alpha text" in bundle.text and "https://b" in bundle.text
        assert bundle.text.startswith("TOON[2]")

    def test_bundle_toon_missing_falls_back(self, monkeypatch):
        monkeypatch.setattr("prompture.extraction.core.toon", None)
        srcs = [OpenedSource(n=1, url="https://a", title="A", content="alpha text")]
        bundle = build_source_bundle(srcs, chars_per_source=1000, compress=True)
        assert bundle.format == "markdown"


# ---------------------------------------------------------------------------
# Agent end-to-end
# ---------------------------------------------------------------------------


class TestResearchAgent:
    def test_full_run_cites_only_opened_sources(self):
        rec = Recorder()
        events: list[Any] = []
        agent = make_agent(rec, on_event=events.append)
        report = agent.run("What is Widget and why use it?")

        opened = report.opened_sources
        assert opened, "expected opened sources"
        assert [s.n for s in opened] == list(range(1, len(opened) + 1))
        assert all(s.n is None for s in report.sources if not s.opened)
        # Every URL the report says it opened was actually read.
        assert {s.url for s in opened} <= set(rec.reads)
        # Citations only reference opened sources.
        import re

        allowed = {str(s.n) for s in opened}
        for marker in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", report.answer + " ".join(report.conflicts)):
            assert {m.strip() for m in marker.split(",")} <= allowed
        assert "[9]" not in report.answer and "7" not in report.answer
        assert any("not opened" in w for w in report.warnings)
        assert report.sources[0].cited
        assert report.conflicts and report.conflicts[0].endswith("paid.")
        assert "Pricing history is not covered." in report.gaps
        assert report.synthesis == "llm"
        assert report.plan_source == "llm"
        # The synthesis prompt only contained opened sources.
        prompt = agent._driver.synthesis_prompts[0]
        for s in report.sources:
            if not s.opened:
                assert f"({s.url})" not in prompt and s.url not in prompt
        # Platform search ran for the sub-question that asked for it.
        assert rec.platform_calls and rec.platform_calls[0][0] == "github"
        # Events flow in order.
        kinds = [e.event_type for e in events]
        assert kinds[0] == "plan" and kinds[-1] == "done"
        assert {"search", "fetch", "synthesize"} <= set(kinds)
        assert events[-1].data["report"] is report

    def test_parallel_fan_out(self):
        rec = Recorder(delay=0.15)
        plan = {
            "sub_questions": [
                {"question": "A?", "queries": ["a1", "a2", "a3"]},
                {"question": "B?", "queries": ["b1", "b2", "b3"]},
            ]
        }
        agent = make_agent(rec, driver=FakeDriver(plan=plan), depth="standard")
        started = time.monotonic()
        agent.run("A and B?")
        elapsed = time.monotonic() - started
        assert sorted(rec.searches) == ["a1", "a2", "a3", "b1", "b2", "b3"]
        assert rec.max_active > 1
        assert elapsed < 6 * 0.15  # would be >= 0.9s sequentially

    def test_fetch_budget_cap(self):
        rec = Recorder(per_query=5)
        agent = make_agent(rec, depth="standard", budget=ResearchBudget(max_fetches=2))
        report = agent.run("What is Widget?")
        assert len(rec.reads) == 2
        assert report.budget_used["fetches"] == 2
        assert report.budget_used["max_fetches"] == 2
        assert "max_fetches" in report.budget_used["limits_hit"]
        assert report.budget_used["degraded"] is True
        assert len(report.opened_sources) == 2
        assert any("stopped early" in g for g in report.gaps)
        assert report.synthesis == "llm"  # still synthesized from what was gathered

    def test_search_budget_cap(self):
        rec = Recorder()
        agent = make_agent(rec, budget=ResearchBudget(max_searches=1))
        report = agent.run("What is Widget?")
        assert len(rec.searches) + len(rec.platform_calls) == 1
        assert "max_searches" in report.budget_used["limits_hit"]

    def test_wall_clock_cap(self):
        release = threading.Event()

        class SlowRec(Recorder):
            def read(self, url: str) -> Page:
                release.wait(5)
                return super().read(url)

        rec = SlowRec()
        agent = make_agent(rec, budget=ResearchBudget(timeout_s=1.0))
        started = time.monotonic()
        try:
            report = agent.run("What is Widget?")
        finally:
            release.set()
        assert time.monotonic() - started < 3.0
        assert "timeout" in report.budget_used["limits_hit"]
        assert report.opened_sources == []
        assert report.synthesis == "none"
        assert any("stopped early" in g for g in report.gaps)

    def test_token_budget_exhausted_gives_extractive_answer(self):
        rec = Recorder()
        # Planning uses 150 tokens; nothing left for synthesis.
        agent = make_agent(rec, budget=ResearchBudget(max_tokens=150))
        report = agent.run("What is Widget?")
        assert report.synthesis == "extractive"
        assert "max_tokens" in report.budget_used["limits_hit"]
        assert len(agent._driver.synthesis_prompts) == 0
        allowed = {str(s.n) for s in report.opened_sources}
        assert allowed and all(f"[{n}]" in report.answer for n in allowed)

    def test_cost_budget_blocks_synthesis(self):
        agent = make_agent(budget=ResearchBudget(max_cost=0.001))
        report = agent.run("What is Widget?")
        assert report.synthesis == "extractive"
        assert "max_cost" in report.budget_used["limits_hit"]
        assert report.cost == pytest.approx(0.001)

    def test_planner_failure_uses_heuristic_plan(self):
        rec = Recorder()
        agent = make_agent(rec, driver=FakeDriver(fail_plan=True))
        report = agent.run("How does Widget work?")
        assert report.plan_source == "heuristic"
        assert any("keyword plan" in w for w in report.warnings)
        assert "How does Widget work?" in rec.searches

    def test_unanswered_sub_question_is_a_gap(self):
        class PickyRec(Recorder):
            def search(self, query: str, **kw: Any) -> Resp:
                if query.startswith("widget benefits"):
                    return Resp(query, [])
                return super().search(query, **kw)

        report = make_agent(PickyRec()).run("What is Widget and why use it?")
        assert "No source could be opened for: Why use Widget?" in report.gaps
        sub = report.sub_questions[1]
        assert sub.answered is False and sub.candidates == 0

    def test_all_searches_fail(self):
        def broken(query: str, **kw: Any) -> Resp:
            raise ConnectionError("offline")

        tools = ResearchTools(search=broken, read=lambda u: Page(u), enable_platforms=False, enable_packs=False)
        driver = FakeDriver()
        report = make_agent(driver=driver, tools=tools).run("What is Widget?")
        assert report.synthesis == "none"
        assert report.opened_sources == []
        assert "No search backend returned results." in report.gaps
        assert any("Every search failed" in w for w in report.warnings)
        assert driver.synthesis_prompts == []
        assert all(not r["ok"] for r in report.routes)

    def test_read_failures_backfill_from_next_candidates(self):
        class FlakyRec(Recorder):
            def read(self, url: str) -> Page:
                with self._lock:
                    self.reads.append(url)
                if "site0" in url:
                    raise TimeoutError("slow site")
                return Page(url)

        plan = {"sub_questions": [{"question": "A?", "queries": ["a"]}]}
        report = make_agent(FlakyRec(per_query=4), driver=FakeDriver(plan=plan)).run("A?")
        assert len(report.opened_sources) == 2  # quick: 2 per sub-question despite the failure
        failed = [s for s in report.sources if not s.opened and s.error]
        assert failed and "TimeoutError" in failed[0].error

    def test_synthesis_failure_falls_back(self):
        driver = FakeDriver(synthesis=RuntimeError("model down"))  # type: ignore[arg-type]
        report = make_agent(driver=driver).run("What is Widget?")
        assert report.synthesis == "extractive"
        assert any("Synthesis failed" in w for w in report.warnings)

    def test_media_sources_are_transcribed(self):
        class VideoRec(Recorder):
            def search(self, query: str, **kw: Any) -> Resp:
                return Resp(query, [Hit("https://youtu.be/abcdefghijk", "talk")])

            def read(self, url: str) -> Page:
                return Page(url, content="short description", kind="video", reader="youtube")

        class Transcript:
            text = "full transcript " * 50

            def to_markdown(self) -> str:
                return "# Transcript\n" + self.text

        transcribed: list[str] = []

        def transcribe(src: str) -> Transcript:
            transcribed.append(src)
            return Transcript()

        rec = VideoRec()
        tools = make_tools(
            rec,
            enable_transcription=True,
            transcribe=transcribe,
            transcription_available=lambda: True,
            enable_platforms=False,
        )
        report = make_agent(rec, tools=tools).run("What did the talk say?")
        assert transcribed == ["https://youtu.be/abcdefghijk"]
        src = report.opened_sources[0]
        assert src.transcribed and src.reader == "youtube+transcript"

    def test_media_not_transcribed_when_disabled(self):
        class VideoRec(Recorder):
            def read(self, url: str) -> Page:
                return Page(url, content="short description of the video that is long enough", kind="video")

        called: list[str] = []
        rec = VideoRec()
        tools = make_tools(rec, transcribe=lambda s: called.append(s), transcription_available=lambda: True)
        agent = make_agent(rec, tools=tools, transcribe_media=False)
        agent.run("What is Widget?")
        assert called == []

    def test_pack_sources(self):
        plan = dict(DEFAULT_PLAN, packs=["finance"])
        runs: list[tuple[str, str]] = []

        def runner(pack: str, question: str) -> tuple[str, dict[str, Any]]:
            runs.append((pack, question))
            return "ACME trades at $12.34 as of today per quote tool.", {"total_tokens": 40, "cost": 0.0005}

        rec = Recorder()
        tools = make_tools(rec, enable_packs=True, pack_runner=runner)
        report = make_agent(rec, driver=FakeDriver(plan=plan), tools=tools).run("ACME stock price?")
        assert runs == [("finance", "ACME stock price?")]
        pack_src = [s for s in report.opened_sources if s.url == "pack:finance"]
        assert pack_src and pack_src[0].reader == "pack:finance"
        assert report.usage["llm_calls"] == 3

    def test_compress_sources_uses_toon(self, monkeypatch):
        monkeypatch.setattr("prompture.extraction.core.toon", FakeToon())
        driver = FakeDriver()
        make_agent(driver=driver, compress_sources=True).run("What is Widget?")
        assert "TOON table" in driver.synthesis_prompts[0]

    def test_report_json_schema(self):
        report = make_agent().run("What is Widget?")
        data = json.loads(json.dumps(report.to_dict()))
        assert set(data) >= {
            "question",
            "answer",
            "sources",
            "sub_questions",
            "gaps",
            "conflicts",
            "budget_used",
            "cost",
            "usage",
            "routes",
            "model",
            "depth",
            "synthesis",
        }
        assert set(data["sources"][0]) >= {"n", "url", "title", "opened", "reader", "cited"}
        assert set(data["budget_used"]) >= {"fetches", "max_fetches", "tokens", "cost", "elapsed_s", "limits_hit"}
        assert data["usage"]["llm_calls"] == 2
        assert data["cost"] == pytest.approx(0.002)
        assert any(r["op"] == "fetch" for r in data["routes"])
        md = report.to_markdown()
        assert md.startswith("# What is Widget?")
        assert "## Sources" in md and "## Conflicting claims" in md and "## Coverage gaps" in md

    def test_run_live_and_arun(self):
        agent = make_agent()
        events = list(agent.run_live("What is Widget?"))
        assert events[-1].event_type == "done"
        assert isinstance(events[-1].data["report"], ResearchReport)
        report = asyncio.run(agent.arun("What is Widget?"))
        assert isinstance(report, ResearchReport)

    def test_empty_question_rejected(self):
        with pytest.raises(ValueError):
            make_agent().run("   ")

    def test_on_event_errors_are_ignored(self):
        def bad(ev: Any) -> None:
            raise RuntimeError("ui crashed")

        report = make_agent(on_event=bad).run("What is Widget?")
        assert report.synthesis == "llm"

    def test_unknown_depth(self):
        with pytest.raises(ValueError):
            ResearchAgent("fake/model", depth="extreme")


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------


class TestDefaultModel:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PROMPTURE_RESEARCH_MODEL", "groq/llama-3.1-8b-instant")
        assert default_research_model() == "groq/llama-3.1-8b-instant"

    def test_none_configured_raises(self, monkeypatch):
        monkeypatch.setattr("prompture.research.agent.default_research_model", lambda: None)
        with pytest.raises(ConfigurationError):
            ResearchAgent()


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class TestResearchTool:
    def test_returns_markdown(self):
        rec = Recorder()
        td = research_tool(driver=FakeDriver(), tools=make_tools(rec))
        assert td.name == "research"
        assert td.parameters["required"] == ["question"]
        out = td.function(question="What is Widget?")
        assert out.startswith("# What is Widget?") and "## Sources" in out

    def test_never_raises(self, monkeypatch):
        monkeypatch.setattr("prompture.research.agent.default_research_model", lambda: None)
        td = research_tool()
        out = td.function(question="anything")
        assert out.startswith("Research failed: ConfigurationError")
        out = research_tool(driver=FakeDriver(), tools=make_tools(Recorder())).function(question="  ")
        assert out.startswith("Research failed: ValueError")
        out = research_tool(driver=FakeDriver(), depth="quick").function(question="q", depth="bogus")
        assert out.startswith("Research failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_cli_agent(monkeypatch):
    import prompture.research as research_pkg

    captured: dict[str, Any] = {}
    real = research_pkg.ResearchAgent

    class CliAgent(real):  # type: ignore[misc,valid-type]
        def __init__(self, model=None, **kw):
            captured["model"] = model
            captured.update(kw)
            rec = Recorder()
            kw["tools"] = make_tools(rec)
            super().__init__(model or "fake/model", driver=FakeDriver(), **kw)

    monkeypatch.setattr(research_pkg, "ResearchAgent", CliAgent)
    return captured


class TestResearchCli:
    def test_markdown_output(self, fake_cli_agent):
        from prompture.cli.research_cmd import research

        result = CliRunner().invoke(research, ["What", "is", "Widget?", "--depth", "quick", "--max-fetches", "3"])
        assert result.exit_code == 0, result.output
        assert "# What is Widget?" in result.output
        assert fake_cli_agent["depth"] == "quick"
        assert fake_cli_agent["budget"].max_fetches == 3

    def test_json_output(self, fake_cli_agent):
        from prompture.cli.research_cmd import research

        result = CliRunner().invoke(
            research,
            ["What is Widget?", "--json", "-q", "--model", "x/y", "--max-cost", "1.5", "--timeout", "30"],
        )
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data["question"] == "What is Widget?"
        assert data["sources"] and data["budget_used"]["max_cost"] == 1.5
        assert fake_cli_agent["model"] == "x/y"
        assert fake_cli_agent["budget"].timeout_s == 30

    def test_missing_model_is_a_clean_error(self, monkeypatch):
        from prompture.cli.research_cmd import research

        monkeypatch.setattr("prompture.research.agent.default_research_model", lambda: None)
        result = CliRunner().invoke(research, ["anything"])
        assert result.exit_code != 0
        assert "--model" in result.output

    def test_commands_export(self):
        from prompture.cli import research_cmd

        assert [c.name for c in research_cmd.COMMANDS] == ["research"]


# ---------------------------------------------------------------------------
# Live (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_quick_research():
    model = default_research_model()
    if not model:
        pytest.skip("no provider configured")
    pytest.importorskip("prompture.tools.web")
    report = ResearchAgent(model, depth="quick", budget=ResearchBudget(max_fetches=3, timeout_s=120)).run(
        "What is the latest stable Python release?"
    )
    assert report.opened_sources
    assert report.budget_used["fetches"] <= 3


class ToolCapableDriver(FakeDriver):
    """FakeDriver that also answers the pack sub-agent's tool-calling turns."""

    supports_messages = True
    supports_tool_use = True

    def generate_messages(self, messages, options):
        return {"text": "ACME closed at $12.34 today according to the quote tool.", "meta": dict(META)}

    def generate_messages_with_tools(self, messages, tools, options):
        with self._lock:
            self.options.append({"tools": [t.get("function", t).get("name") for t in tools]})
        return {
            "text": "ACME closed at $12.34 today according to the quote tool.",
            "meta": dict(META),
            "tool_calls": [],
            "stop_reason": "stop",
        }


def test_default_pack_runner_uses_agent_with_pack_tools():
    from prompture.agents.tools_schema import ToolDefinition

    def quote(symbol: str) -> str:
        return f"{symbol}: 12.34"

    pack_def = ToolDefinition(
        name="stock_quote",
        description="Latest quote for a ticker.",
        parameters={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
        function=quote,
    )
    requested: list[str] = []

    def pack_tools(name: str) -> list[ToolDefinition]:
        requested.append(name)
        return [pack_def]

    rec = Recorder()
    driver = ToolCapableDriver(plan=dict(DEFAULT_PLAN, packs=["finance"]))
    tools = make_tools(rec, enable_packs=True, pack_tools=pack_tools)
    report = make_agent(rec, driver=driver, tools=tools).run("ACME stock price?")
    assert requested == ["finance"]
    pack_src = [s for s in report.opened_sources if s.url == "pack:finance"]
    assert pack_src, report.routes
    assert any("stock_quote" in (o.get("tools") or []) for o in driver.options)
