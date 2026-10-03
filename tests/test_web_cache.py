"""Tests for the local web cache: lifetimes, hits, bypasses and persistence."""

from __future__ import annotations

import pytest

from prompture.capabilities.backends import ChainResult
from prompture.tools.web import cache as web_cache
from prompture.tools.web import fetch as fetch_mod
from prompture.tools.web import platform as platform_mod
from prompture.tools.web import readers as readers_mod
from prompture.tools.web import search as search_mod
from prompture.tools.web._types import SearchResponse, SearchResult
from prompture.tools.web.fetch import FetchedPage
from prompture.tools.web.readers.base import ReadResult

PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    monkeypatch.setattr("prompture.capabilities.url_safety.resolve_host", lambda host, port=None: [PUBLIC_IP])


# ---------------------------------------------------------------------------
# Lifetimes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "recency", "expected"),
    [
        ("weather in Miami", None, 10 * 60),
        ("bitcoin price", None, 10 * 60),
        ("latest news on the election", None, 10 * 60),
        ("python asyncio tutorial", 1, 10 * 60),
        ("python asyncio tutorial", 7, 3600),
        ("how does a b-tree work", None, 6 * 3600),
    ],
)
def test_search_ttl(query, recency, expected):
    assert web_cache.search_ttl(query, recency) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/2401.00001", 7 * 86400),
        ("https://example.com/paper.pdf", 7 * 86400),
        ("https://doi.org/10.1000/xyz", 7 * 86400),
        ("https://github.com/o/r/blob/" + "a" * 40 + "/README.md", 7 * 86400),
        ("https://en.wikipedia.org/wiki/Python", 86400),
        ("https://news.ycombinator.com/news", 10 * 60),
        ("https://www.reddit.com/r/python", 10 * 60),
        ("https://status.openai.com/", 10 * 60),
        ("https://example.com/blog/post", 3600),
    ],
)
def test_page_ttl(url, expected):
    assert web_cache.page_ttl(url) == expected


def test_read_ttl_by_reader():
    assert web_cache.read_ttl("https://youtu.be/x", "youtube", "video") == 7 * 86400
    assert web_cache.read_ttl("https://github.com/o/r/issues/1", "github", "issue") == 10 * 60
    assert web_cache.read_ttl("https://github.com/o/r", "github", "repo") == 3600
    assert web_cache.read_ttl("https://example.com/feed.xml", "feeds", "feed") == 15 * 60
    assert web_cache.read_ttl("https://news.ycombinator.com/item?id=1", "hackernews", "thread") == 10 * 60


def test_mode_parsing(monkeypatch):
    monkeypatch.setenv("PROMPTURE_WEB_CACHE", "off")
    assert web_cache.cache_mode() == "off"
    monkeypatch.setenv("PROMPTURE_WEB_CACHE", "false")
    assert web_cache.cache_mode() == "off"
    monkeypatch.setenv("PROMPTURE_WEB_CACHE", "bogus")
    assert web_cache.cache_mode() == "disk"
    monkeypatch.delenv("PROMPTURE_WEB_CACHE")
    assert web_cache.cache_mode() == "disk"


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def _counting_search(monkeypatch, results=1, fail_first=False):
    calls = {"n": 0}

    def fake_run_search(chain, request, *, only=None):
        calls["n"] += 1
        if fail_first and calls["n"] == 1:
            raise RuntimeError("backend down")
        hits = [SearchResult(title=f"T{i}", url=f"https://example.com/{i}", snippet="s") for i in range(results)]
        return SearchResponse(
            query=request.query,
            results=hits,
            served_by="exa_mcp",
            route={"served_by": "exa_mcp", "fallback": False, "attempts": []},
        )

    monkeypatch.setattr(search_mod, "run_search", fake_run_search)
    return calls


def test_search_second_call_is_cached(monkeypatch):
    calls = _counting_search(monkeypatch)
    first = search_mod.web_search("how does a b-tree work")
    second = search_mod.web_search("how does a b-tree work")
    assert calls["n"] == 1
    assert "cached" not in first.route
    assert second.route["cached"] is True and second.route["cache_age_s"] >= 0
    assert second.served_by == "exa_mcp"
    assert [r.url for r in second.results] == [r.url for r in first.results]
    assert "cached just now" in second.to_markdown()


def test_search_key_includes_parameters(monkeypatch):
    calls = _counting_search(monkeypatch)
    search_mod.web_search("q", max_results=3)
    search_mod.web_search("q", max_results=5)
    search_mod.web_search("q", max_results=3, include_domains=["example.com"])
    search_mod.web_search("q", max_results=3, providers=["exa_mcp"])
    assert calls["n"] == 4


def test_search_bypass_and_zero_ttl(monkeypatch):
    calls = _counting_search(monkeypatch)
    search_mod.web_search("q", use_cache=False)
    search_mod.web_search("q", use_cache=False)
    assert calls["n"] == 2
    search_mod.web_search("q2", cache_ttl=0)
    search_mod.web_search("q2")
    assert calls["n"] == 4


def test_empty_and_failed_searches_are_not_cached(monkeypatch):
    calls = _counting_search(monkeypatch, results=0)
    search_mod.web_search("nothing here")
    search_mod.web_search("nothing here")
    assert calls["n"] == 2

    calls = _counting_search(monkeypatch, fail_first=True)
    with pytest.raises(RuntimeError):
        search_mod.web_search("flaky")
    search_mod.web_search("flaky")
    search_mod.web_search("flaky")
    assert calls["n"] == 2  # the failure wasn't stored; the success was


def test_off_mode_never_caches(monkeypatch):
    monkeypatch.setenv("PROMPTURE_WEB_CACHE", "off")
    calls = _counting_search(monkeypatch)
    search_mod.web_search("q")
    search_mod.web_search("q")
    assert calls["n"] == 2


def test_disk_cache_survives_a_new_process(monkeypatch, tmp_path):
    monkeypatch.setenv("PROMPTURE_WEB_CACHE", "disk")
    monkeypatch.setenv("PROMPTURE_WEB_CACHE_PATH", str(tmp_path / "web.db"))
    calls = _counting_search(monkeypatch)
    search_mod.web_search("how does a b-tree work")
    web_cache._reset_for_tests()  # what a fresh process would see
    again = search_mod.web_search("how does a b-tree work")
    assert calls["n"] == 1 and again.route["cached"] is True
    info = web_cache.cache_info()
    assert info["mode"] == "disk" and info["entries"] == 1
    web_cache.clear_web_cache()
    assert web_cache.cache_info()["entries"] == 0


def test_expired_entries_are_refetched(monkeypatch):
    calls = _counting_search(monkeypatch)
    search_mod.web_search("weather in Miami")
    now = web_cache.time.time()
    monkeypatch.setattr("prompture.infra.cache.time.time", lambda: now + 11 * 60)
    search_mod.web_search("weather in Miami")
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# web_fetch / read_url / platform search
# ---------------------------------------------------------------------------


def _counting_fetch(monkeypatch, content="hello world"):
    calls = {"n": 0}

    class FakeChain:
        def run(self, url, only=None):
            calls["n"] += 1
            page = FetchedPage(url=url, final_url=url, title="T", content=content, content_type="text/html")
            return ChainResult(page, "direct", [{"backend": "direct", "status": "ok"}])

    monkeypatch.setattr(fetch_mod, "fetch_chain", lambda **kw: FakeChain())
    return calls


def test_fetch_is_cached_and_paged_from_cache(monkeypatch):
    calls = _counting_fetch(monkeypatch, content="x" * 100)
    first = fetch_mod.web_fetch("https://example.com/a", max_chars=40)
    second = fetch_mod.web_fetch("https://example.com/a", max_chars=40, start=first.next_start)
    assert calls["n"] == 1
    assert first.cached is False and second.cached is True
    assert second.route["cached"] is True


def test_fetch_validates_url_before_cache(monkeypatch):
    from prompture.capabilities import UnsafeURLError

    _counting_fetch(monkeypatch)
    with pytest.raises(UnsafeURLError):
        fetch_mod.web_fetch("http://127.0.0.1/admin")


def test_read_url_caches_full_content(monkeypatch):
    calls = {"n": 0}

    def fake_live(safe_url, *, reader, fallback, session, **kwargs):
        calls["n"] += 1
        return ReadResult(
            url=safe_url,
            title="Video",
            content="transcript " * 50,
            reader="youtube",
            kind="video",
            route={"reader": "youtube", "served_by": "youtube/transcript_api", "fallback": False, "attempts": []},
        )

    monkeypatch.setattr(readers_mod, "_read_url_live", fake_live)
    first = readers_mod.read_url("https://www.youtube.com/watch?v=abc", max_chars=100)
    second = readers_mod.read_url("https://www.youtube.com/watch?v=abc", max_chars=100, start=first.next_start)
    assert calls["n"] == 1
    assert first.truncated and second.route["cached"] is True
    assert second.content != first.content  # a different slice of the same cached transcript
    readers_mod.read_url("https://www.youtube.com/watch?v=abc", languages=["es"])
    assert calls["n"] == 2  # different reader options, different entry


def test_platform_search_is_cached(monkeypatch):
    calls = {"n": 0}

    def fake_live(chain, query, *, max_results, kind, session):
        calls["n"] += 1
        return SearchResponse(
            query=query,
            results=[SearchResult(title="r", url="https://github.com/o/r", snippet="")],
            served_by="github/rest",
            route={"served_by": "rest", "fallback": False, "attempts": []},
        )

    monkeypatch.setattr(platform_mod, "_platform_search_live", fake_live)
    platform_mod.search_platform("github", "vector db")
    results = platform_mod.search_platform("github", "vector db")
    assert calls["n"] == 1 and results[0].url == "https://github.com/o/r"
    platform_mod.search_platform("github", "vector db", kind="issues")
    assert calls["n"] == 2


def test_cache_info_never_creates_the_database(monkeypatch, tmp_path):
    db = tmp_path / "sub" / "web.db"
    monkeypatch.setenv("PROMPTURE_WEB_CACHE", "disk")
    monkeypatch.setenv("PROMPTURE_WEB_CACHE_PATH", str(db))
    assert web_cache.cache_info() == {"mode": "disk", "path": str(db), "entries": 0}
    assert not db.exists() and not db.parent.exists()

    from prompture.tools.web.health import check_web_cache

    assert check_web_cache().status == "ok"
    assert not db.exists()
