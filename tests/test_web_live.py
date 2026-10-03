"""Live smoke tests for the keyless web tools (``pytest --run-integration``)."""

from __future__ import annotations

import pytest

from prompture.tools.web import clear_fetch_cache, read_url, search_platform, web_fetch, web_search

pytestmark = pytest.mark.integration


def test_live_keyless_search():
    resp = web_search("python programming language", max_results=3, providers=["exa_mcp"])
    assert resp.served_by == "exa_mcp"
    assert resp.results
    assert all(r.url.startswith(("http://", "https://")) for r in resp.results)


def test_live_fetch_example_com():
    clear_fetch_cache()
    res = web_fetch("https://example.com/", max_chars=2000)
    assert "Example Domain" in (res.title + res.content)


def test_live_read_arxiv_and_hn():
    paper = read_url("https://arxiv.org/abs/1706.03762")
    assert paper.reader == "arxiv"
    assert "Attention" in paper.title
    thread = read_url("https://news.ycombinator.com/item?id=8863")
    assert thread.reader == "hackernews"


def test_live_platform_hackernews():
    hits = search_platform("hackernews", "python", max_results=3)
    assert hits
