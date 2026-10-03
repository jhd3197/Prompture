"""Tests for ``prompture.tools.web.search`` — backend chain, keyless floor, failover.

No network: every HTTP call goes to a fake ``requests.Session``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from prompture.capabilities import AllBackendsFailedError
from prompture.tools import WebSearchTool
from prompture.tools.web import (
    SEARCH_BACKENDS,
    SearchResponse,
    SearchResult,
    asearch,
    search_chain,
    web_search,
)
from prompture.tools.web import search as search_mod
from prompture.tools.web._mcp_http import MCPError, MCPHttpClient, parse_sse_messages
from prompture.tools.web.search import parse_exa_mcp_text

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResp:
    def __init__(
        self,
        status: int = 200,
        body: bytes | str = b"",
        *,
        json_data: Any = None,
        headers: dict[str, str] | None = None,
        url: str = "",
    ) -> None:
        self.headers = CaseInsensitiveDict(headers or {})
        if json_data is not None:
            body = json.dumps(json_data)
            self.headers.setdefault("Content-Type", "application/json")
        self.content = body.encode() if isinstance(body, str) else body
        self.status_code = status
        self.reason = "OK" if status < 400 else "Error"
        self.encoding = "utf-8"
        self.url = url

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")

    def json(self) -> Any:
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: {self.reason} for url: {self.url}", response=self
            )

    def iter_content(self, chunk_size: int = 65536):
        yield self.content

    def close(self) -> None:
        pass


Handler = Callable[[str, str, dict[str, Any]], FakeResp]


class FakeSession:
    """Routes requests by URL substring; records every call."""

    def __init__(self, routes: list[tuple[str, str, FakeResp | Handler]] | None = None) -> None:
        self.routes = routes or []
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _dispatch(self, method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        self.calls.append((method, url, kw))
        for m, needle, resp in self.routes:
            if m == method and needle in url:
                out = resp(method, url, kw) if callable(resp) else resp
                out.url = out.url or url
                return out
        raise AssertionError(f"unexpected {method} {url}")

    def get(self, url: str, **kw: Any) -> FakeResp:
        return self._dispatch("GET", url, kw)

    def post(self, url: str, **kw: Any) -> FakeResp:
        return self._dispatch("POST", url, kw)

    def delete(self, url: str, **kw: Any) -> FakeResp:
        return self._dispatch("DELETE", url, kw)

    def called(self, needle: str) -> list[tuple[str, str, dict[str, Any]]]:
        return [c for c in self.calls if needle in c[1]]


EXA_TEXT = (
    "Title: First Hit\nURL: https://one.example/page\nPublished: 2026-09-30T00:00:00.000Z\nAuthor: N/A\n"
    "Highlights:\nFirst snippet line.\n...\nMore text.\n\n---\n\n"
    "Title: Second Hit\nURL: https://two.example/\nPublished: N/A\nAuthor: Jane\nHighlights:\nSecond snippet."
)


def sse(payload: dict[str, Any]) -> str:
    return f"event: message\ndata: {json.dumps(payload)}\n\n"


def exa_mcp_handler(text: str = EXA_TEXT, *, record: list[dict[str, Any]] | None = None) -> Handler:
    def handler(method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        payload = kw["json"]
        if record is not None:
            record.append({"payload": payload, "headers": dict(kw.get("headers") or {})})
        if payload.get("method") == "initialize":
            return FakeResp(
                200,
                sse({"jsonrpc": "2.0", "id": payload["id"], "result": {"protocolVersion": "2025-03-26"}}),
                headers={"Content-Type": "text/event-stream", "Mcp-Session-Id": "sess-1"},
            )
        if "id" not in payload:
            return FakeResp(202, b"")
        assert payload["method"] == "tools/call"
        result = {"content": [{"type": "text", "text": text}]}
        return FakeResp(
            200,
            sse({"jsonrpc": "2.0", "id": payload["id"], "result": result}),
            headers={"Content-Type": "text/event-stream"},
        )

    return handler


KEY_ENV = (
    "TAVILY_API_KEY",
    "EXA_API_KEY",
    "SERPER_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "JINA_API_KEY",
    "SEARXNG_ENDPOINT",
    "PROMPTURE_SEARCH_PROVIDERS",
    "PROMPTURE_PROXY",
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """No keys from the developer's environment; fresh MCP client cache."""
    for var in KEY_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("prompture.tools.web._common.settings", SimpleNamespace())
    empty = SimpleNamespace(
        tavily_api_key=None,
        serper_api_key=None,
        brave_search_api_key=None,
        searxng_endpoint=None,
        exa_api_key=None,
        jina_api_key=None,
    )
    monkeypatch.setattr("prompture.tools.web_search.settings", empty)
    search_mod._mcp_clients.clear()
    yield
    search_mod._mcp_clients.clear()


# ---------------------------------------------------------------------------
# Keyless floor
# ---------------------------------------------------------------------------


def test_unconfigured_uses_keyless_exa_mcp():
    record: list[dict[str, Any]] = []
    sess = FakeSession([("POST", "mcp.exa.ai", exa_mcp_handler(record=record))])
    resp = web_search("what is new", max_results=5, session=sess)

    assert isinstance(resp, SearchResponse)
    assert resp.served_by == "exa_mcp"
    assert [r.url for r in resp.results] == ["https://one.example/page", "https://two.example/"]
    assert resp.results[0].snippet.startswith("First snippet line.")
    assert resp.results[1].extra["author"] == "Jane"
    assert resp.route["fallback"] is False
    skipped = [a["backend"] for a in resp.route["attempts"] if a["status"] == "skipped"]
    assert skipped == ["tavily", "exa", "serper", "brave", "jina", "searxng"]
    md = resp.to_markdown()
    assert "_served by exa_mcp_" in md
    assert "<https://one.example/page>" in md

    methods = [r["payload"].get("method") for r in record]
    assert methods == ["initialize", "notifications/initialized", "tools/call"]
    call = record[-1]
    assert call["headers"]["Mcp-Session-Id"] == "sess-1"
    assert call["payload"]["params"]["name"] == "web_search_exa"
    assert call["payload"]["params"]["arguments"]["query"] == "what is new"


def test_exa_mcp_session_is_reused_across_calls():
    record: list[dict[str, Any]] = []
    sess = FakeSession([("POST", "mcp.exa.ai", exa_mcp_handler(record=record))])
    web_search("a", session=sess)
    web_search("b", session=sess)
    methods = [r["payload"].get("method") for r in record]
    assert methods.count("initialize") == 1
    assert methods.count("tools/call") == 2


def test_exa_mcp_retries_without_objective_on_invalid_params():
    calls: list[dict[str, Any]] = []
    base = exa_mcp_handler()

    def handler(method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        payload = kw["json"]
        if payload.get("method") == "tools/call":
            calls.append(dict(payload["params"]["arguments"]))
            if "objective" in payload["params"]["arguments"]:
                return FakeResp(
                    200,
                    sse({"jsonrpc": "2.0", "id": payload["id"], "error": {"code": -32602, "message": "bad objective"}}),
                    headers={"Content-Type": "text/event-stream"},
                )
        return base(method, url, kw)

    resp = web_search("q", session=FakeSession([("POST", "mcp.exa.ai", handler)]))
    assert len(resp.results) == 2
    assert "objective" in calls[0]
    assert "objective" not in calls[1]


def test_parse_exa_mcp_json_shape():
    text = json.dumps(
        {"results": [{"title": "T", "url": "https://x.example", "text": "body", "publishedDate": "2026-01-01"}]}
    )
    out = parse_exa_mcp_text(text)
    assert out[0].title == "T"
    assert out[0].snippet == "body"
    assert out[0].extra["published_date"] == "2026-01-01"


def test_parse_exa_mcp_text_shape_drops_na():
    out = parse_exa_mcp_text(EXA_TEXT)
    assert len(out) == 2
    assert out[1].extra["published_date"] is None
    assert out[0].extra["author"] is None


# ---------------------------------------------------------------------------
# Failover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status,category", [(429, "rate_limit"), (401, "auth")])
def test_keyed_provider_failure_falls_over_to_keyless(monkeypatch, status, category):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    sess = FakeSession(
        [
            ("POST", "api.tavily.com", FakeResp(status, b'{"detail": "nope"}')),
            ("POST", "mcp.exa.ai", exa_mcp_handler()),
        ]
    )
    resp = web_search("q", session=sess)
    assert resp.served_by == "exa_mcp"
    assert resp.route["fallback"] is True
    failed = [a for a in resp.route["attempts"] if a["status"] == "error"]
    assert failed[0]["backend"] == "tavily"
    assert failed[0]["category"] == category
    assert len(sess.called("api.tavily.com")) == 1  # no retry for auth / rate limit
    assert "fallback after tavily failed" in resp.to_markdown()


def test_bad_request_from_one_provider_is_not_fatal(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    sess = FakeSession(
        [
            ("POST", "google.serper.dev", FakeResp(400, b"bad")),
            ("POST", "mcp.exa.ai", exa_mcp_handler()),
        ]
    )
    resp = web_search("q", session=sess)
    assert resp.served_by == "exa_mcp"
    assert resp.route["attempts"][2]["category"] == "unsupported"


def test_transient_error_is_retried_once(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
    calls = {"n": 0}

    def flaky(method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("reset")
        return FakeResp(
            json_data={"web": {"results": [{"title": "B", "url": "https://b.example", "description": "d"}]}}
        )

    monkeypatch.setattr(search_mod.BackendChain, "__init__", _fast_chain_init(search_mod.BackendChain.__init__))
    resp = web_search("q", session=FakeSession([("GET", "api.search.brave.com", flaky)]))
    assert resp.served_by == "brave"
    assert calls["n"] == 2
    assert resp.route["fallback"] is True  # the first try failed


def _fast_chain_init(orig):
    def init(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        self.retry_delay = 0

    return init


def test_all_backends_failing_raises_with_attempts(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    sess = FakeSession(
        [
            ("POST", "api.tavily.com", FakeResp(401, b"")),
            ("POST", "mcp.exa.ai", FakeResp(403, b"forbidden")),
        ]
    )
    with pytest.raises(AllBackendsFailedError) as info:
        web_search("q", session=sess)
    assert [a["backend"] for a in info.value.attempts if a["status"] == "error"] == ["tavily", "exa_mcp"]


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_override_env_reorders(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setenv("PROMPTURE_SEARCH_PROVIDERS", "exa_mcp,tavily,unknown_backend")
    sess = FakeSession([("POST", "mcp.exa.ai", exa_mcp_handler())])
    resp = web_search("q", session=sess)
    assert resp.served_by == "exa_mcp"
    assert not sess.called("tavily")


def test_default_order_puts_keyed_first_and_exa_mcp_last():
    names = [b.name for b in search_chain().ordered()]
    assert names[-1] == "exa_mcp"
    assert names[0] == "tavily"
    assert set(names) == set(SEARCH_BACKENDS)


def test_providers_argument_restricts(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    sess = FakeSession(
        [
            (
                "GET",
                "api.search.brave.com",
                FakeResp(json_data={"web": {"results": [{"title": "B", "url": "https://b.example"}]}}),
            )
        ]
    )
    resp = web_search("q", providers=["brave"], session=sess)
    assert resp.served_by == "brave"
    assert [a["backend"] for a in resp.route["attempts"]] == ["brave"]


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unknown search provider"):
        web_search("q", providers=["bing"])


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        web_search("   ")


# ---------------------------------------------------------------------------
# Filters, dedupe, provider payloads
# ---------------------------------------------------------------------------


def test_domain_filters_and_dedupe(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    payload = {
        "results": [
            {"title": "A", "url": "https://docs.python.org/3/a", "content": "a"},
            {"title": "A dup", "url": "https://www.docs.python.org/3/a/?utm_source=x", "content": "a"},
            {"title": "Sub", "url": "https://sub.docs.python.org/b", "content": "b"},
            {"title": "Other", "url": "https://other.example/c", "content": "c"},
            {"title": "Excluded", "url": "https://bad.docs.python.org/d", "content": "d"},
            {"title": "No URL", "url": "", "content": "e"},
        ]
    }
    sess = FakeSession([("POST", "api.tavily.com", FakeResp(json_data=payload))])
    resp = web_search(
        "q",
        include_domains=["https://docs.python.org/"],
        exclude_domains=["bad.docs.python.org"],
        session=sess,
    )
    assert [r.title for r in resp.results] == ["A", "Sub"]
    sent = sess.calls[0][2]["json"]
    assert sent["include_domains"] == ["docs.python.org"]
    assert sent["exclude_domains"] == ["bad.docs.python.org"]


def test_recency_filter_drops_old_dated_results(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    new = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    payload = {
        "results": [
            {"title": "Old", "url": "https://o.example", "published_date": old},
            {"title": "New", "url": "https://n.example", "published_date": new},
            {"title": "Undated", "url": "https://u.example"},
        ]
    }
    sess = FakeSession([("POST", "api.tavily.com", FakeResp(json_data=payload))])
    resp = web_search("q", recency_days=7, session=sess)
    assert [r.title for r in resp.results] == ["New", "Undated"]
    assert sess.calls[0][2]["json"]["time_range"] == "week"


def test_max_results_caps_output(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    payload = {"results": [{"title": str(i), "url": f"https://x{i}.example"} for i in range(10)]}
    sess = FakeSession([("POST", "api.tavily.com", FakeResp(json_data=payload))])
    assert len(web_search("q", max_results=3, session=sess).results) == 3


def test_exa_rest_payload_and_parsing(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    payload = {"results": [{"title": "E", "url": "https://e.example", "highlights": ["h1", "h2"], "score": 0.9}]}
    sess = FakeSession([("POST", "api.exa.ai/search", FakeResp(json_data=payload))])
    resp = web_search("q", include_domains=["e.example"], recency_days=30, session=sess)
    assert resp.served_by == "exa"
    assert resp.results[0].snippet == "h1 h2"
    _, _, kw = sess.calls[0]
    assert kw["headers"]["x-api-key"] == "exa-key"
    assert kw["json"]["includeDomains"] == ["e.example"]
    assert kw["json"]["startPublishedDate"].endswith("Z")
    assert kw["json"]["contents"]


def test_jina_search_uses_site_header_and_bearer(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "jina_key")
    data = {"data": [{"title": "J", "url": "https://j.example/x", "description": "desc"}]}
    sess = FakeSession([("GET", "s.jina.ai", FakeResp(json_data=data))])
    resp = web_search("q", providers=["jina"], include_domains=["j.example"], session=sess)
    assert resp.results[0].snippet == "desc"
    kw = sess.calls[0][2]
    assert kw["headers"]["X-Site"] == "j.example"
    assert kw["headers"]["Authorization"] == "Bearer jina_key"
    assert kw["params"] == {"q": "q"}


def test_serper_operators_and_recency(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    sess = FakeSession([("POST", "google.serper.dev", FakeResp(json_data={"organic": []}))])
    web_search("q", include_domains=["a.com"], exclude_domains=["b.com"], recency_days=1, session=sess)
    sent = sess.calls[0][2]["json"]
    assert sent["q"] == "q site:a.com -site:b.com"
    assert sent["tbs"] == "qdr:d"


def test_searxng_backend_with_endpoint(monkeypatch):
    monkeypatch.setenv("SEARXNG_ENDPOINT", "https://searx.example/")
    sess = FakeSession(
        [("GET", "searx.example/search", FakeResp(json_data={"results": [{"title": "S", "url": "https://s.example"}]}))]
    )
    resp = web_search("q", session=sess)
    assert resp.served_by == "searxng"


def test_asearch_runs_chain():
    sess = FakeSession([("POST", "mcp.exa.ai", exa_mcp_handler())])
    resp = asyncio.run(asearch("q", session=sess))
    assert resp.served_by == "exa_mcp"


def test_search_response_markdown_empty_and_answer():
    empty = SearchResponse(query="nothing", results=[], served_by="exa_mcp", route={"fallback": False})
    assert 'No results for "nothing"' in empty.to_markdown()
    full = SearchResponse(
        query="q",
        results=[SearchResult("T", "https://t.example", "s")],
        served_by="tavily",
        route={},
        answer="42",
    )
    md = full.to_markdown()
    assert md.startswith("**Answer:** 42")
    assert md.rstrip().endswith("_served by tavily_")


def test_offline_health_check_touches_no_network():
    class Exploding:
        def __getattr__(self, name):
            raise AssertionError("network used in offline check")

    row = search_chain(session=Exploding()).check(False)  # type: ignore[arg-type]
    assert row.status == "ok"
    assert row.active_backend == "exa_mcp"
    assert row.fix_hint and "TAVILY_API_KEY" in row.fix_hint


# ---------------------------------------------------------------------------
# MCP client
# ---------------------------------------------------------------------------


def test_mcp_client_json_responses_and_tool_error():
    def handler(method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        payload = kw["json"]
        if payload.get("method") == "initialize":
            return FakeResp(json_data={"jsonrpc": "2.0", "id": payload["id"], "result": {}})
        if "id" not in payload:
            return FakeResp(202, b"")
        return FakeResp(
            json_data={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"isError": True, "content": [{"type": "text", "text": "quota"}]},
            }
        )

    client = MCPHttpClient("https://mcp.example/mcp", session=FakeSession([("POST", "mcp.example", handler)]))  # type: ignore[arg-type]
    with pytest.raises(MCPError, match="quota"):
        client.call_tool("t", {})
    assert client.session_id is None


def test_mcp_client_reinitializes_on_expired_session():
    state = {"inits": 0, "calls": 0}

    def handler(method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        payload = kw["json"]
        if payload.get("method") == "initialize":
            state["inits"] += 1
            return FakeResp(
                json_data={"jsonrpc": "2.0", "id": payload["id"], "result": {}},
                headers={"Mcp-Session-Id": f"s{state['inits']}"},
            )
        if "id" not in payload:
            return FakeResp(202, b"")
        state["calls"] += 1
        if state["calls"] == 1:
            return FakeResp(404, b"session expired")
        assert kw["headers"]["Mcp-Session-Id"] == "s2"
        return FakeResp(json_data={"jsonrpc": "2.0", "id": payload["id"], "result": {"content": []}})

    client = MCPHttpClient("https://mcp.example/mcp", session=FakeSession([("POST", "mcp.example", handler)]))  # type: ignore[arg-type]
    assert client.call_tool("t", {}) == {"content": []}
    assert state["inits"] == 2


def test_parse_sse_messages_handles_multiline_and_comments():
    body = ': ping\n\nevent: message\ndata: {"id": 1,\ndata:  "result": {}}\n\n'
    assert parse_sse_messages(body) == [{"id": 1, "result": {}}]


# ---------------------------------------------------------------------------
# WebSearchTool compatibility
# ---------------------------------------------------------------------------


def test_web_search_tool_auto_mode_is_keyless_and_has_footer():
    sess = FakeSession([("POST", "mcp.exa.ai", exa_mcp_handler())])
    tool = WebSearchTool(session=sess)  # type: ignore[arg-type]
    assert tool.provider == "exa_mcp"
    out = tool.to_tool_definition().function(query="anything")
    assert "First Hit" in out
    assert "_served by exa_mcp_" in out


def test_web_search_tool_auto_mode_fails_over(monkeypatch):
    monkeypatch.setattr(
        "prompture.tools.web_search.settings",
        SimpleNamespace(
            tavily_api_key="k",
            serper_api_key=None,
            brave_search_api_key=None,
            searxng_endpoint=None,
            exa_api_key=None,
            jina_api_key=None,
        ),
    )
    sess = FakeSession(
        [
            ("POST", "api.tavily.com", FakeResp(429, b"slow down")),
            ("POST", "mcp.exa.ai", exa_mcp_handler()),
        ]
    )
    tool = WebSearchTool(session=sess)  # type: ignore[arg-type]
    assert tool.provider == "tavily"
    results = tool.search("q")
    assert results[0].url == "https://one.example/page"
    assert tool.last_response is not None
    assert tool.last_response.route["fallback"] is True


def test_web_search_tool_pinned_provider_errors_are_scrubbed():
    sess = FakeSession()

    def boom(method: str, url: str, kw: dict[str, Any]) -> FakeResp:
        raise requests.ConnectionError("failed https://api.example/?api_key=supersecretvalue123")

    sess.routes.append(("POST", "api.tavily.com", boom))
    tool = WebSearchTool(provider="tavily", api_key="tvly-abc", session=sess)  # type: ignore[arg-type]
    out = tool.to_tool_definition().function(query="x")
    assert out.startswith("Error:")
    assert "ConnectionError" in out
    assert "supersecretvalue123" not in out


def test_web_search_tool_accepts_new_providers():
    tool = WebSearchTool(provider="exa_mcp")
    assert tool.provider == "exa_mcp"
    with pytest.raises(RuntimeError, match="EXA_API_KEY"):
        WebSearchTool(provider="exa")
