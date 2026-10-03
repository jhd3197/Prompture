"""Tests for domain tool packs (prompture.tools.packs)."""

from __future__ import annotations

import asyncio
import json
import types

import pytest

from prompture.agents.tools_schema import ToolDefinition, ToolRegistry
from prompture.capabilities.health import list_capabilities
from prompture.tools import packs
from prompture.tools.named import expand_tool_specs, resolve_tool_spec
from prompture.tools.packs import (
    PackTool,
    ToolPack,
    format_tool_result,
    get_pack,
    list_packs,
    register_pack,
    resolve_pack_tools,
    stringify_tool,
    unregister_pack,
)

KEYS = ("FINNHUB_API_KEY", "NEWSAPI_API_KEY", "GOOGLE_MAPS_API_KEY", "OPENCAGE_API_KEY")


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_web(monkeypatch):
    """A stand-in for prompture.tools.web with the documented contract."""
    calls: list[tuple] = []

    class ReadResult:
        def __init__(self, url):
            self.url = url
            self.title = f"Title of {url}"
            self.content = "body text"

    class SearchResult:
        def __init__(self, i):
            self.title = f"hit {i}"
            self.url = f"https://example.com/{i}"
            self.snippet = "snippet   with   spaces"

    def read_url(url, **kw):
        calls.append(("read", url))
        return ReadResult(url)

    def search_platform(platform, query, max_results=10, **kw):
        calls.append(("search", platform, query, max_results, kw.get("kind")))
        return [SearchResult(i) for i in range(min(max_results, 2))]

    module = types.ModuleType("prompture.tools.web")
    module.read_url = read_url
    module.search_platform = search_platform
    monkeypatch.setitem(__import__("sys").modules, "prompture.tools.web", module)
    return calls


def _names(tds: list[ToolDefinition]) -> list[str]:
    return [t.name for t in tds]


def _tool(tds, name):
    return next(t for t in tds if t.name == name)


# ---------------------------------------------------------------------------
# registry + health
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_builtin_packs_registered(self):
        assert {"finance", "news", "dev", "places"} <= {p.name for p in list_packs()}
        rows = {c.name for c in list_capabilities(category="tools")}
        assert {"pack:finance", "pack:news", "pack:dev", "pack:places"} <= rows

    def test_unknown_pack(self):
        with pytest.raises(ValueError, match="Unknown tool pack"):
            get_pack("weather")
        with pytest.raises(ValueError):
            resolve_tool_spec("pack:weather")

    def test_custom_pack_lifecycle(self, monkeypatch):
        monkeypatch.setenv("CUSTOM_PACK_KEY", "k")
        monkeypatch.delenv("CUSTOM_MISSING_KEY", raising=False)

        def ping() -> str:
            """Ping."""
            return "pong"

        pack = ToolPack(
            "custom-test",
            "test pack",
            [
                PackTool("ping", packs.function_tool(ping), requires_env=("CUSTOM_PACK_KEY",)),
                PackTool("locked", packs.function_tool(ping), requires_env=("CUSTOM_MISSING_KEY",)),
                PackTool("absent", packs.function_tool(ping), requires_modules=("no_such_module_xyz",)),
            ],
        )
        register_pack(pack)
        try:
            assert _names(resolve_pack_tools("custom-test")) == ["ping"]
            row = next(r for c in list_capabilities("tools") if c.name == "pack:custom-test" for r in c.run())
            assert row.status == "degraded"
            assert row.message.startswith("1/3 tools live")
            assert "Set CUSTOM_MISSING_KEY (locked)" in row.fix_hint
            assert "install no_such_module_xyz (absent)" in row.fix_hint
            by_name = {t["name"]: t for t in row.details["tools"]}
            assert by_name["locked"]["missing_env"] == ["CUSTOM_MISSING_KEY"]
            assert by_name["absent"]["status"] == "missing"
        finally:
            unregister_pack("custom-test")
        assert "pack:custom-test" not in {c.name for c in list_capabilities("tools")}

    def test_status_unconfigured_and_missing(self):
        unconf = ToolPack("u", "", [PackTool("a", lambda: None, requires_env=("NOPE_KEY_1",))])
        assert unconf.check().status == "unconfigured"
        missing = ToolPack("m", "", [PackTool("a", lambda: None, requires_modules=("no_such_module_xyz",))])
        assert missing.check().status == "missing"
        ok = ToolPack("o", "", [PackTool("a", lambda: None)])
        assert ok.check().status == "ok"

    def test_broken_builder_is_skipped(self, caplog):
        def boom():
            raise ImportError("plugin vanished")

        pack = ToolPack("b", "", [PackTool("x", boom)])
        assert pack.resolve() == []
        assert "could not load x" in caplog.text


# ---------------------------------------------------------------------------
# finance / news / places (tukuy-backed)
# ---------------------------------------------------------------------------


class TestTukuyPacks:
    @pytest.fixture(autouse=True)
    def _need_tukuy(self):
        pytest.importorskip("tukuy.plugins.finnhub")
        pytest.importorskip("httpx")

    def test_finance_without_key_excludes_finnhub(self):
        tds = resolve_tool_spec("pack:finance")
        assert _names(tds) == ["crypto_price", "crypto_search", "crypto_trending"]
        row = get_pack("finance").check()
        assert row.status == "degraded"
        assert row.fix_hint == "Set FINNHUB_API_KEY (stock_quote, stock_news, stock_search)"

    def test_finance_with_key(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        assert _names(resolve_pack_tools("finance"))[:3] == ["stock_quote", "stock_news", "stock_search"]
        assert get_pack("finance").check().status == "ok"

    def test_news_and_places_keys(self, monkeypatch, fake_web):
        assert _names(resolve_pack_tools("news")) == ["read_feed"]
        assert _names(resolve_pack_tools("places")) == ["country_info", "country_search"]
        monkeypatch.setenv("NEWSAPI_API_KEY", "k")
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "k")
        assert _names(resolve_pack_tools("news")) == ["news_headlines", "news_search", "read_feed"]
        assert "maps_geocode" in _names(resolve_pack_tools("places"))

    def test_tukuy_tool_returns_string(self, monkeypatch):
        import httpx

        real = httpx.AsyncClient

        def handler(request):
            assert request.url.host == "api.coingecko.com"
            return httpx.Response(200, json={"bitcoin": {"usd": 100.0, "usd_24h_change": 1.5}})

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, transport=httpx.MockTransport(handler), **k))
        td = _tool(resolve_pack_tools("finance"), "crypto_price")
        out = td.function(coins="bitcoin")
        assert isinstance(out, str)
        data = json.loads(out)
        assert data["prices"]["bitcoin"]["usd"]["price"] == 100.0
        assert "success" not in data
        # async path (AsyncAgent) goes through _async_fn
        assert json.loads(asyncio.run(td.function._async_fn(coins="bitcoin"))) == data

    def test_tukuy_failure_is_error_string_without_secret(self, monkeypatch):
        import httpx

        monkeypatch.setenv("FINNHUB_API_KEY", "abcdef0123456789secret")
        real = httpx.AsyncClient

        def handler(request):
            raise httpx.ConnectError(f"cannot reach {request.url}")  # URL carries ?token=<key>

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, transport=httpx.MockTransport(handler), **k))
        out = _tool(resolve_pack_tools("finance"), "stock_quote").function(symbol="AAPL")
        assert out.startswith("Error:")
        assert "abcdef0123456789secret" not in out

    def test_sync_call_inside_running_loop(self, monkeypatch):
        import httpx

        real = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda *a, **k: real(*a, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})), **k),
        )
        td = _tool(resolve_pack_tools("finance"), "crypto_trending")

        async def main():
            return td.function()

        assert isinstance(asyncio.run(main()), str)


# ---------------------------------------------------------------------------
# result shaping
# ---------------------------------------------------------------------------


class TestStringify:
    def _td(self, fn):
        return ToolDefinition("t", "d", {"type": "object", "properties": {}}, fn)

    def test_failure_dict(self):
        assert format_tool_result({"success": False, "error": "nope"}) == "Error: nope"
        assert format_tool_result({"error": "bad"}) == "Error: bad"

    def test_success_dict_and_truncation(self):
        assert json.loads(format_tool_result({"success": True, "a": 1})) == {"a": 1}
        out = format_tool_result("x" * 100, max_chars=10)
        assert out.startswith("x" * 10) and "truncated" in out

    def test_raising_function_never_raises(self):
        def boom(**kw):
            raise RuntimeError("failed calling https://api.example.com/q?api_key=supersecretvalue1")

        out = stringify_tool(self._td(boom)).function()
        assert out.startswith("Error: RuntimeError") and "supersecretvalue1" not in out

    def test_async_function(self):
        async def fn(**kw):
            return {"success": True, "v": kw["x"]}

        td = stringify_tool(self._td(fn))
        assert json.loads(td.function(x=2)) == {"v": 2}
        assert json.loads(asyncio.run(td.function._async_fn(x=3))) == {"v": 3}


# ---------------------------------------------------------------------------
# dev pack
# ---------------------------------------------------------------------------


class TestDevPack:
    def test_resolves_through_named_spec(self, fake_web):
        tds = resolve_tool_spec("pack:dev")
        assert _names(tds) == [
            "github_search",
            "github_read",
            "hackernews_search",
            "hackernews_read",
            "arxiv_search",
            "arxiv_read",
            "pypi_package",
            "npm_package",
        ]
        assert all(isinstance(t, ToolDefinition) for t in expand_tool_specs(["pack:dev"]))
        assert get_pack("dev").check().status == "ok"

    def test_without_web_readers_only_registries(self, monkeypatch):
        monkeypatch.setattr(packs, "_module_available", lambda name: not name.startswith("prompture.tools.web"))
        assert _names(resolve_pack_tools("dev")) == ["pypi_package", "npm_package"]
        row = get_pack("dev").check()
        assert row.status == "degraded" and "prompture.tools.web" in row.fix_hint

    def test_web_tools(self, fake_web):
        tds = resolve_pack_tools("dev")
        out = _tool(tds, "github_search").function(query="llm agents", kind="code", max_results=50)
        assert out.startswith("1. hit 0\n   https://example.com/0\n   snippet with spaces")
        assert fake_web[-1] == ("search", "github", "llm agents", 25, "code")
        _tool(tds, "github_search").function(query="x")
        assert fake_web[-1][-1] is None  # default kind not forwarded

        assert (
            _tool(tds, "github_read")
            .function(target="owner/repo")
            .startswith("# Title of https://github.com/owner/repo")
        )
        assert _tool(tds, "github_read").function(target="https://evil.example/x").startswith("Error:")
        _tool(tds, "hackernews_read").function(item="123")
        assert fake_web[-1] == ("read", "https://news.ycombinator.com/item?id=123")
        _tool(tds, "arxiv_read").function(paper="2401.01234v2")
        assert fake_web[-1] == ("read", "https://arxiv.org/abs/2401.01234v2")
        assert _tool(tds, "arxiv_read").function(paper="not a paper").startswith("Error:")
        assert _tool(tds, "hackernews_search").function(query="  ") == "Error: query must not be empty"

    def test_registry_validates_and_executes(self, fake_web):
        registry = ToolRegistry()
        for td in resolve_pack_tools("dev"):
            registry.add(td)
        assert registry.execute("arxiv_search", {"query": "rlhf", "max_results": 1}).startswith("1. hit 0")
        assert "kind" in registry.get("github_search").parameters["properties"]

    def test_pypi(self, monkeypatch):
        captured = {}

        def fake_get_json(url, *, max_bytes):
            captured["url"] = url
            return {
                "info": {
                    "name": "requests",
                    "version": "2.32.0",
                    "summary": "HTTP",
                    "requires_python": ">=3.8",
                    "requires_dist": ["idna"],
                    "project_urls": {"Source": "https://github.com/psf/requests"},
                },
                "urls": [{"upload_time_iso_8601": "2024-05-20T00:00:00Z"}],
                "releases": {"1": [], "2": []},
            }

        monkeypatch.setattr(packs, "_get_json", fake_get_json)
        data = json.loads(packs.pypi_package("requests"))
        assert captured["url"] == "https://pypi.org/pypi/requests/json"
        assert data["version"] == "2.32.0" and data["release_count"] == 2 and data["dependencies"] == ["idna"]
        packs.pypi_package("requests", "2.0.0")
        assert captured["url"] == "https://pypi.org/pypi/requests/2.0.0/json"
        assert packs.pypi_package("../etc").startswith("Error:")

    def test_pypi_not_found(self, monkeypatch):
        def missing(url, *, max_bytes):
            raise RuntimeError("HTTP 404 for https://pypi.org/pypi/nope/json")

        monkeypatch.setattr(packs, "_get_json", missing)
        td = packs.function_tool(packs.pypi_package)()
        assert td.function(name="nope") == "Error: package 'nope' not found on PyPI"

    def test_npm(self, monkeypatch):
        captured = {}

        def fake_get_json(url, *, max_bytes):
            captured["url"] = url
            return {
                "name": "@types/node",
                "version": "22.0.0",
                "license": "MIT",
                "repository": {"url": "git+https://github.com/x/y.git"},
                "dependencies": {"undici-types": "~6"},
            }

        monkeypatch.setattr(packs, "_get_json", fake_get_json)
        data = json.loads(packs.npm_package("@types/node"))
        assert captured["url"] == "https://registry.npmjs.org/@types%2Fnode/latest"
        assert data["repository"] == "git+https://github.com/x/y.git" and data["license"] == "MIT"
        assert packs.npm_package("Bad Name").startswith("Error:")
        assert packs.npm_package("react", "1.0.0; rm").startswith("Error:")

    def test_unexpected_exception_becomes_error(self, monkeypatch):
        def explode(url, *, max_bytes):
            raise ConnectionError("down")

        monkeypatch.setattr(packs, "_get_json", explode)
        td = packs.function_tool(packs.npm_package)()
        assert td.function(name="react") == "Error: ConnectionError: down"


# ---------------------------------------------------------------------------
# news feed + pack:all
# ---------------------------------------------------------------------------


class TestNewsAndAll:
    def test_read_feed(self, fake_web):
        td = _tool(resolve_pack_tools("news"), "read_feed")
        assert td.function(url="https://blog.example.com/feed.xml").startswith(
            "# Title of https://blog.example.com/feed.xml"
        )
        assert td.function(url="file:///etc/passwd").startswith("Error:")

    def test_pack_all_dedupes(self, fake_web):
        names = _names(resolve_tool_spec("pack:all"))
        assert len(names) == len(set(names))
        assert "pypi_package" in names and "read_feed" in names
