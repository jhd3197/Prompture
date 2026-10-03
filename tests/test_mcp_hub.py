"""Tests for the MCP hub: registry, secrets, pooling, prefixing, health, import and CLI.

No network and no real servers: sessions come from an in-process fake factory.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from click.testing import CliRunner

from prompture.agents import ToolRegistry
from prompture.mcp import hub as hub_mod
from prompture.mcp.hub import (
    InlineSecretError,
    MCPConfigError,
    MCPHub,
    MCPServerConfig,
    MCPSessionPool,
    MissingEnvError,
    check_server,
    env_refs,
    find_inline_secrets,
    get_preset,
    load_mcp_registry_by_name,
    load_mcp_registry_by_name_sync,
    looks_like_secret,
    prefixed_tool_name,
    resolve_env_refs,
    resolve_mcp_tools,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Tool:
    def __init__(self, name, description="d", schema=None):
        self.name = name
        self.description = description
        self.inputSchema = schema or {"type": "object", "properties": {"q": {"type": "string"}}}


class _Text:
    def __init__(self, text):
        self.text = text


class _Result:
    def __init__(self, text, is_error=False):
        self.content = [_Text(text)]
        self.isError = is_error


class FakeSession:
    def __init__(self, server, tools):
        self.server = server
        self.tools = tools
        self.calls = []
        self.loop = None

    async def list_tools(self, **_):
        class _L:
            pass

        listing = _L()
        listing.tools = self.tools
        listing.nextCursor = None
        return listing

    async def call_tool(self, name, arguments):
        self.loop = asyncio.get_running_loop()
        self.calls.append((name, arguments))
        if name == "slow":
            await asyncio.sleep(5)
        if name == "boom":
            raise RuntimeError("server exploded")
        if name == "bad":
            return _Result("nope", is_error=True)
        return _Result(f"{self.server}:{name}:{json.dumps(arguments, sort_keys=True)}")


class FakeFactory:
    """Session factory recording how many sessions were opened/closed per server."""

    def __init__(self, tools_by_server=None):
        self.tools_by_server = tools_by_server or {}
        self.opened: dict[str, int] = {}
        self.closed: dict[str, int] = {}
        self.sessions: dict[str, FakeSession] = {}

    @asynccontextmanager
    async def __call__(self, config):
        config.resolved()  # surface MissingEnvError like the real factory
        self.opened[config.name] = self.opened.get(config.name, 0) + 1
        tools = self.tools_by_server.get(config.name, [_Tool("search"), _Tool("fetch.page")])
        session = FakeSession(config.name, tools)
        self.sessions[config.name] = session
        try:
            yield session
        finally:
            self.closed[config.name] = self.closed.get(config.name, 0) + 1


@pytest.fixture
def registry_paths(tmp_path, monkeypatch):
    user = tmp_path / "home" / ".prompture" / "mcp.json"
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.setenv(hub_mod.USER_CONFIG_ENV, str(user))
    monkeypatch.chdir(project_dir)
    return user, project_dir / ".prompture" / "mcp.json"


@pytest.fixture
def pool():
    factory = FakeFactory()
    p = MCPSessionPool(session_factory=factory)
    p.factory = factory
    yield p
    p.shutdown()


def _hub_with(*configs, user_path):
    hub = MCPHub(user_path)
    for cfg in configs:
        hub.add(cfg)
    return hub


# ---------------------------------------------------------------------------
# ${ENV} references and secrets
# ---------------------------------------------------------------------------


class TestEnvRefs:
    def test_refs_found_in_nested(self):
        assert env_refs({"a": "${A}", "b": ["x${B}y", "${A}"]}) == ["A", "B"]

    def test_resolve(self):
        assert resolve_env_refs("Bearer ${TOK}", {"TOK": "abc"}) == "Bearer abc"

    def test_default(self):
        assert resolve_env_refs("${MISSING:-fallback}", {}) == "fallback"

    def test_missing_raises_with_names_only(self):
        with pytest.raises(MissingEnvError) as info:
            resolve_env_refs("${A} ${B}", {"A": "secretvalue"})
        assert info.value.missing == ["B"]
        assert "secretvalue" not in str(info.value)

    def test_config_resolved_copy_does_not_mutate(self, monkeypatch):
        monkeypatch.setenv("HUB_TEST_TOKEN", "t0ken-value")
        cfg = MCPServerConfig(
            "s", transport="http", url="https://x.example/mcp", headers={"Authorization": "Bearer ${HUB_TEST_TOKEN}"}
        )
        resolved = cfg.resolved()
        assert resolved.headers["Authorization"] == "Bearer t0ken-value"
        assert cfg.headers["Authorization"] == "Bearer ${HUB_TEST_TOKEN}"
        assert "t0ken-value" not in json.dumps(cfg.to_dict())

    def test_missing_env_listed(self, monkeypatch):
        monkeypatch.delenv("NOPE_VAR", raising=False)
        cfg = MCPServerConfig("s", command="npx", env={"K": "${NOPE_VAR}", "D": "${OTHER:-x}"})
        assert cfg.missing_env() == ["NOPE_VAR"]
        with pytest.raises(MissingEnvError):
            cfg.resolved()


class TestSecretDetection:
    @pytest.mark.parametrize(
        "key,value",
        [
            ("GITHUB_TOKEN", "abc123def456ghi"),
            ("Authorization", "Bearer abcdefghijklmnop"),
            ("X", "sk-abcdefghijklmnopqrstuvwx"),
            ("X", "ghp_abcdefghijklmnopqrstuvwxyz0123"),
            ("API_KEY", "a1b2c3d4e5"),
        ],
    )
    def test_secret(self, key, value):
        assert looks_like_secret(key, value)

    @pytest.mark.parametrize(
        "key,value",
        [
            ("GITHUB_TOKEN", "${GITHUB_TOKEN}"),
            ("Authorization", "Bearer ${TOKEN}"),
            ("SESSION_DIR", "/tmp/sessions"),
            ("AUTH_MODE", "oauth"),
            ("LOG_LEVEL", "debug"),
            ("TOKEN", ""),
        ],
    )
    def test_not_secret(self, key, value):
        assert not looks_like_secret(key, value)

    def test_find_inline_secrets_fields(self):
        cfg = MCPServerConfig(
            "s",
            command="npx",
            args=["--api-key", "abcdef123456", "--token=${T}", "plain"],
            env={"API_KEY": "abcdef123456", "OK": "${OK}"},
        )
        assert find_inline_secrets(cfg) == ["env.API_KEY", "args[1]"]

    def test_url_secret(self):
        cfg = MCPServerConfig("s", transport="http", url="https://x.example/mcp?apiKey=abcdef123456")
        assert find_inline_secrets(cfg) == ["url"]
        ref = MCPServerConfig("s", transport="http", url="https://x.example/mcp?apiKey=${KEY}")
        assert find_inline_secrets(ref) == []


# ---------------------------------------------------------------------------
# Config + registry
# ---------------------------------------------------------------------------


class TestConfig:
    def test_roundtrip(self):
        cfg = MCPServerConfig("gh", transport="http", url="https://x.example/mcp", headers={"A": "${B}"}, timeout=10)
        back = MCPServerConfig.from_dict("gh", cfg.to_dict())
        assert back == cfg

    def test_type_alias_and_inference(self):
        assert (
            MCPServerConfig.from_dict("a", {"type": "streamable-http", "url": "https://x.example"}).transport == "http"
        )
        assert MCPServerConfig.from_dict("a", {"url": "https://x.example"}).transport == "http"
        assert MCPServerConfig.from_dict("a", {"command": "npx"}).transport == "stdio"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": "bad name"},
            {"name": "a__b"},
            {"name": "x", "transport": "sse", "url": "https://x"},
            {"name": "x", "transport": "stdio"},
            {"name": "x", "transport": "http", "url": "ftp://x"},
            {"name": "x", "command": "npx", "timeout": 0},
            {"name": "x", "command": "npx", "env": {"1BAD": "v"}},
        ],
    )
    def test_invalid(self, kwargs):
        with pytest.raises(MCPConfigError):
            MCPServerConfig(**kwargs).validate()

    def test_url_with_ref_validates(self):
        MCPServerConfig("x", transport="http", url="https://${HOST}/mcp").validate()

    def test_presets_valid(self):
        for name in hub_mod.PRESETS:
            cfg = get_preset(name)
            cfg.validate()
            assert find_inline_secrets(cfg) == []
        assert get_preset("exa").url == "https://mcp.exa.ai/mcp"

    def test_preset_extra_args_replace_defaults(self):
        assert get_preset("filesystem").args[-1] == "."
        cfg = get_preset("filesystem", name="docs", extra_args=["/srv/docs"])
        assert cfg.name == "docs" and cfg.args[-1] == "/srv/docs" and "." not in cfg.args

    def test_unknown_preset(self):
        with pytest.raises(MCPConfigError):
            get_preset("nope")


class TestRegistry:
    def test_no_file_until_first_write(self, registry_paths):
        user, project = registry_paths
        hub = MCPHub()
        assert hub.list() == []
        assert not user.exists() and not user.parent.exists()
        hub.add(MCPServerConfig("a", command="npx"), dry_run=True)
        assert not user.exists()
        hub.add(MCPServerConfig("a", command="npx"))
        data = json.loads(user.read_text(encoding="utf-8"))
        assert data["version"] == 1 and data["servers"]["a"]["command"] == "npx"
        assert not project.exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
    def test_owner_only_permissions(self, registry_paths):
        user, _ = registry_paths
        MCPHub().add(MCPServerConfig("a", command="npx"))
        assert (user.stat().st_mode & 0o777) == 0o600

    def test_atomic_write_leaves_no_temp_files(self, registry_paths):
        user, _ = registry_paths
        hub = MCPHub()
        for i in range(3):
            hub.add(MCPServerConfig(f"s{i}", command="npx"))
        assert sorted(p.name for p in user.parent.iterdir()) == ["mcp.json"]

    def test_project_overrides_user(self, registry_paths):
        hub = MCPHub()
        hub.add(MCPServerConfig("s", command="user-cmd"))
        hub.add(MCPServerConfig("s", command="project-cmd"), scope="project")
        hub.add(MCPServerConfig("u", command="only-user"))
        fresh = MCPHub()
        assert fresh.get("s").command == "project-cmd"
        assert fresh.get("s").scope == "project"
        assert fresh.names() == ["s", "u"]
        fresh.remove("s")  # removes the project entry first
        assert MCPHub().get("s").command == "user-cmd"

    def test_duplicate_requires_replace(self, registry_paths):
        hub = MCPHub()
        hub.add(MCPServerConfig("s", command="a"))
        with pytest.raises(MCPConfigError, match="already exists"):
            hub.add(MCPServerConfig("s", command="b"))
        hub.add(MCPServerConfig("s", command="b"), replace=True)
        assert MCPHub().get("s").command == "b"

    def test_inline_secret_refused(self, registry_paths):
        user, _ = registry_paths
        hub = MCPHub()
        with pytest.raises(InlineSecretError) as info:
            hub.add(
                MCPServerConfig(
                    "s", transport="http", url="https://x.example", headers={"Authorization": "Bearer abcdefghijkl1234"}
                )
            )
        assert info.value.fields == ["headers.Authorization"]
        assert "abcdefghijkl1234" not in str(info.value)
        assert not user.exists()

    def test_bad_entries_reported_not_fatal(self, registry_paths):
        user, _ = registry_paths
        user.parent.mkdir(parents=True)
        user.write_text(json.dumps({"servers": {"ok": {"command": "npx"}, "bad": "nope"}}), encoding="utf-8")
        hub = MCPHub()
        assert hub.names() == ["ok"]
        assert any("bad" in e for e in hub.errors)

    def test_accepts_mcpservers_key(self, registry_paths):
        user, _ = registry_paths
        user.parent.mkdir(parents=True)
        user.write_text(json.dumps({"mcpServers": {"x": {"url": "https://x.example/mcp"}}}), encoding="utf-8")
        assert MCPHub().get("x").transport == "http"

    def test_require_falls_back_to_preset(self, registry_paths):
        hub = MCPHub()
        assert hub.get("exa") is None
        assert hub.require("exa").scope == "preset"
        with pytest.raises(MCPConfigError, match="Unknown MCP server"):
            hub.require("nothing-here")

    def test_set_enabled(self, registry_paths):
        hub = MCPHub()
        hub.add(MCPServerConfig("s", command="npx"))
        hub.set_enabled("s", False)
        assert MCPHub().get("s").enabled is False
        assert MCPHub().list(include_disabled=False) == []


# ---------------------------------------------------------------------------
# Tool naming
# ---------------------------------------------------------------------------


class TestPrefixing:
    def test_basic(self):
        assert prefixed_tool_name("search", "web_search") == "search__web_search"

    def test_sanitized(self):
        assert prefixed_tool_name("gh", "repo.list/all") == "gh__repo_list_all"

    def test_truncated_unique(self):
        taken: set[str] = set()
        a = prefixed_tool_name("server", "x" * 80 + "a", taken)
        b = prefixed_tool_name("server", "x" * 80 + "b", taken)
        assert len(a) <= 64 and len(b) <= 64 and a != b

    def test_collision_after_sanitizing(self):
        taken: set[str] = set()
        a = prefixed_tool_name("s", "a.b", taken)
        b = prefixed_tool_name("s", "a_b", taken)
        assert a == "s__a_b" and b != a
        assert hub_mod._TOOL_NAME_RE.match(b)


# ---------------------------------------------------------------------------
# Pool, resolution and dispatch
# ---------------------------------------------------------------------------


class TestResolveAndDispatch:
    def test_resolve_prefixes_and_metadata(self, registry_paths, pool):
        hub = _hub_with(MCPServerConfig("search", command="npx"), user_path=registry_paths[0])
        defs = resolve_mcp_tools("search", hub=hub, pool=pool)
        assert [d.name for d in defs] == ["search__search", "search__fetch_page"]
        assert defs[1].metadata == {"source": "mcp", "mcp_server": "search", "mcp_tool": "fetch.page"}
        assert defs[0].parameters["properties"] == {"q": {"type": "string"}}

    def test_lazy_connect_and_pooled_session(self, registry_paths, pool):
        hub = _hub_with(MCPServerConfig("search", command="npx"), user_path=registry_paths[0])
        assert pool.factory.opened == {}
        reg = ToolRegistry()
        for d in resolve_mcp_tools("search", hub=hub, pool=pool):
            reg.add(d)
        for d in resolve_mcp_tools("search", hub=hub, pool=pool):  # second resolve reuses session
            assert d.name in reg
        assert reg.execute("search__search", {"q": "a"}) == 'search:search:{"q": "a"}'
        assert reg.execute("search__fetch_page", {"q": "b"}) == 'search:fetch.page:{"q": "b"}'
        assert pool.factory.opened == {"search": 1}
        assert pool.is_connected("search")

    def test_sync_execute_inside_running_loop(self, registry_paths, pool):
        hub = _hub_with(MCPServerConfig("s", command="npx"), user_path=registry_paths[0])
        reg = load_mcp_registry_by_name_sync("s", hub=hub, pool=pool)

        async def main():
            # Sync dispatch from a thread that already runs a loop (sync Agent inside async code).
            return reg.execute("s__search", {"q": "x"})

        assert asyncio.run(main()) == 's:search:{"q": "x"}'

    def test_async_execute_runs_on_pool_loop(self, registry_paths, pool):
        hub = _hub_with(MCPServerConfig("s", command="npx"), user_path=registry_paths[0])

        async def main():
            reg = await load_mcp_registry_by_name("s", hub=hub, pool=pool)
            out = await reg.aexecute("s__search", {"q": "y"})
            return out, asyncio.get_running_loop()

        out, caller_loop = asyncio.run(main())
        assert out == 's:search:{"q": "y"}'
        assert pool.factory.sessions["s"].loop is not caller_loop

    def test_tool_filter(self, registry_paths, pool):
        hub = _hub_with(MCPServerConfig("s", command="npx"), user_path=registry_paths[0])
        defs = resolve_mcp_tools("s/fetch.page", hub=hub, pool=pool)
        assert [d.name for d in defs] == ["s__fetch_page"]
        assert [d.name for d in resolve_mcp_tools("s/s__search", hub=hub, pool=pool)] == ["s__search"]
        with pytest.raises(MCPConfigError, match="no tool"):
            resolve_mcp_tools("s/missing", hub=hub, pool=pool)

    def test_all_skips_disabled(self, registry_paths, pool):
        hub = _hub_with(
            MCPServerConfig("a", command="npx"),
            MCPServerConfig("b", command="npx"),
            MCPServerConfig("c", command="npx", enabled=False),
            user_path=registry_paths[0],
        )
        names = {d.metadata["mcp_server"] for d in resolve_mcp_tools("all", hub=hub, pool=pool)}
        assert names == {"a", "b"}
        with pytest.raises(MCPConfigError, match="disabled"):
            resolve_mcp_tools("c", hub=hub, pool=pool)

    def test_all_with_nothing_registered(self, registry_paths, pool):
        with pytest.raises(MCPConfigError, match="No enabled"):
            resolve_mcp_tools("all", hub=MCPHub(), pool=pool)

    def test_preset_by_name_without_registration(self, registry_paths, pool):
        defs = resolve_mcp_tools("exa", hub=MCPHub(), pool=pool)
        assert defs[0].name.startswith("exa__")
        assert not registry_paths[0].exists()

    def test_error_results_are_strings(self, registry_paths):
        factory = FakeFactory({"s": [_Tool("boom"), _Tool("bad"), _Tool("ok")]})
        p = MCPSessionPool(session_factory=factory)
        try:
            hub = _hub_with(MCPServerConfig("s", command="npx"), user_path=registry_paths[0])
            reg = load_mcp_registry_by_name_sync("s", hub=hub, pool=p)
            boom = reg.execute("s__boom", {})
            assert boom.startswith("Error calling MCP tool 's__boom'") and "server exploded" in boom
            assert reg.execute("s__bad", {}) == "Error from MCP tool 's__bad': nope"
            # Session survives a tool-level failure.
            assert reg.execute("s__ok", {}) == "s:ok:{}"
            assert factory.opened == {"s": 1}
        finally:
            p.shutdown()

    def test_call_timeout(self, registry_paths):
        factory = FakeFactory({"s": [_Tool("slow")]})
        p = MCPSessionPool(session_factory=factory)
        try:
            hub = _hub_with(MCPServerConfig("s", command="npx", timeout=0.2), user_path=registry_paths[0])
            reg = load_mcp_registry_by_name_sync("s", hub=hub, pool=p)
            start = time.monotonic()
            out = reg.execute("s__slow", {})
            assert "did not answer within 0.2s" in out
            assert time.monotonic() - start < 3
        finally:
            p.shutdown()

    def test_startup_timeout(self, registry_paths):
        @asynccontextmanager
        async def hanging(config):
            await asyncio.sleep(10)
            yield None

        p = MCPSessionPool(session_factory=hanging)
        try:
            cfg = MCPServerConfig("s", transport="http", url="https://x.example/mcp", timeout=0.2)
            with pytest.raises(hub_mod.MCPTimeoutError):
                p.list_tools(cfg)
        finally:
            p.shutdown()

    def test_connect_error_wrapped(self, registry_paths):
        @asynccontextmanager
        async def failing(config):
            raise OSError("spawn failed")
            yield  # pragma: no cover

        p = MCPSessionPool(session_factory=failing)
        try:
            with pytest.raises(hub_mod.MCPConnectionError, match="spawn failed"):
                p.list_tools(MCPServerConfig("s", command="npx"))
        finally:
            p.shutdown()

    def test_missing_env_surfaces(self, registry_paths, pool, monkeypatch):
        monkeypatch.delenv("HUB_NOT_SET", raising=False)
        cfg = MCPServerConfig("s", command="npx", env={"T": "${HUB_NOT_SET}"})
        with pytest.raises(MissingEnvError):
            pool.list_tools(cfg)

    def test_config_change_reconnects(self, registry_paths, pool):
        pool.list_tools(MCPServerConfig("s", command="npx"))
        pool.list_tools(MCPServerConfig("s", command="npx", args=["-y"]))
        assert pool.factory.opened == {"s": 2}
        assert pool.factory.closed == {"s": 1}

    def test_shutdown_closes_sessions(self, registry_paths):
        factory = FakeFactory()
        p = MCPSessionPool(session_factory=factory)
        p.list_tools(MCPServerConfig("a", command="npx"))
        p.list_tools(MCPServerConfig("b", command="npx"))
        p.shutdown()
        assert factory.closed == {"a": 1, "b": 1}

    def test_usage_recorded(self, registry_paths, pool):
        from prompture.infra.tracker import get_tracker

        events = []
        tracker = get_tracker()
        tracker.add_sink(events.append)
        try:
            hub = _hub_with(MCPServerConfig("s", command="npx"), user_path=registry_paths[0])
            reg = load_mcp_registry_by_name_sync("s", hub=hub, pool=pool)
            reg.execute("s__search", {"q": "z"})
        finally:
            tracker.remove_sink(events.append)
        mcp_events = [e for e in events if e.operation == "mcp_tool_call"]
        assert len(mcp_events) == 1
        ev = mcp_events[0]
        assert ev.model_name == "mcp/s" and ev.provider == "mcp" and ev.status == "success"
        assert ev.metadata["mcp_server"] == "s" and ev.metadata["mcp_tool"] == "search"
        assert ev.tool_name == "s__search" and ev.elapsed_ms >= 0


class TestNamedSpec:
    def test_mcp_namespace_resolves(self, registry_paths, pool, monkeypatch):
        from prompture.tools.named import resolve_tool_spec

        MCPHub().add(MCPServerConfig("search", command="npx"))
        monkeypatch.setattr(hub_mod, "_pool", pool)
        try:
            defs = resolve_tool_spec("mcp:search")
        finally:
            monkeypatch.setattr(hub_mod, "_pool", None)
        assert [d.name for d in defs] == ["search__search", "search__fetch_page"]

    def test_agent_accepts_mcp_spec(self, registry_paths, pool, monkeypatch):
        from prompture.agents import Agent

        MCPHub().add(MCPServerConfig("search", command="npx"))
        monkeypatch.setattr(hub_mod, "_pool", pool)
        try:
            agent = Agent("openai/gpt-4o-mini", tools=["mcp:search"])
        finally:
            monkeypatch.setattr(hub_mod, "_pool", None)
        assert "search__search" in agent._tools


# ---------------------------------------------------------------------------
# One-shot check + health
# ---------------------------------------------------------------------------


class TestCheckAndHealth:
    def test_check_ok(self):
        res = check_server(MCPServerConfig("s", command="npx"), session_factory=FakeFactory())
        assert res["status"] == "ok" and res["tool_count"] == 2 and res["tools"] == ["search", "fetch.page"]

    def test_check_timeout(self):
        @asynccontextmanager
        async def hanging(config):
            await asyncio.sleep(10)
            yield None

        res = check_server(MCPServerConfig("s", command="npx"), timeout=0.2, session_factory=hanging)
        assert res["status"] == "timeout"

    def test_check_unconfigured(self, monkeypatch):
        monkeypatch.delenv("HUB_NOT_SET", raising=False)
        res = check_server(
            MCPServerConfig("s", command="npx", env={"T": "${HUB_NOT_SET}"}), session_factory=FakeFactory()
        )
        assert res["status"] == "unconfigured" and "HUB_NOT_SET" in res["message"]

    def test_check_inside_running_loop(self):
        async def main():
            return check_server(MCPServerConfig("s", command="npx"), session_factory=FakeFactory())

        assert asyncio.run(main())["status"] == "ok"

    def test_health_registered(self):
        import prompture.mcp.health  # noqa: F401
        from prompture.capabilities.health import list_capabilities

        assert any(c.name == "mcp_servers" and c.category == "mcp" for c in list_capabilities("mcp"))

    def test_health_rows(self, registry_paths, monkeypatch):
        from prompture.mcp import health

        monkeypatch.delenv("HUB_NOT_SET", raising=False)
        monkeypatch.setattr(health, "_mcp_installed", lambda: True)
        hub = MCPHub()
        assert [r.status for r in health.check_mcp_servers(hub=hub)] == ["skipped"]
        hub.add(MCPServerConfig("web", transport="http", url="https://x.example/mcp"))
        hub.add(
            MCPServerConfig("needs", transport="http", url="https://x.example/mcp", headers={"A": "${HUB_NOT_SET}"})
        )
        hub.add(MCPServerConfig("off", command="npx", enabled=False))
        hub.add(MCPServerConfig("ghost", command="definitely-not-a-real-binary-xyz"))
        rows = {r.name: r for r in health.check_mcp_servers(hub=MCPHub())}
        assert rows["mcp:web"].status == "ok" and rows["mcp:web"].category == "mcp"
        assert rows["mcp:needs"].status == "unconfigured" and "HUB_NOT_SET" in rows["mcp:needs"].fix_hint
        assert rows["mcp:off"].status == "skipped"
        assert rows["mcp:ghost"].status == "missing"

    def test_health_launcher_probe(self, registry_paths, monkeypatch):
        from prompture.capabilities.probe import ProbeResult
        from prompture.mcp import health

        monkeypatch.setattr(health, "_mcp_installed", lambda: True)
        monkeypatch.setattr(health, "cached_probe", lambda cmd, **kw: ProbeResult("broken", cmd, hint="reinstall node"))
        row = health.check_server_health(MCPServerConfig("m", command="npx"))
        assert row.status == "broken" and row.fix_hint == "reinstall node"

    def test_health_missing_package(self):
        from prompture.mcp import health

        row = health.check_server_health(MCPServerConfig("m", command="npx"), mcp_available=False)
        assert row.status == "missing" and "prompture[mcp]" in row.fix_hint

    def test_health_live(self, monkeypatch):
        from prompture.mcp import health

        monkeypatch.setattr(
            health,
            "check_server",
            lambda cfg: {"status": "ok", "tool_count": 3, "tools": ["a", "b", "c"], "elapsed_ms": 5},
        )
        row = health.check_server_health(
            MCPServerConfig("w", transport="http", url="https://x.example/mcp"), live=True, mcp_available=True
        )
        assert row.status == "ok" and row.details["tool_count"] == 3
        monkeypatch.setattr(health, "check_server", lambda cfg: {"status": "timeout", "message": "slow", "tools": []})
        row = health.check_server_health(
            MCPServerConfig("w", transport="http", url="https://x.example/mcp"), live=True, mcp_available=True
        )
        assert row.status == "timeout"


# ---------------------------------------------------------------------------
# Editor import
# ---------------------------------------------------------------------------


class TestImport:
    def _write(self, path: Path, data) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_claude_desktop_style_converts_secrets(self, tmp_path, monkeypatch):
        from prompture.mcp.importers import import_from_editor

        monkeypatch.delenv("GITHUB_PERSONAL_ACCESS_TOKEN", raising=False)
        src = self._write(
            tmp_path / "cfg.json",
            {
                "mcpServers": {
                    "github": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-github"],
                        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_abcdefghijklmnopqrstuvwxyz0123", "LOG": "info"},
                    },
                    "remote api": {
                        "type": "http",
                        "url": "https://api.example.com/mcp?apiKey=abc123secret",
                        "headers": {"Authorization": "Bearer abcdefghijklmnop1234"},
                    },
                    "legacy": {"type": "sse", "url": "https://old.example.com/sse"},
                }
            },
        )
        result = import_from_editor("claude-desktop", path=src)
        by_name = {s.name: s for s in result.servers}
        assert set(by_name) == {"github", "remote-api"}
        gh = by_name["github"]
        assert gh.env == {"GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}", "LOG": "info"}
        remote = by_name["remote-api"]
        assert remote.headers["Authorization"] == "Bearer ${REMOTE_API_TOKEN}"
        assert remote.url == "https://api.example.com/mcp?apiKey=${REMOTE_API_APIKEY}"
        assert set(result.env_to_set) >= {"GITHUB_PERSONAL_ACCESS_TOKEN", "REMOTE_API_TOKEN", "REMOTE_API_APIKEY"}
        assert result.skipped and result.skipped[0][0] == "legacy"
        dumped = json.dumps([s.to_dict() for s in result.servers])
        for secret in ("ghp_abcdefghijklmnopqrstuvwxyz0123", "abc123secret", "abcdefghijklmnop1234"):
            assert secret not in dumped
        for cfg in result.servers:
            assert find_inline_secrets(cfg) == []

    def test_vscode_inputs_and_env_refs(self, tmp_path):
        from prompture.mcp.importers import import_from_editor

        src = self._write(
            tmp_path / ".vscode" / "mcp.json",
            {
                "inputs": [{"id": "api-key", "type": "promptString", "password": True}],
                "servers": {
                    "svc": {
                        "type": "stdio",
                        "command": "uvx",
                        "args": ["svc"],
                        "env": {"KEY": "${input:api-key}", "H": "${env:HOME_X}"},
                    }
                },
            },
        )
        result = import_from_editor("vscode", project_dir=tmp_path)
        assert result.source == src
        assert result.servers[0].env == {"KEY": "${API_KEY}", "H": "${HOME_X}"}

    def test_windsurf_server_url_and_cli_args(self, tmp_path):
        from prompture.mcp.importers import convert_editor_entry

        cfg, reason, env_vars, _ = convert_editor_entry("w", {"serverUrl": "https://w.example/mcp"})
        assert reason is None and cfg.transport == "http"
        cfg, _, env_vars, _converted = convert_editor_entry(
            "x", {"command": "tool", "args": ["--token", "abcdef123456", "--api-key=zyx987654321"]}
        )
        assert cfg.args == ["--token", "${X_TOKEN}", "--api-key=${X_API_KEY}"]
        assert env_vars == ["X_TOKEN", "X_API_KEY"]

    def test_claude_code_project_entries(self, tmp_path):
        from prompture.mcp.importers import import_from_editor

        src = self._write(
            tmp_path / ".claude.json",
            {
                "mcpServers": {"a": {"command": "npx"}},
                "projects": {str(tmp_path): {"mcpServers": {"b": {"command": "uvx"}}}},
            },
        )
        result = import_from_editor("claude-code", path=src, project_dir=tmp_path)
        assert sorted(s.name for s in result.servers) == ["a", "b"]

    def test_missing_source(self, tmp_path):
        from prompture.mcp.importers import import_from_editor

        result = import_from_editor("cursor", path=tmp_path / "nope.json")
        assert result.servers == []

    def test_unknown_editor(self):
        from prompture.mcp.importers import editor_config_paths

        with pytest.raises(MCPConfigError):
            editor_config_paths("emacs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    @pytest.fixture
    def run(self, registry_paths):
        from prompture.cli.mcp_cmd import mcp

        runner = CliRunner()

        def _run(*args, **kw):
            return runner.invoke(mcp, list(args), catch_exceptions=False, **kw)

        return _run

    def test_commands_export(self):
        from prompture.cli import mcp_cmd

        assert [c.name for c in mcp_cmd.COMMANDS] == ["mcp"]
        assert {"add", "list", "remove", "test", "import", "enable", "disable"} <= set(mcp_cmd.mcp.commands)

    def test_add_preset_dry_run_then_real(self, run, registry_paths):
        user, _ = registry_paths
        res = run("add", "--preset", "exa", "--dry-run")
        assert res.exit_code == 0 and "[dry-run]" in res.output and not user.exists()
        res = run("add", "--preset", "exa")
        assert res.exit_code == 0, res.output
        assert json.loads(user.read_text(encoding="utf-8"))["servers"]["exa"]["url"] == "https://mcp.exa.ai/mcp"

    def test_add_custom_and_list_json(self, run):
        res = run(
            "add",
            "gh",
            "--url",
            "https://x.example/mcp",
            "--header",
            "Authorization=Bearer ${HUB_GH_TOKEN}",
            "--timeout",
            "15",
        )
        assert res.exit_code == 0, res.output
        assert "HUB_GH_TOKEN" in res.output
        res = run("add", "local", "--command", "python", "--arg", "-m", "--arg", "srv", "--project")
        assert res.exit_code == 0, res.output
        data = json.loads(run("list", "--json").output)
        rows = {r["name"]: r for r in data["servers"]}
        assert rows["gh"]["transport"] == "http" and rows["gh"]["timeout"] == 15.0
        assert rows["local"]["scope"] == "project" and rows["local"]["target"] == "python -m srv"

    def test_add_refuses_inline_secret(self, run, registry_paths):
        res = run("add", "gh", "--url", "https://x.example/mcp", "--header", "Authorization=Bearer abcdefghijkl12345")
        assert res.exit_code == 2
        assert "abcdefghijkl12345" not in res.output
        assert not registry_paths[0].exists()

    def test_add_requires_target(self, run):
        assert run("add", "x").exit_code == 2

    def test_remove_and_toggle(self, run):
        run("add", "a", "--command", "npx")
        assert run("disable", "a").exit_code == 0
        assert MCPHub().get("a").enabled is False
        assert run("enable", "a").exit_code == 0
        assert run("remove", "a", "--dry-run").exit_code == 0
        assert MCPHub().get("a") is not None
        assert run("remove", "a").exit_code == 0
        assert MCPHub().get("a") is None
        assert run("remove", "a").exit_code == 2

    def test_list_presets(self, run):
        res = run("list", "--presets", "--json")
        names = {r["name"] for r in json.loads(res.output)}
        assert {"exa", "fetch", "memory", "filesystem"} <= names

    def test_test_command(self, run, monkeypatch):
        from prompture.cli import mcp_cmd

        run("add", "a", "--command", "npx")
        monkeypatch.setattr(
            mcp_cmd,
            "check_server",
            lambda cfg, timeout=None: {
                "name": cfg.name,
                "status": "ok",
                "tools": ["t1"],
                "tool_count": 1,
                "message": "1 tool(s)",
                "elapsed_ms": 3,
            },
        )
        res = run("test", "a", "--json")
        assert res.exit_code == 0 and json.loads(res.output)["results"][0]["tools"] == ["t1"]
        monkeypatch.setattr(
            mcp_cmd,
            "check_server",
            lambda cfg, timeout=None: {
                "name": cfg.name,
                "status": "error",
                "tools": [],
                "message": "boom",
                "elapsed_ms": 3,
            },
        )
        assert run("test").exit_code == 1
        assert run("test", "unknown-server").exit_code == 2

    def test_import_requires_confirmation(self, run, tmp_path, registry_paths):
        src = tmp_path / "desk.json"
        src.write_text(json.dumps({"mcpServers": {"m": {"command": "npx", "args": ["-y", "pkg"]}}}), encoding="utf-8")
        res = run("import", "--from", "claude-desktop", "--path", str(src), input="n\n")
        assert res.exit_code == 1 and "nothing was read" in res.output
        assert not registry_paths[0].exists()
        res = run("import", "--from", "claude-desktop", "--path", str(src), "--yes", "--dry-run")
        assert res.exit_code == 0 and "[dry-run] would import m" in res.output
        assert not registry_paths[0].exists()
        res = run("import", "--from", "claude-desktop", "--path", str(src), input="y\n")
        assert res.exit_code == 0, res.output
        assert MCPHub().get("m").description == "imported from Claude Desktop"
        assert run("import", "--from", "claude-desktop", "--path", str(src), "--yes").exit_code == 1  # exists

    def test_import_reports_env_without_values(self, run, tmp_path):
        src = tmp_path / "cursor.json"
        src.write_text(
            json.dumps({"mcpServers": {"k": {"command": "npx", "env": {"SERVICE_API_KEY": "zz-secret-value-123456"}}}}),
            encoding="utf-8",
        )
        res = run("import", "--from", "cursor", "--path", str(src), "--yes")
        assert res.exit_code == 0, res.output
        assert "SERVICE_API_KEY" in res.output and "zz-secret-value-123456" not in res.output
        stored = MCPHub().user_path.read_text(encoding="utf-8")
        assert "zz-secret-value-123456" not in stored and "${SERVICE_API_KEY}" in stored


# ---------------------------------------------------------------------------
# Optional live stdio round trip (needs the `mcp` package; no network)
# ---------------------------------------------------------------------------

_SERVER_SCRIPT = """
try:  # mcp 2.x
    from mcp.server.mcpserver import MCPServer as App
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as App

app = App("hub-test")


@app.tool()
def add(a: int, b: int) -> str:
    \"\"\"Add two integers.\"\"\"
    return str(a + b)


app.run()
"""


@pytest.mark.integration
def test_stdio_roundtrip(tmp_path, registry_paths):
    pytest.importorskip("mcp")
    script = tmp_path / "srv.py"
    script.write_text(_SERVER_SCRIPT, encoding="utf-8")
    # The SDK starts stdio servers with a minimal environment; keep the import path.
    env = {"PYTHONPATH": os.environ["PYTHONPATH"]} if os.environ.get("PYTHONPATH") else {}
    cfg = MCPServerConfig("calc", command=sys.executable, args=[str(script)], env=env, timeout=30)
    MCPHub().add(cfg)
    p = MCPSessionPool()
    try:
        reg = load_mcp_registry_by_name_sync("calc", hub=MCPHub(), pool=p)
        assert "calc__add" in reg
        assert reg.execute("calc__add", {"a": 2, "b": 3}) == "5"
        assert check_server(cfg)["tools"] == ["add"]
    finally:
        p.shutdown()
