"""Domain tool packs: curated, health-checked tool bundles mountable by name.

``Agent(..., tools=["pack:finance"])`` resolves through
:func:`resolve_pack_tools`. Each pack lists its tools with what they need
(an env var, an importable plugin); only tools whose requirements are met are
resolved, and the pack's doctor row (``pack:<name>``, category ``tools``)
names the exact env var or package that would enable the rest.

Shipped packs:

========== ===================================================================
``finance`` stock quotes, company news, symbol search (Finnhub,
            ``FINNHUB_API_KEY``); crypto prices, search, trending (CoinGecko,
            keyless)
``news``    top headlines and topic search (NewsAPI, ``NEWSAPI_API_KEY``);
            RSS/Atom feed reader (web readers)
``dev``     GitHub search/reader, Hacker News, arXiv (web readers); PyPI and
            npm package lookups (public JSON APIs)
``places``  geocoding, reverse geocoding, place search (Google Maps,
            ``GOOGLE_MAPS_API_KEY``); OpenCage geocoding
            (``OPENCAGE_API_KEY``); country data (REST Countries, keyless)
========== ===================================================================

Tukuy-backed tools go through :mod:`prompture.extraction.tukuy_bridge`. Every
pack tool returns a string and never raises; errors come back as
``"Error: ..."`` with credentials scrubbed.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import importlib
import importlib.util
import inspect
import json
import logging
import os
import re
import threading
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..agents.tools_schema import ToolDefinition, tool_from_function
from ..capabilities.health import HealthStatus, register_capability, unregister_capability
from ..security.redaction import scrub_secrets

logger = logging.getLogger("prompture.tools.packs")

DEFAULT_MAX_CHARS = 12_000

_INSTALL_HINTS = {
    "tukuy": "pip install -U tukuy",
    "httpx": "pip install httpx",
    "prompture.tools.web": "upgrade Prompture to a version that ships the web readers (prompture.tools.web)",
}


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------


def _error(exc: BaseException | str) -> str:
    text = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return "Error: " + scrub_secrets(str(text))[:1000]


def _clip(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    text = scrub_secrets(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n[truncated at {max_chars} characters]"


def format_tool_result(value: Any, *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Turn a plugin result into the string a model sees.

    ``{"success": False, "error": ...}`` dicts become ``"Error: ..."``; other
    dicts/lists are serialised as compact JSON without the ``success`` flag.
    """
    if value is None:
        return "No result."
    if isinstance(value, str):
        return _clip(value, max_chars)
    if isinstance(value, dict):
        if value.get("success") is False or (value.get("error") and value.get("success") is not True):
            return _error(str(value.get("error") or "the tool reported a failure"))
        value = {k: v for k, v in value.items() if k != "success"}
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return _clip(text, max_chars)


def _run_sync(coro: Any) -> Any:
    """Run *coro* to completion from sync code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def never_raises(fn: Callable[..., str]) -> Callable[..., str]:
    """Decorator: exceptions become scrubbed ``"Error: ..."`` strings."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.debug("pack tool %s failed", fn.__name__, exc_info=True)
            return _error(exc)

    return wrapper


def stringify_tool(
    td: ToolDefinition, *, description: str | None = None, max_chars: int = DEFAULT_MAX_CHARS
) -> ToolDefinition:
    """Wrap *td* so it returns a string, never raises, and works sync and async."""
    inner = td.function
    inner_async = getattr(inner, "_async_fn", None)

    async def _acall(**kwargs: Any) -> str:
        try:
            value = await inner_async(**kwargs) if inner_async is not None else inner(**kwargs)
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            logger.debug("pack tool %s failed", td.name, exc_info=True)
            return _error(exc)
        return format_tool_result(value, max_chars=max_chars)

    def _call(**kwargs: Any) -> str:
        try:
            return _run_sync(_acall(**kwargs))
        except Exception as exc:
            return _error(exc)

    _call.__name__ = td.name
    _call._async_fn = _acall  # type: ignore[attr-defined]
    skill = getattr(inner, "__skill__", None)
    if skill is not None:
        _call.__skill__ = skill  # type: ignore[attr-defined]
    return ToolDefinition(
        name=td.name,
        description=(description or td.description)[:1024],
        parameters=td.parameters,
        function=_call,
        metadata={**td.metadata, "source": "pack"},
    )


def tukuy_tool(plugin: str, skill: str, *, description: str | None = None) -> Callable[[], ToolDefinition]:
    """Builder for a pack tool backed by ``tukuy.plugins.<plugin>.<skill>``."""

    def build() -> ToolDefinition:
        from ..extraction.tukuy_bridge import skill_to_tool_definition

        module = importlib.import_module(f"tukuy.plugins.{plugin}")
        td = skill_to_tool_definition(getattr(module, skill))
        wrapped = stringify_tool(td, description=description)
        wrapped.metadata.update(plugin=f"tukuy.{plugin}", skill=skill)
        return wrapped

    return build


def function_tool(fn: Callable[..., str]) -> Callable[[], ToolDefinition]:
    """Builder for a pack tool backed by a plain typed function."""

    def build() -> ToolDefinition:
        td = tool_from_function(never_raises(fn))
        td.metadata["source"] = "pack"
        return td

    return build


# ---------------------------------------------------------------------------
# Pack model
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    """``"pkg.mod"`` → importable; ``"pkg.mod:attr"`` → importable and exposes *attr*."""
    module, _, attr = name.partition(":")
    try:
        if not attr:
            return importlib.util.find_spec(module) is not None
        return hasattr(importlib.import_module(module), attr)
    except Exception:  # broken optional packages must not break health checks
        return False


@dataclass
class PackTool:
    """One tool in a :class:`ToolPack` and what it needs to be live.

    Attributes:
        name: Tool name the model sees.
        build: Zero-arg callable returning the :class:`ToolDefinition`.
        description: Short summary for health output.
        requires_env: Env vars that must all be set.
        requires_modules: Importable modules that must all be present.
        optional_env: Env vars that improve the tool (shown, never required).
        source: Where it comes from (``"tukuy:finnhub"``, ``"web"``, ``"http"``).
    """

    name: str
    build: Callable[[], ToolDefinition]
    description: str = ""
    requires_env: tuple[str, ...] = ()
    requires_modules: tuple[str, ...] = ()
    optional_env: tuple[str, ...] = ()
    source: str = ""

    def availability(self) -> dict[str, Any]:
        """``{"name", "live", "status", "fix_hint", "requires_env", "missing_env", ...}``."""
        row: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "requires_env": list(self.requires_env),
            "optional_env": list(self.optional_env),
        }
        missing_mods = [m for m in self.requires_modules if not _module_available(m)]
        if missing_mods:
            hints = sorted(
                {
                    _INSTALL_HINTS.get(m.split(":")[0]) or _INSTALL_HINTS.get(m.split(".")[0]) or f"install {m}"
                    for m in missing_mods
                }
            )
            return {
                **row,
                "live": False,
                "status": "missing",
                "missing_modules": missing_mods,
                "fix_hint": "; ".join(hints),
            }
        missing_env = [e for e in self.requires_env if not os.environ.get(e)]
        if missing_env:
            return {
                **row,
                "live": False,
                "status": "unconfigured",
                "missing_env": missing_env,
                "fix_hint": "Set " + " and ".join(missing_env),
            }
        return {**row, "live": True, "status": "ok", "fix_hint": None}

    @property
    def live(self) -> bool:
        return bool(self.availability()["live"])


@dataclass
class ToolPack:
    """A named bundle of :class:`PackTool` entries."""

    name: str
    description: str
    tools: list[PackTool] = field(default_factory=list)

    def tool_status(self) -> list[dict[str, Any]]:
        return [t.availability() for t in self.tools]

    def live_tools(self) -> list[PackTool]:
        return [t for t in self.tools if t.live]

    def resolve(self) -> list[ToolDefinition]:
        """Build the live tools (tools with missing requirements are left out)."""
        out: list[ToolDefinition] = []
        for tool in self.live_tools():
            try:
                out.append(tool.build())
            except Exception as exc:
                logger.warning("pack:%s: could not load %s: %s", self.name, tool.name, scrub_secrets(str(exc)))
        return out

    def check(self, live: bool = False) -> HealthStatus:
        """Doctor row ``pack:<name>``. Offline: inspects env vars and importability only."""
        rows = self.tool_status()
        live_names = [r["name"] for r in rows if r["live"]]
        total = len(rows)
        if total and len(live_names) == total:
            status = "ok"
        elif live_names:
            status = "degraded"
        elif rows and all(r["status"] == "unconfigured" for r in rows):
            status = "unconfigured"
        else:
            status = "missing"
        # Group fixes so the hint reads "Set FINNHUB_API_KEY (stock_quote, stock_news)".
        fixes: dict[str, list[str]] = {}
        for r in rows:
            if not r["live"] and r.get("fix_hint"):
                fixes.setdefault(r["fix_hint"], []).append(r["name"])
        fix_hint = "; ".join(f"{hint} ({', '.join(names)})" for hint, names in fixes.items()) or None
        message = f"{len(live_names)}/{total} tools live" + (f": {', '.join(live_names)}" if live_names else "")
        return HealthStatus(
            f"pack:{self.name}",
            status,  # type: ignore[arg-type]
            category="tools",
            active_backend=None,
            message=message,
            fix_hint=fix_hint,
            details={"description": self.description, "live": live_names, "tools": rows},
        )


_packs: dict[str, ToolPack] = {}
_lock = threading.Lock()


def register_pack(pack: ToolPack, *, replace: bool = True) -> ToolPack:
    """Register *pack* (and its ``pack:<name>`` doctor row)."""
    key = pack.name.lower()
    with _lock:
        if not replace and key in _packs:
            return _packs[key]
        _packs[key] = pack
    register_capability(f"pack:{key}", "tools", pack.check, description=pack.description)
    return pack


def unregister_pack(name: str) -> None:
    with _lock:
        _packs.pop(name.lower(), None)
    unregister_capability(f"pack:{name.lower()}")


def get_pack(name: str) -> ToolPack:
    """Return the pack called *name*.

    Raises:
        ValueError: No such pack.
    """
    with _lock:
        pack = _packs.get((name or "").strip().lower())
        known = sorted(_packs)
    if pack is None:
        raise ValueError(f"Unknown tool pack {name!r}. Known: {', '.join(known)}, all")
    return pack


def list_packs() -> list[ToolPack]:
    with _lock:
        return list(_packs.values())


def resolve_pack_tools(name: str) -> list[ToolDefinition]:
    """Resolve ``pack:<name>`` (or ``pack:all``) into its live tool definitions."""
    key = (name or "").strip().lower()
    if key in ("all", "*"):
        seen: set[str] = set()
        out: list[ToolDefinition] = []
        for pack in list_packs():
            for td in pack.resolve():
                if td.name not in seen:
                    seen.add(td.name)
                    out.append(td)
        return out
    pack = get_pack(key)
    tools = pack.resolve()
    if not tools:
        row = pack.check()
        logger.warning("pack:%s has no live tools. %s", pack.name, row.fix_hint or "")
    return tools


# ---------------------------------------------------------------------------
# Web-reader backed tools (prompture.tools.web, imported lazily)
# ---------------------------------------------------------------------------


def _web() -> Any:
    try:
        return importlib.import_module("prompture.tools.web")
    except ImportError as exc:
        raise RuntimeError("web readers are not available (prompture.tools.web could not be imported)") from exc


def _format_search(results: Any, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    items = list(results or [])
    if not items:
        return "No results."
    lines: list[str] = []
    for i, r in enumerate(items, 1):
        get = r.get if isinstance(r, dict) else lambda k, _r=r: getattr(_r, k, None)
        title = get("title") or "(untitled)"
        url = get("url") or ""
        snippet = get("snippet") or get("content") or get("description") or ""
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   {url}")
        if snippet:
            lines.append("   " + " ".join(str(snippet).split())[:300])
    return _clip("\n".join(lines), max_chars)


def _format_read(result: Any, max_chars: int = 20_000) -> str:
    title = getattr(result, "title", None) or ""
    content = getattr(result, "content", None)
    if content is None:
        content = str(result)
    text = f"# {title}\n\n{content}" if title else str(content)
    return _clip(text, max_chars)


def _search(platform: str, query: str, max_results: int, kind: str | None = None) -> str:
    query = (query or "").strip()
    if not query:
        return "Error: query must not be empty"
    n = max(1, min(int(max_results), 25))
    kwargs: dict[str, Any] = {"max_results": n}
    if kind:
        kwargs["kind"] = kind
    return _format_search(_web().search_platform(platform, query, **kwargs))


def _read(url: str) -> str:
    return _format_read(_web().read_url(url))


_GITHUB_SHORT = re.compile(r"^[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}$")
_ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?)$")


def _host_of(url: str) -> str:
    return (urllib.parse.urlsplit(url).hostname or "").lower()


def github_search(
    query: str, kind: Literal["repositories", "issues", "code"] = "repositories", max_results: int = 10
) -> str:
    """Search GitHub repositories, issues or code.

    Args:
        query: Search terms (GitHub qualifiers like ``language:python`` work).
        kind: What to search: ``repositories``, ``issues`` or ``code``.
        max_results: Number of results (1-25).
    """
    return _search("github", query, max_results, None if kind == "repositories" else kind)


def github_read(target: str) -> str:
    """Read a GitHub repository, file, issue, pull request or discussion.

    Args:
        target: A github.com URL or ``owner/repo``.
    """
    target = (target or "").strip()
    if _GITHUB_SHORT.match(target):
        target = f"https://github.com/{target}"
    host = _host_of(target)
    if host not in ("github.com", "www.github.com", "gist.github.com", "raw.githubusercontent.com"):
        return "Error: expected a github.com URL or owner/repo"
    return _read(target)


def hackernews_search(query: str, max_results: int = 10) -> str:
    """Search Hacker News stories and comments.

    Args:
        query: Search terms.
        max_results: Number of results (1-25).
    """
    return _search("hackernews", query, max_results)


def hackernews_read(item: str) -> str:
    """Read a Hacker News item (story with its comment thread).

    Args:
        item: Item id (``"40123456"``) or news.ycombinator.com item URL.
    """
    item = (item or "").strip()
    if item.isdigit():
        item = f"https://news.ycombinator.com/item?id={item}"
    if _host_of(item) != "news.ycombinator.com":
        return "Error: expected a Hacker News item id or news.ycombinator.com URL"
    return _read(item)


def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers.

    Args:
        query: Search terms.
        max_results: Number of results (1-25).
    """
    return _search("arxiv", query, max_results)


def arxiv_read(paper: str) -> str:
    """Read an arXiv paper's abstract page.

    Args:
        paper: arXiv id (``"2401.01234"``) or arxiv.org abs/pdf URL.
    """
    paper = (paper or "").strip()
    if _ARXIV_ID.match(paper):
        paper = f"https://arxiv.org/abs/{paper}"
    if _host_of(paper) not in ("arxiv.org", "www.arxiv.org", "export.arxiv.org"):
        return "Error: expected an arXiv id or arxiv.org URL"
    return _read(paper)


def read_feed(url: str, max_chars: int = 15_000) -> str:
    """Read an RSS/Atom feed (or a page that advertises one) and list its latest entries.

    Args:
        url: Feed URL or a site URL that links to its feed.
        max_chars: Maximum characters returned.
    """
    url = (url or "").strip()
    if urllib.parse.urlsplit(url).scheme not in ("http", "https"):
        return "Error: expected an http(s) URL"
    return _format_read(_web().read_url(url), max_chars=max(500, min(int(max_chars), 50_000)))


# ---------------------------------------------------------------------------
# Package registries (public JSON APIs via safe_get)
# ---------------------------------------------------------------------------

_PYPI_NAME = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]{0,200}[A-Za-z0-9])?$")
_PYPI_VERSION = re.compile(r"^[A-Za-z0-9.!+_-]{1,64}$")
_NPM_NAME = re.compile(r"^(@[a-z0-9][a-z0-9._~-]{0,213}/)?[a-z0-9][a-z0-9._~-]{0,213}$")
_NPM_VERSION = re.compile(r"^[A-Za-z0-9.+_~^<>=*-]{1,64}$")


def _get_json(url: str, *, max_bytes: int) -> Any:
    from ..capabilities.http import safe_get

    resp = safe_get(url, headers={"Accept": "application/json"}, max_bytes=max_bytes, check_challenge=False)
    return resp.json()


def pypi_package(name: str, version: str = "") -> str:
    """Look up a Python package on PyPI: latest version, summary, links, Python requirement, dependencies.

    Args:
        name: Package name (e.g. ``"requests"``).
        version: Specific version (default: latest).
    """
    name = (name or "").strip()
    version = (version or "").strip()
    if not _PYPI_NAME.match(name):
        return f"Error: {name!r} is not a valid PyPI package name"
    if version and not _PYPI_VERSION.match(version):
        return f"Error: {version!r} is not a valid version"
    path = f"{urllib.parse.quote(name)}/{urllib.parse.quote(version)}" if version else urllib.parse.quote(name)
    try:
        data = _get_json(f"https://pypi.org/pypi/{path}/json", max_bytes=16_000_000)
    except Exception as exc:
        if "404" in str(exc):
            return f"Error: package {name!r}{' version ' + version if version else ''} not found on PyPI"
        raise
    info = data.get("info") or {}
    files = data.get("urls") or []
    uploaded = next((f.get("upload_time_iso_8601") or f.get("upload_time") for f in files if f), None)
    requires = info.get("requires_dist") or []
    out = {
        "name": info.get("name"),
        "version": info.get("version"),
        "summary": info.get("summary"),
        "requires_python": info.get("requires_python"),
        "license": info.get("license_expression") or (info.get("license") or "")[:200] or None,
        "author": info.get("author") or info.get("author_email"),
        "homepage": info.get("home_page") or None,
        "project_urls": info.get("project_urls"),
        "released": uploaded,
        "yanked": info.get("yanked"),
        "release_count": len(data["releases"]) if isinstance(data.get("releases"), dict) else None,
        "dependencies": requires[:40],
        "url": info.get("package_url") or f"https://pypi.org/project/{name}/",
    }
    return format_tool_result({k: v for k, v in out.items() if v not in (None, "", [], {})})


def npm_package(name: str, version: str = "latest") -> str:
    """Look up a JavaScript package on the npm registry: version, description, license, repository, dependencies.

    Args:
        name: Package name (e.g. ``"react"`` or ``"@types/node"``).
        version: Version or dist-tag (default ``"latest"``).
    """
    name = (name or "").strip().lower()
    version = (version or "latest").strip()
    if not _NPM_NAME.match(name):
        return f"Error: {name!r} is not a valid npm package name"
    if not _NPM_VERSION.match(version):
        return f"Error: {version!r} is not a valid version or tag"
    url = f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='@')}/{urllib.parse.quote(version)}"
    try:
        data = _get_json(url, max_bytes=4_000_000)
    except Exception as exc:
        if "404" in str(exc):
            return f"Error: package {name!r}@{version} not found on npm"
        raise
    if isinstance(data, str):  # registry answers a bare string for unknown tags
        return _error(data)
    repo = data.get("repository")
    out = {
        "name": data.get("name"),
        "version": data.get("version"),
        "description": data.get("description"),
        "license": data.get("license") if isinstance(data.get("license"), str) else None,
        "homepage": data.get("homepage"),
        "repository": repo.get("url") if isinstance(repo, dict) else repo,
        "engines": data.get("engines"),
        "dependencies": data.get("dependencies"),
        "peer_dependencies": data.get("peerDependencies"),
        "deprecated": data.get("deprecated"),
        "url": f"https://www.npmjs.com/package/{data.get('name') or name}",
    }
    return format_tool_result({k: v for k, v in out.items() if v not in (None, "", [], {})})


# ---------------------------------------------------------------------------
# Shipped packs
# ---------------------------------------------------------------------------

_TUKUY = ("tukuy", "httpx")
_WEB = ("prompture.tools.web:read_url", "prompture.tools.web:search_platform")


def _tk(
    plugin: str, skill: str, summary: str, *, env: tuple[str, ...] = (), description: str | None = None
) -> PackTool:
    return PackTool(
        name=skill,
        build=tukuy_tool(plugin, skill, description=description),
        description=summary,
        requires_env=env,
        requires_modules=(*_TUKUY, f"tukuy.plugins.{plugin}"),
        source=f"tukuy:{plugin}",
    )


def _fn(
    fn: Callable[..., str],
    summary: str,
    *,
    modules: tuple[str, ...] = (),
    optional_env: tuple[str, ...] = (),
    source: str = "",
) -> PackTool:
    return PackTool(
        name=fn.__name__,
        build=function_tool(fn),
        description=summary,
        requires_modules=modules,
        optional_env=optional_env,
        source=source,
    )


def builtin_packs() -> list[ToolPack]:
    """Fresh instances of the shipped packs."""
    finnhub = ("FINNHUB_API_KEY",)
    news = ("NEWSAPI_API_KEY",)
    maps = ("GOOGLE_MAPS_API_KEY",)
    return [
        ToolPack(
            "finance",
            "Stock quotes, company news and crypto prices.",
            [
                _tk("finnhub", "stock_quote", "real-time stock quote", env=finnhub),
                _tk("finnhub", "stock_news", "recent company news", env=finnhub),
                _tk("finnhub", "stock_search", "find ticker symbols", env=finnhub),
                _tk("coingecko", "crypto_price", "crypto prices, 24h change, market cap"),
                _tk("coingecko", "crypto_search", "find CoinGecko coin ids"),
                _tk("coingecko", "crypto_trending", "trending coins"),
            ],
        ),
        ToolPack(
            "news",
            "News headlines, topic search and RSS/Atom feeds.",
            [
                _tk("newsapi", "news_headlines", "top headlines by country/category", env=news),
                _tk("newsapi", "news_search", "search articles by topic", env=news),
                _fn(read_feed, "read an RSS/Atom feed", modules=_WEB, source="web"),
            ],
        ),
        ToolPack(
            "dev",
            "GitHub, Hacker News, arXiv and package registry lookups.",
            [
                _fn(github_search, "search GitHub", modules=_WEB, optional_env=("GITHUB_TOKEN",), source="web"),
                _fn(
                    github_read,
                    "read GitHub repos/files/issues/PRs",
                    modules=_WEB,
                    optional_env=("GITHUB_TOKEN",),
                    source="web",
                ),
                _fn(hackernews_search, "search Hacker News", modules=_WEB, source="web"),
                _fn(hackernews_read, "read a Hacker News thread", modules=_WEB, source="web"),
                _fn(arxiv_search, "search arXiv", modules=_WEB, source="web"),
                _fn(arxiv_read, "read an arXiv abstract", modules=_WEB, source="web"),
                _fn(pypi_package, "PyPI package lookup", source="http:pypi.org"),
                _fn(npm_package, "npm package lookup", source="http:registry.npmjs.org"),
            ],
        ),
        ToolPack(
            "places",
            "Geocoding, place search and country data.",
            [
                _tk("google_maps", "maps_geocode", "address → coordinates", env=maps),
                _tk("google_maps", "maps_reverse_geocode", "coordinates → address", env=maps),
                _tk("google_maps", "maps_places_search", "find places", env=maps),
                _tk("geocoding", "geocode", "address → coordinates (OpenCage)", env=("OPENCAGE_API_KEY",)),
                _tk("country", "country_info", "country facts by name or code"),
                _tk("country", "country_search", "countries by region/currency/language"),
            ],
        ),
    ]


for _pack in builtin_packs():
    register_pack(_pack)
