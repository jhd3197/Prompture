"""Web search with ordered, failover-capable backends.

:func:`web_search` runs a :class:`~prompture.capabilities.BackendChain` of
search backends. Keyed providers come first; the keyless Exa MCP endpoint is
the floor, so search works with nothing configured:

======== ======================== =========================================
Name     Key                      Notes
======== ======================== =========================================
tavily   ``TAVILY_API_KEY``       AI-oriented snippets + synthesized answer
exa      ``EXA_API_KEY``          REST ``/search`` with page contents
serper   ``SERPER_API_KEY``       Google results
brave    ``BRAVE_SEARCH_API_KEY`` Independent index
jina     ``JINA_API_KEY``         ``s.jina.ai``
searxng  ``SEARXNG_ENDPOINT``     Self-hosted metasearch
exa_mcp  none                     Public Exa MCP endpoint (keyless floor)
======== ======================== =========================================

``PROMPTURE_SEARCH_PROVIDERS="brave,exa_mcp"`` moves the named backends to
the front; ``providers=[...]`` restricts a single call. Auth, quota and
rate-limit errors move on to the next backend immediately; transient errors
retry once first. Every :class:`SearchResponse` says which backend served it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import threading
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from ...capabilities.backends import BackendChain, BaseBackend
from ...capabilities.http import proxies_for
from ._common import (
    API_USER_AGENT,
    clean_domain,
    config_value,
    dedupe_key,
    default_session,
    domain_matches,
    ensure_error_rules,
    parse_date,
    raise_for_status,
)
from ._mcp_http import MCPError, MCPHttpClient, tool_text
from ._types import SearchResponse, SearchResult

SEARCH_OVERRIDE_ENV = "PROMPTURE_SEARCH_PROVIDERS"
EXA_MCP_URL = "https://mcp.exa.ai/mcp"
DEFAULT_SEARCH_ORDER = ("tavily", "exa", "serper", "brave", "jina", "searxng", "exa_mcp")


@dataclass
class SearchRequest:
    """Normalized search parameters handed to every backend."""

    query: str
    max_results: int = 5
    include_domains: list[str] = field(default_factory=list)
    exclude_domains: list[str] = field(default_factory=list)
    recency_days: int | None = None

    @property
    def cutoff(self) -> datetime | None:
        if not self.recency_days or self.recency_days <= 0:
            return None
        return datetime.now(timezone.utc) - timedelta(days=self.recency_days)

    def query_with_operators(self) -> str:
        """Query with ``site:`` / ``-site:`` operators for engines that support them."""
        q = self.query
        if self.include_domains:
            sites = " OR ".join(f"site:{d}" for d in self.include_domains)
            q = f"{q} ({sites})" if len(self.include_domains) > 1 else f"{q} {sites}"
        for d in self.exclude_domains:
            q += f" -site:{d}"
        return q


@dataclass
class SearchHits:
    """Raw output of one backend before filtering."""

    results: list[SearchResult]
    answer: str | None = None


def _recency_bucket(days: int | None) -> str | None:
    if not days or days <= 0:
        return None
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    return "year"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class SearchBackend(BaseBackend):
    """Base class for search backends.

    Args:
        api_key: Explicit key; otherwise read from ``env_var`` / settings.
        endpoint: Base URL for self-hosted backends (SearXNG).
        session: ``requests.Session`` for connection reuse / tests.
        timeout: HTTP timeout in seconds.
    """

    name = "search"
    env_var: str | None = None
    settings_attr: str | None = None
    category = "tools"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._session = session
        self.timeout = timeout
        self.requires = (self.env_var,) if self.env_var else ()

    @property
    def session(self) -> requests.Session:
        return self._session or default_session()

    @property
    def proxies(self) -> dict[str, str] | None:
        return proxies_for(self.name)

    def key(self) -> str | None:
        return self._api_key or config_value(self.settings_attr, self.env_var)

    def available(self) -> bool:
        return self.keyless or bool(self.key())

    def unavailable_hint(self) -> str | None:
        return f"Set {self.env_var}" if self.env_var else None

    def run(self, request: SearchRequest) -> SearchHits:  # pragma: no cover - abstract
        raise NotImplementedError

    def live_check(self) -> None:
        self.run(SearchRequest("open source software", max_results=1))


class TavilyBackend(SearchBackend):
    name = "tavily"
    env_var = "TAVILY_API_KEY"
    settings_attr = "tavily_api_key"

    def __init__(self, *, include_answer: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.include_answer = include_answer

    def run(self, request: SearchRequest) -> SearchHits:
        payload: dict[str, Any] = {
            "api_key": self.key(),
            "query": request.query,
            "max_results": request.max_results,
            "include_answer": self.include_answer,
        }
        if request.include_domains:
            payload["include_domains"] = list(request.include_domains)
        if request.exclude_domains:
            payload["exclude_domains"] = list(request.exclude_domains)
        bucket = _recency_bucket(request.recency_days)
        if bucket:
            payload["time_range"] = bucket
        resp = self.session.post(
            "https://api.tavily.com/search", json=payload, timeout=self.timeout, proxies=self.proxies
        )
        raise_for_status(resp, self.name)
        data = resp.json() or {}
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=r.get("score"),
                extra={"published_date": r.get("published_date")},
            )
            for r in data.get("results", [])
        ]
        return SearchHits(results, data.get("answer"))


class ExaBackend(SearchBackend):
    name = "exa"
    env_var = "EXA_API_KEY"
    settings_attr = "exa_api_key"

    def run(self, request: SearchRequest) -> SearchHits:
        payload: dict[str, Any] = {
            "query": request.query,
            "numResults": request.max_results,
            "type": "auto",
            "contents": {"text": {"maxCharacters": 1200}},
        }
        if request.include_domains:
            payload["includeDomains"] = list(request.include_domains)
        if request.exclude_domains:
            payload["excludeDomains"] = list(request.exclude_domains)
        if request.cutoff:
            payload["startPublishedDate"] = request.cutoff.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        resp = self.session.post(
            "https://api.exa.ai/search",
            json=payload,
            headers={"x-api-key": self.key() or "", "Content-Type": "application/json"},
            timeout=self.timeout,
            proxies=self.proxies,
        )
        raise_for_status(resp, self.name)
        data = resp.json() or {}
        results = []
        for r in data.get("results", []):
            highlights = r.get("highlights") or []
            snippet = " ".join(highlights) if highlights else (r.get("text") or r.get("summary") or "")
            results.append(
                SearchResult(
                    title=r.get("title") or "",
                    url=r.get("url", ""),
                    snippet=snippet,
                    score=r.get("score"),
                    extra={"published_date": r.get("publishedDate"), "author": r.get("author")},
                )
            )
        return SearchHits(results)


class SerperBackend(SearchBackend):
    name = "serper"
    env_var = "SERPER_API_KEY"
    settings_attr = "serper_api_key"

    def run(self, request: SearchRequest) -> SearchHits:
        payload: dict[str, Any] = {"q": request.query_with_operators(), "num": request.max_results}
        bucket = _recency_bucket(request.recency_days)
        if bucket:
            payload["tbs"] = f"qdr:{bucket[0]}"
        resp = self.session.post(
            "https://google.serper.dev/search",
            json=payload,
            headers={"X-API-KEY": self.key() or "", "Content-Type": "application/json"},
            timeout=self.timeout,
            proxies=self.proxies,
        )
        raise_for_status(resp, self.name)
        data = resp.json() or {}
        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                extra={"position": r.get("position"), "published_date": r.get("date")},
            )
            for r in data.get("organic", [])[: request.max_results]
        ]
        answer_box = data.get("answerBox") or {}
        return SearchHits(results, answer_box.get("answer") or answer_box.get("snippet"))


class BraveBackend(SearchBackend):
    name = "brave"
    env_var = "BRAVE_SEARCH_API_KEY"
    settings_attr = "brave_search_api_key"

    def run(self, request: SearchRequest) -> SearchHits:
        params: dict[str, Any] = {"q": request.query_with_operators(), "count": request.max_results}
        bucket = _recency_bucket(request.recency_days)
        if bucket:
            params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}[bucket]
        resp = self.session.get(
            "https://api.search.brave.com/res/v1/web/search",
            params=params,
            headers={"X-Subscription-Token": self.key() or "", "Accept": "application/json"},
            timeout=self.timeout,
            proxies=self.proxies,
        )
        raise_for_status(resp, self.name)
        data = resp.json() or {}
        web_results = (data.get("web") or {}).get("results", [])
        return SearchHits(
            [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("description", ""),
                    extra={"age": r.get("age"), "published_date": r.get("page_age")},
                )
                for r in web_results[: request.max_results]
            ]
        )


class JinaSearchBackend(SearchBackend):
    name = "jina"
    env_var = "JINA_API_KEY"
    settings_attr = "jina_api_key"

    def run(self, request: SearchRequest) -> SearchHits:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.key()}",
            "X-Respond-With": "no-content",
            "User-Agent": API_USER_AGENT,
        }
        query = request.query
        if len(request.include_domains) == 1:
            headers["X-Site"] = request.include_domains[0]
        elif request.include_domains or request.exclude_domains:
            query = request.query_with_operators()
        resp = self.session.get(
            "https://s.jina.ai/",
            params={"q": query},
            headers=headers,
            timeout=max(self.timeout, 20.0),
            proxies=self.proxies,
        )
        raise_for_status(resp, self.name)
        data = resp.json() or {}
        items = data.get("data") or []
        return SearchHits(
            [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("description") or r.get("content") or "",
                    extra={"published_date": r.get("date") or r.get("publishedTime")},
                )
                for r in items[: request.max_results]
                if isinstance(r, dict)
            ]
        )


class SearxngBackend(SearchBackend):
    name = "searxng"
    env_var = "SEARXNG_ENDPOINT"
    settings_attr = "searxng_endpoint"

    def key(self) -> str | None:
        return self._endpoint or config_value(self.settings_attr, self.env_var)

    def run(self, request: SearchRequest) -> SearchHits:
        endpoint = self.key()
        if not endpoint:
            raise RuntimeError("searxng requires SEARXNG_ENDPOINT")
        params: dict[str, Any] = {"q": request.query_with_operators(), "format": "json"}
        bucket = _recency_bucket(request.recency_days)
        if bucket:
            params["time_range"] = bucket
        resp = self.session.get(
            endpoint.rstrip("/") + "/search", params=params, timeout=self.timeout, proxies=self.proxies
        )
        raise_for_status(resp, self.name)
        data = resp.json() or {}
        return SearchHits(
            [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    score=r.get("score"),
                    extra={"engine": r.get("engine"), "published_date": r.get("publishedDate")},
                )
                for r in data.get("results", [])[: request.max_results]
            ]
        )


_FIELD_RE = re.compile(r"^(Title|URL|Published(?: Date)?|Author|Score):\s*(.*)$", re.IGNORECASE)
_BODY_RE = re.compile(r"^(Highlights|Text|Summary|Content):\s*(.*)$", re.IGNORECASE)


def parse_exa_mcp_text(text: str) -> list[SearchResult]:
    """Parse the text payload of Exa's ``web_search_exa`` MCP tool.

    Handles both the JSON shape (``{"results": [...]}``) and the plain-text
    shape (``Title: …\\nURL: …\\nHighlights:\\n…`` blocks separated by ``---``).
    """
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            data = json.loads(stripped)
        except ValueError:
            data = None
        if data is not None:
            items = data.get("results", []) if isinstance(data, dict) else data
            out = []
            for r in items:
                if not isinstance(r, dict) or not r.get("url"):
                    continue
                highlights = r.get("highlights") or []
                out.append(
                    SearchResult(
                        title=r.get("title") or "",
                        url=r["url"],
                        snippet=" ".join(highlights) if highlights else (r.get("text") or r.get("summary") or ""),
                        score=r.get("score"),
                        extra={"published_date": r.get("publishedDate"), "author": r.get("author")},
                    )
                )
            return out

    blocks = re.split(r"\n\s*---+\s*\n|\n(?=Title: )", "\n" + stripped)
    out: list[SearchResult] = []
    for block in blocks:
        fields: dict[str, str] = {}
        body: list[str] = []
        in_body = False
        for line in block.strip().splitlines():
            if not in_body:
                m = _FIELD_RE.match(line.strip())
                if m:
                    fields[m.group(1).lower().split()[0]] = m.group(2).strip()
                    continue
                b = _BODY_RE.match(line.strip())
                if b:
                    in_body = True
                    if b.group(2):
                        body.append(b.group(2))
                    continue
            body.append(line)
        url = fields.get("url")
        if not url:
            continue
        published = fields.get("published")
        author = fields.get("author")
        snippet = " ".join(line.strip() for line in body if line.strip() and line.strip() != "...")
        out.append(
            SearchResult(
                title=fields.get("title", ""),
                url=url,
                snippet=snippet[:1200],
                extra={
                    "published_date": None if not published or published.upper() == "N/A" else published,
                    "author": None if not author or author.upper() == "N/A" else author,
                },
            )
        )
    return out


_mcp_clients: weakref.WeakKeyDictionary[Any, MCPHttpClient] = weakref.WeakKeyDictionary()
_mcp_lock = threading.Lock()


class ExaMCPBackend(SearchBackend):
    """Keyless search through Exa's public MCP endpoint (``web_search_exa``)."""

    name = "exa_mcp"
    keyless = True
    tool_name = "web_search_exa"

    def __init__(self, *, url: str = EXA_MCP_URL, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.url = url
        self.timeout = max(self.timeout, 30.0)

    def _client(self) -> MCPHttpClient:
        sess = self.session
        with _mcp_lock:
            try:
                client = _mcp_clients.get(sess)
            except TypeError:
                client = None
            if client is None or client.url != self.url:
                client = MCPHttpClient(self.url, session=sess, timeout=self.timeout, proxy_backend=self.name)
                with contextlib.suppress(TypeError):
                    _mcp_clients[sess] = client
            return client

    def run(self, request: SearchRequest) -> SearchHits:
        filtered = bool(request.include_domains or request.exclude_domains or request.recency_days)
        count = min(request.max_results * (3 if filtered else 1), 25)
        objective = f"Find the most relevant pages about: {request.query}"
        if request.include_domains:
            objective += f". Only pages on {', '.join(request.include_domains)}"
        if request.exclude_domains:
            objective += f". Exclude {', '.join(request.exclude_domains)}"
        if request.recency_days:
            objective += f". Published within the last {request.recency_days} days"
        args: dict[str, Any] = {"query": request.query, "numResults": count, "objective": objective}
        client = self._client()
        try:
            result = client.call_tool(self.tool_name, args)
        except MCPError as exc:
            # Older server versions reject the ``objective`` argument.
            if "objective" in str(exc).lower() or exc.code == -32602:
                args.pop("objective", None)
                result = client.call_tool(self.tool_name, args)
            else:
                raise
        return SearchHits(parse_exa_mcp_text(tool_text(result)))


SEARCH_BACKENDS: dict[str, type[SearchBackend]] = {
    "tavily": TavilyBackend,
    "exa": ExaBackend,
    "serper": SerperBackend,
    "brave": BraveBackend,
    "jina": JinaSearchBackend,
    "searxng": SearxngBackend,
    "exa_mcp": ExaMCPBackend,
}

# Env var that configures each backend (``None`` = keyless).
SEARCH_ENV_VARS: dict[str, str | None] = {name: cls.env_var for name, cls in SEARCH_BACKENDS.items()}


def build_search_backends(
    *,
    session: requests.Session | None = None,
    timeout: float = 10.0,
    names: Sequence[str] = DEFAULT_SEARCH_ORDER,
    api_keys: dict[str, str | None] | None = None,
    endpoint: str | None = None,
) -> list[SearchBackend]:
    """Instantiate search backends in *names* order."""
    keys = api_keys or {}
    out: list[SearchBackend] = []
    for name in names:
        cls = SEARCH_BACKENDS[name]
        kwargs: dict[str, Any] = {"session": session, "timeout": timeout, "api_key": keys.get(name)}
        if name == "searxng":
            kwargs["endpoint"] = endpoint
        out.append(cls(**kwargs))
    return out


def search_chain(
    *,
    session: requests.Session | None = None,
    timeout: float = 10.0,
    backends: Sequence[SearchBackend] | None = None,
) -> BackendChain[SearchHits]:
    """The default search chain (keyed providers first, ``exa_mcp`` last)."""
    ensure_error_rules()
    return BackendChain(
        list(backends) if backends is not None else build_search_backends(session=session, timeout=timeout),
        override_env=SEARCH_OVERRIDE_ENV,
        name="web_search",
    )


def finalize_results(results: Sequence[SearchResult], request: SearchRequest) -> list[SearchResult]:
    """Apply domain/recency filters, drop empty and duplicate URLs, cap the count."""
    seen: set[str] = set()
    cutoff = request.cutoff
    out: list[SearchResult] = []
    for r in results:
        if not r.url or not r.url.lower().startswith(("http://", "https://")):
            continue
        if request.include_domains and not domain_matches(r.url, request.include_domains):
            continue
        if request.exclude_domains and domain_matches(r.url, request.exclude_domains):
            continue
        if cutoff is not None:
            published = parse_date((r.extra or {}).get("published_date"))
            if published is not None and published < cutoff:
                continue
        key = dedupe_key(r.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= request.max_results:
            break
    return out


def make_request(
    query: str,
    *,
    max_results: int = 5,
    include_domains: Sequence[str] | None = None,
    exclude_domains: Sequence[str] | None = None,
    recency_days: int | None = None,
) -> SearchRequest:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return SearchRequest(
        query=query.strip(),
        max_results=max(1, min(int(max_results or 5), 50)),
        include_domains=[d for d in (clean_domain(x) for x in include_domains or ()) if d],
        exclude_domains=[d for d in (clean_domain(x) for x in exclude_domains or ()) if d],
        recency_days=int(recency_days) if recency_days else None,
    )


def run_search(
    chain: BackendChain[SearchHits], request: SearchRequest, *, only: Sequence[str] | None = None
) -> SearchResponse:
    """Run *request* through *chain* and build a :class:`SearchResponse`."""
    res = chain.run(request, only=only)
    hits: SearchHits = res.value
    return SearchResponse(
        query=request.query,
        results=finalize_results(hits.results, request),
        served_by=res.served_by,
        route=res.route,
        answer=hits.answer,
    )


def web_search(
    query: str,
    *,
    max_results: int = 5,
    providers: Sequence[str] | None = None,
    include_domains: Sequence[str] | None = None,
    exclude_domains: Sequence[str] | None = None,
    recency_days: int | None = None,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> SearchResponse:
    """Search the public web through the backend chain.

    Args:
        query: Search query.
        max_results: Number of results to return (1–50).
        providers: Restrict to these backend names, in this order.
        include_domains: Only keep results on these domains (and subdomains).
        exclude_domains: Drop results on these domains.
        recency_days: Prefer / keep results published within this many days.
        session: Optional ``requests.Session`` (connection reuse / tests).
        timeout: Per-request timeout in seconds.

    Raises:
        ValueError: Empty query or unknown provider name.
        AllBackendsFailedError: Every backend failed; ``attempts`` has the route.
    """
    request = make_request(
        query,
        max_results=max_results,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        recency_days=recency_days,
    )
    only: list[str] | None = None
    if providers is not None:
        only = [p.strip().lower() for p in ([providers] if isinstance(providers, str) else providers) if p.strip()]
        unknown = [p for p in only if p not in SEARCH_BACKENDS]
        if unknown:
            raise ValueError(f"Unknown search provider(s) {unknown}. Known: {', '.join(SEARCH_BACKENDS)}")
    return run_search(search_chain(session=session, timeout=timeout), request, only=only)


async def asearch(
    query: str,
    *,
    max_results: int = 5,
    providers: Sequence[str] | None = None,
    include_domains: Sequence[str] | None = None,
    exclude_domains: Sequence[str] | None = None,
    recency_days: int | None = None,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> SearchResponse:
    """Async :func:`web_search` (runs the blocking chain in a worker thread)."""
    return await asyncio.to_thread(
        web_search,
        query,
        max_results=max_results,
        providers=providers,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        recency_days=recency_days,
        session=session,
        timeout=timeout,
    )
