"""Web search tool for Prompture agents.

Provides :class:`WebSearchTool` — a drop-in ``web_search`` tool on top of
:mod:`prompture.tools.web.search`:

* **Tavily** — AI-friendly snippets + answer. ``TAVILY_API_KEY``
* **Exa** — neural search with page contents. ``EXA_API_KEY``
* **Serper** — Google Search API wrapper. ``SERPER_API_KEY``
* **Brave Search** — independent index. ``BRAVE_SEARCH_API_KEY``
* **Jina** — ``s.jina.ai``. ``JINA_API_KEY``
* **SearXNG** — self-hosted metasearch (no key, set ``SEARXNG_ENDPOINT``)
* **Exa MCP** — Exa's public MCP endpoint; no key at all (the default floor)

With ``provider=None`` every configured provider is tried in that order with
failover, ending at the keyless Exa MCP endpoint — so the tool works with
nothing configured. ``provider="x"`` pins a single provider (no failover).

The tool returns Markdown-formatted results so the LLM can cite each
source by URL inline, with a ``served by <provider>`` footer.

Example::

    from prompture import Agent, ToolRegistry, WebSearchTool

    registry = ToolRegistry()
    WebSearchTool().register_on(registry)  # keyed providers first, keyless fallback

    agent = Agent("openai/gpt-4o", tools=registry)
    print(agent.run("What is the latest LangChain release?").output)
"""

from __future__ import annotations

from typing import Any

import requests

from ..agents.tools_schema import ToolDefinition
from ..capabilities.backends import BackendChain
from ..infra.settings import settings
from .web._common import error_text
from .web._types import SearchResponse, SearchResult
from .web.search import (
    DEFAULT_SEARCH_ORDER,
    SEARCH_BACKENDS,
    SEARCH_OVERRIDE_ENV,
    SearchBackend,
    TavilyBackend,
    finalize_results,
    make_request,
    run_search,
)

__all__ = ["SearchResponse", "SearchResult", "WebSearchTool", "web_search_tool"]

# Providers in auto-detect priority order (``exa_mcp`` is the keyless floor).
_PROVIDER_ORDER = list(DEFAULT_SEARCH_ORDER)

_SETTINGS_ATTR = {
    "tavily": "tavily_api_key",
    "exa": "exa_api_key",
    "serper": "serper_api_key",
    "brave": "brave_search_api_key",
    "jina": "jina_api_key",
    "searxng": "searxng_endpoint",
}


def _configured_providers() -> list[str]:
    """Keyed providers that have a key (or endpoint) in ``settings``, in priority order."""
    return [p for p in _PROVIDER_ORDER if p in _SETTINGS_ATTR and getattr(settings, _SETTINGS_ATTR[p], None)]


def _auto_detect_provider() -> str:
    """Pick the first configured provider in priority order, else the keyless ``exa_mcp``."""
    configured = _configured_providers()
    return configured[0] if configured else "exa_mcp"


class WebSearchTool:
    """Build a ``web_search`` tool.

    Args:
        provider: ``"tavily"``, ``"exa"``, ``"serper"``, ``"brave"``,
            ``"jina"``, ``"searxng"`` or ``"exa_mcp"``. When ``None`` (the
            default), every configured provider is tried in that order with
            failover, ending at the keyless ``exa_mcp``.
        api_key: Override the key for the selected provider. Falls back
            to the matching ``Settings`` field.
        endpoint: Override the SearXNG base URL (ignored by other
            providers). Defaults to ``Settings.searxng_endpoint``.
        max_results: Number of hits to fetch per query (default 5).
        timeout: HTTP timeout in seconds (default 10).
        include_raw_answer: When ``True`` and the provider supplies a
            synthesised answer (Tavily), include it at the top of the
            Markdown output. Default ``True``.
        tool_name: Override the tool name (default ``web_search``).
        tool_description: Override the tool description shown to the LLM.
        session: Optional ``requests.Session`` for connection reuse /
            test injection.
    """

    DEFAULT_DESCRIPTION = (
        "Search the public web and return relevant snippets with URLs. "
        "Use this when the user asks about current events, real-world "
        "facts, or anything that may have changed recently. Cite "
        "results by URL when answering."
    )

    def __init__(
        self,
        *,
        provider: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        max_results: int = 5,
        timeout: float = 10.0,
        include_raw_answer: bool = True,
        tool_name: str = "web_search",
        tool_description: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.pinned = provider is not None
        self.provider = (provider or _auto_detect_provider()).lower()
        if self.provider not in SEARCH_BACKENDS:
            raise ValueError(f"Unknown provider {self.provider!r}. Expected one of: {', '.join(_PROVIDER_ORDER)}")
        self.api_key = api_key or self._resolve_api_key(self.provider)
        self.endpoint = endpoint or getattr(settings, "searxng_endpoint", None)
        if self.pinned:
            if self.provider == "searxng" and not self.endpoint:
                raise RuntimeError("provider='searxng' requires SEARXNG_ENDPOINT (or the endpoint= argument).")
            if self.provider not in ("searxng", "exa_mcp") and not self.api_key:
                raise RuntimeError(
                    f"provider={self.provider!r} requires an API key. Set {self._env_name(self.provider)} or pass api_key=."
                )

        self.max_results = max_results
        self.timeout = timeout
        self.include_raw_answer = include_raw_answer
        self.tool_name = tool_name
        self.tool_description = tool_description or self.DEFAULT_DESCRIPTION
        self._session = session or requests.Session()
        self._last_answer: str | None = None
        self.last_response: SearchResponse | None = None

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _backend(self, name: str, api_key: str | None) -> SearchBackend:
        cls = SEARCH_BACKENDS[name]
        kwargs: dict[str, Any] = {"session": self._session, "timeout": self.timeout, "api_key": api_key}
        if name == "searxng":
            kwargs["endpoint"] = self.endpoint
        if cls is TavilyBackend:
            kwargs["include_answer"] = self.include_raw_answer
        return cls(**kwargs)

    def _chain(self) -> BackendChain[Any]:
        if self.pinned:
            return BackendChain([self._backend(self.provider, self.api_key)], name="web_search")
        names = [*_configured_providers(), "exa_mcp"]
        backends = [self._backend(n, self.api_key if n == self.provider else self._resolve_api_key(n)) for n in names]
        return BackendChain(backends, override_env=SEARCH_OVERRIDE_ENV, name="web_search")

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search_response(self, query: str) -> SearchResponse:
        """Search and return a :class:`SearchResponse` (results + route)."""
        request = make_request(query, max_results=self.max_results)
        if self.pinned:
            # Single provider: call it directly so its errors surface unchanged.
            backend = self._backend(self.provider, self.api_key)
            hits = backend.run(request)
            resp = SearchResponse(
                query=request.query,
                results=finalize_results(hits.results, request),
                served_by=backend.name,
                route={
                    "served_by": backend.name,
                    "fallback": False,
                    "attempts": [{"backend": backend.name, "status": "ok", "try": 1}],
                },
                answer=hits.answer,
            )
        else:
            resp = run_search(self._chain(), request)
        self._last_answer = resp.answer
        self.last_response = resp
        return resp

    def search(self, query: str) -> list[SearchResult]:
        """Run *query* and return the results."""
        return self.search_response(query).results

    def search_markdown(self, query: str) -> str:
        """Search and return the results as a Markdown block."""
        resp = self.search_response(query)
        if not resp.results:
            return f'No results for "{query}".'
        return resp.to_markdown(include_answer=self.include_raw_answer)

    # ------------------------------------------------------------------
    # ToolDefinition
    # ------------------------------------------------------------------

    def to_tool_definition(self) -> ToolDefinition:
        """Return a :class:`ToolDefinition` ready to register on a registry."""

        def _web_search(query: str) -> str:
            """Search the public web.

            Args:
                query: The search query string.
            """
            try:
                return self.search_markdown(query)
            except Exception as exc:
                return f"Error: web search failed ({error_text(exc)})"

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        }

        return ToolDefinition(
            name=self.tool_name,
            description=self.tool_description,
            parameters=parameters,
            function=_web_search,
        )

    def register_on(self, registry: Any) -> ToolDefinition:
        """Add the tool to *registry* and return its definition."""
        td = self.to_tool_definition()
        registry.add(td)
        return td

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_key(provider: str) -> str | None:
        attr = _SETTINGS_ATTR.get(provider)
        if attr is None or provider == "searxng":
            return None
        return getattr(settings, attr, None)

    @staticmethod
    def _env_name(provider: str) -> str:
        return SEARCH_BACKENDS[provider].env_var or provider.upper()

    def __call__(self, query: str) -> str:
        return self.search_markdown(query)


def web_search_tool(**kwargs: Any) -> ToolDefinition:
    """Shortcut: build and return a ``web_search`` ToolDefinition.

    Accepts the same keyword arguments as :class:`WebSearchTool`.
    """
    return WebSearchTool(**kwargs).to_tool_definition()
