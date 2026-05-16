"""Web search tool for Prompture agents.

Provides :class:`WebSearchTool` — a drop-in ``web_search`` tool backed by
one of four search providers:

* **Tavily** — default; AI-friendly snippets, free tier. ``TAVILY_API_KEY``
* **Serper** — Google Search API wrapper. ``SERPER_API_KEY``
* **Brave Search** — independent index. ``BRAVE_SEARCH_API_KEY``
* **SearXNG** — self-hosted metasearch (no key, set ``SEARXNG_ENDPOINT``)

The tool returns Markdown-formatted results so the LLM can cite each
source by URL inline.

Example::

    from prompture import Agent, ToolRegistry, WebSearchTool

    registry = ToolRegistry()
    WebSearchTool().register_on(registry)  # auto-pick provider from env

    agent = Agent("openai/gpt-4o", tools=registry)
    print(agent.run("What is the latest LangChain release?").output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from ..agents.tools_schema import ToolDefinition
from ..infra.settings import settings


@dataclass
class SearchResult:
    """A single web-search hit.

    Attributes:
        title: Page title.
        url: Canonical URL.
        snippet: Short excerpt or summary.
        score: Provider-supplied relevance score (0.0–1.0 when present).
        extra: Provider-specific raw fields (e.g. published date).
    """

    title: str
    url: str
    snippet: str
    score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# Provider name → (env-key attr on Settings, search function)
_PROVIDER_ORDER = ["tavily", "serper", "brave", "searxng"]


def _auto_detect_provider() -> str:
    """Pick the first configured provider in priority order."""
    if settings.tavily_api_key:
        return "tavily"
    if settings.serper_api_key:
        return "serper"
    if settings.brave_search_api_key:
        return "brave"
    if settings.searxng_endpoint:
        return "searxng"
    raise RuntimeError(
        "No web-search provider configured. Set one of "
        "TAVILY_API_KEY, SERPER_API_KEY, BRAVE_SEARCH_API_KEY, or "
        "SEARXNG_ENDPOINT — see .env.copy for examples."
    )


class WebSearchTool:
    """Build a ``web_search`` tool.

    Args:
        provider: ``"tavily"`` (default), ``"serper"``, ``"brave"``, or
            ``"searxng"``.  When ``None`` (the default), the first
            configured backend is used in the order above.
        api_key: Override the key for the selected provider.  Falls back
            to the matching ``Settings`` field.
        endpoint: Override the SearXNG base URL (ignored by other
            providers).  Defaults to ``Settings.searxng_endpoint``.
        max_results: Number of hits to fetch per query (default 5).
        timeout: HTTP timeout in seconds (default 10).
        include_raw_answer: When ``True`` and the provider supplies a
            synthesised answer (Tavily), include it at the top of the
            Markdown output.  Default ``True``.
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
        self.provider = (provider or _auto_detect_provider()).lower()
        if self.provider not in _PROVIDER_ORDER:
            raise ValueError(f"Unknown provider {self.provider!r}. Expected one of: {', '.join(_PROVIDER_ORDER)}")
        self.api_key = api_key or self._resolve_api_key(self.provider)
        self.endpoint = endpoint or settings.searxng_endpoint
        if self.provider == "searxng" and not self.endpoint:
            raise RuntimeError("provider='searxng' requires SEARXNG_ENDPOINT (or the endpoint= argument).")
        if self.provider != "searxng" and not self.api_key:
            raise RuntimeError(
                f"provider={self.provider!r} requires an API key. Set {self._env_name(self.provider)} or pass api_key=."
            )

        self.max_results = max_results
        self.timeout = timeout
        self.include_raw_answer = include_raw_answer
        self.tool_name = tool_name
        self.tool_description = tool_description or self.DEFAULT_DESCRIPTION
        self._session = session or requests.Session()

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[SearchResult]:
        """Run *query* through the configured backend and return results."""
        fn = self._backends()[self.provider]
        return fn(query)

    def search_markdown(self, query: str) -> str:
        """Search and return the results as a Markdown block."""
        results = self.search(query)
        if not results:
            return f'No results for "{query}".'
        lines: list[str] = []
        # Tavily synthesises a short answer when available.
        if self.include_raw_answer and self.provider == "tavily":
            answer = getattr(self, "_last_answer", None)
            if answer:
                lines.append(f"**Answer:** {answer}\n")
        lines.append(f'### Search results for "{query}"')
        for i, r in enumerate(results, 1):
            snippet = (r.snippet or "").strip().replace("\n", " ")
            lines.append(f"{i}. **{r.title}** — <{r.url}>\n   {snippet}")
        return "\n".join(lines)

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
            except requests.RequestException as exc:
                return f"Error: web search failed ({type(exc).__name__}): {exc}"

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
    # Provider implementations
    # ------------------------------------------------------------------

    def _backends(self) -> dict[str, Callable[[str], list[SearchResult]]]:
        return {
            "tavily": self._search_tavily,
            "serper": self._search_serper,
            "brave": self._search_brave,
            "searxng": self._search_searxng,
        }

    def _search_tavily(self, query: str) -> list[SearchResult]:
        resp = self._session.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "max_results": self.max_results,
                "include_answer": self.include_raw_answer,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._last_answer = data.get("answer")
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=r.get("score"),
                extra={"published_date": r.get("published_date")},
            )
            for r in data.get("results", [])
        ]

    def _search_serper(self, query: str) -> list[SearchResult]:
        resp = self._session.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": self.max_results},
            headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                extra={"position": r.get("position")},
            )
            for r in data.get("organic", [])[: self.max_results]
        ]

    def _search_brave(self, query: str) -> list[SearchResult]:
        resp = self._session.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": self.max_results},
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        web_results = (data.get("web") or {}).get("results", [])
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                extra={"age": r.get("age")},
            )
            for r in web_results[: self.max_results]
        ]

    def _search_searxng(self, query: str) -> list[SearchResult]:
        url = self.endpoint.rstrip("/") + "/search"
        resp = self._session.get(
            url,
            params={"q": query, "format": "json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                score=r.get("score"),
                extra={"engine": r.get("engine")},
            )
            for r in data.get("results", [])[: self.max_results]
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_key(provider: str) -> str | None:
        return {
            "tavily": settings.tavily_api_key,
            "serper": settings.serper_api_key,
            "brave": settings.brave_search_api_key,
            "searxng": None,
        }.get(provider)

    @staticmethod
    def _env_name(provider: str) -> str:
        return {
            "tavily": "TAVILY_API_KEY",
            "serper": "SERPER_API_KEY",
            "brave": "BRAVE_SEARCH_API_KEY",
            "searxng": "SEARXNG_ENDPOINT",
        }[provider]

    def __call__(self, query: str) -> str:
        return self.search_markdown(query)


def web_search_tool(**kwargs: Any) -> ToolDefinition:
    """Shortcut: build and return a ``web_search`` ToolDefinition.

    Accepts the same keyword arguments as :class:`WebSearchTool`.
    """
    return WebSearchTool(**kwargs).to_tool_definition()
