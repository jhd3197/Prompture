"""Agent tools for the web capability: search, fetch, read, platform search, media.

:class:`WebToolkit` builds :class:`~prompture.agents.tools_schema.ToolDefinition`
objects whose functions always return a string — failures come back as a
scrubbed ``Error: ...`` line the model can react to, never as an exception.

``tools=["web:all"]`` (or ``web:search``, ``web:fetch``, ``web:read``,
``web:platform``, ``web:media``) resolves through :func:`resolve_web_tools`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import requests

from ...agents.tools_schema import ToolDefinition, tool_from_function
from ._common import error_text
from .fetch import web_fetch as _web_fetch
from .platform import PLATFORMS, platform_search
from .readers import read_url as _read_url
from .search import web_search as _web_search

logger = logging.getLogger("prompture.tools.web")

TOOL_GROUPS = ("search", "fetch", "read", "platform", "media")


def _error(action: str, exc: BaseException) -> str:
    return f"Error: {action} failed ({error_text(exc)})"


class WebToolkit:
    """Bundle of web tools for agents.

    Args:
        include: Groups to build — any of ``search``, ``fetch``, ``read``,
            ``platform``, ``media`` (media tools are added only when
            :mod:`prompture.media.understand` is importable).
        max_results: Default number of search hits.
        max_chars: Characters per fetched/read slice.
        providers: Restrict ``web_search`` to these backends.
        summarize_model: LLM used by ``summarize_media``.
        session: Optional ``requests.Session`` shared by every tool.
    """

    def __init__(
        self,
        *,
        include: Sequence[str] = TOOL_GROUPS,
        max_results: int = 5,
        max_chars: int = 20000,
        providers: Sequence[str] | None = None,
        summarize_model: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        unknown = [g for g in include if g not in TOOL_GROUPS]
        if unknown:
            raise ValueError(f"Unknown web tool group(s) {unknown}. Known: {', '.join(TOOL_GROUPS)}")
        self.include = tuple(include)
        self.max_results = max_results
        self.max_chars = max_chars
        self.providers = list(providers) if providers else None
        self.summarize_model = summarize_model
        self.session = session

    # ------------------------------------------------------------------
    # Individual tools
    # ------------------------------------------------------------------

    def web_search_tool(self) -> ToolDefinition:
        kit = self

        def web_search(
            query: str,
            max_results: int = kit.max_results,
            include_domains: list[str] | None = None,
            exclude_domains: list[str] | None = None,
            recency_days: int | None = None,
        ) -> str:
            """Search the public web and return results with URLs.

            Use for current events, facts that may have changed, or finding
            pages to read. Cite results by URL.

            Args:
                query: The search query.
                max_results: Number of results (1-20).
                include_domains: Only return results from these domains, e.g. ["python.org"].
                exclude_domains: Drop results from these domains.
                recency_days: Only results published within this many days.
            """
            try:
                resp = _web_search(
                    query,
                    max_results=min(int(max_results or kit.max_results), 20),
                    providers=kit.providers,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                    recency_days=recency_days,
                    session=kit.session,
                )
                return resp.to_markdown()
            except Exception as exc:
                return _error("web search", exc)

        return tool_from_function(web_search, metadata={"category": "web"})

    def web_fetch_tool(self) -> ToolDefinition:
        kit = self

        def web_fetch(url: str, start: int = 0) -> str:
            """Fetch a web page (or PDF) and return its content as Markdown.

            Long pages are returned in slices; when the output ends with
            "[truncated — call again with start=N]", call again with that start.

            Args:
                url: Public http(s) URL to fetch.
                start: Character offset for the next slice of a long page.
            """
            try:
                return _web_fetch(
                    url, start=int(start or 0), max_chars=kit.max_chars, session=kit.session
                ).to_markdown()
            except Exception as exc:
                return _error("web fetch", exc)

        return tool_from_function(web_fetch, metadata={"category": "web"})

    def read_url_tool(self) -> ToolDefinition:
        kit = self

        def read_url(url: str, start: int = 0) -> str:
            """Read a URL with the best specialized reader.

            YouTube videos return transcripts, GitHub links return repo/file/issue/PR
            content, Hacker News threads return comments, arXiv returns paper
            metadata and abstract, Wikipedia returns the article, feeds return the
            latest entries, podcast episodes return transcripts. Anything else is
            fetched as a web page.

            Args:
                url: Public http(s) URL to read.
                start: Character offset for the next slice of long content.
            """
            try:
                return _read_url(url, start=int(start or 0), max_chars=kit.max_chars, session=kit.session).to_markdown()
            except Exception as exc:
                return _error("read_url", exc)

        return tool_from_function(read_url, metadata={"category": "web"})

    def search_platform_tool(self) -> ToolDefinition:
        kit = self

        def search_platform(platform: str, query: str, max_results: int = 10, kind: str = "") -> str:
            """Search inside a platform: youtube, github, hackernews or arxiv.

            Args:
                platform: One of youtube, github, hackernews, arxiv.
                query: Search query.
                max_results: Number of results (1-25).
                kind: GitHub: repositories (default), issues or code. Hacker News: story (default), comment or all.
            """
            try:
                resp = platform_search(
                    platform,
                    query,
                    max_results=min(int(max_results or 10), 25),
                    kind=kind or None,
                    session=kit.session,
                )
                return resp.to_markdown()
            except Exception as exc:
                return _error("platform search", exc)

        td = tool_from_function(search_platform, metadata={"category": "web"})
        td.parameters["properties"]["platform"]["enum"] = list(PLATFORMS)
        return td

    def media_tools(self) -> list[ToolDefinition]:
        """``transcribe_media`` / ``summarize_media`` when media understanding is installed."""
        try:
            from ...media.understand.tools import summarize_media_tool, transcribe_media_tool
        except ImportError:
            return []
        out: list[ToolDefinition] = []
        for build in (transcribe_media_tool, lambda: summarize_media_tool(model=self.summarize_model)):
            try:
                out.append(build())
            except Exception:
                logger.debug("media tool unavailable", exc_info=True)
        return out

    # ------------------------------------------------------------------
    # Bundles
    # ------------------------------------------------------------------

    def tools(self) -> list[ToolDefinition]:
        """Every tool in the included groups."""
        out: list[ToolDefinition] = []
        if "search" in self.include:
            out.append(self.web_search_tool())
        if "fetch" in self.include:
            out.append(self.web_fetch_tool())
        if "read" in self.include:
            out.append(self.read_url_tool())
        if "platform" in self.include:
            out.append(self.search_platform_tool())
        if "media" in self.include:
            out.extend(self.media_tools())
        return out

    def register_on(self, registry: Any) -> list[ToolDefinition]:
        """Add every tool to *registry* (anything with ``add(ToolDefinition)``)."""
        tools = self.tools()
        for td in tools:
            registry.add(td)
        return tools


_NAMESPACE_GROUPS = {
    "all": TOOL_GROUPS,
    "*": TOOL_GROUPS,
    "search": ("search",),
    "fetch": ("fetch",),
    "read": ("read",),
    "reader": ("read",),
    "platform": ("platform",),
    "media": ("media",),
}


def resolve_web_tools(name: str) -> list[ToolDefinition]:
    """Resolver for the ``web:`` tool namespace (``web:all``, ``web:search``, ...).

    Several groups may be combined with ``+``: ``web:search+fetch``.
    """
    groups: list[str] = []
    for part in (name or "all").lower().replace(",", "+").split("+"):
        part = part.strip() or "all"
        if part not in _NAMESPACE_GROUPS:
            raise ValueError(f"Unknown web tool set {part!r}. Known: {', '.join(sorted(_NAMESPACE_GROUPS))}")
        groups.extend(g for g in _NAMESPACE_GROUPS[part] if g not in groups)
    return WebToolkit(include=groups).tools()
