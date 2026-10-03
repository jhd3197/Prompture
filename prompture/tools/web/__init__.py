"""Web capability: search, fetch, URL readers and platform search — zero keys required.

* :func:`web_search` — ordered search backends with failover; keyless Exa MCP floor.
* :func:`web_fetch` — URL → Markdown (Jina reader ▸ direct), paging, 10-minute cache.
* :func:`read_url` — URL-routed readers (YouTube, GitHub, Hacker News, arXiv,
  Wikipedia, podcasts, feeds) with ``web_fetch`` fallback.
* :func:`search_platform` — YouTube / GitHub / Hacker News / arXiv search.
* :class:`WebToolkit` — the above as agent tools; ``tools=["web:all"]``.

Example::

    from prompture.tools.web import web_search, read_url

    resp = web_search("latest CPython release", max_results=3)
    print(resp.to_markdown())          # ... _served by exa_mcp_
    print(read_url(resp.results[0].url).content[:500])

Health rows for ``prompture doctor`` live in :mod:`prompture.tools.web.health`.
"""

from __future__ import annotations

from ._mcp_http import MCPError, MCPHttpClient
from ._types import SearchResponse, SearchResult
from .cache import cache_info, clear_web_cache
from .fetch import (
    FETCH_BACKENDS,
    FETCH_OVERRIDE_ENV,
    DirectBackend,
    FetchBackend,
    FetchResult,
    JinaReaderBackend,
    afetch,
    clear_fetch_cache,
    fetch_chain,
    web_fetch,
)
from .html2md import find_feed_links, html_to_markdown
from .platform import PLATFORMS, platform_search, search_platform
from .readers import (
    BaseReader,
    Reader,
    ReadResult,
    StepBackend,
    aread_url,
    get_reader,
    list_readers,
    read_url,
    register_reader,
    unregister_reader,
)
from .search import (
    DEFAULT_SEARCH_ORDER,
    SEARCH_BACKENDS,
    SEARCH_OVERRIDE_ENV,
    BraveBackend,
    ExaBackend,
    ExaMCPBackend,
    JinaSearchBackend,
    SearchBackend,
    SearchRequest,
    SearxngBackend,
    SerperBackend,
    TavilyBackend,
    asearch,
    search_chain,
    web_search,
)
from .toolkit import WebToolkit, resolve_web_tools

__all__ = [
    "DEFAULT_SEARCH_ORDER",
    "FETCH_BACKENDS",
    "FETCH_OVERRIDE_ENV",
    "PLATFORMS",
    "SEARCH_BACKENDS",
    "SEARCH_OVERRIDE_ENV",
    "BaseReader",
    "BraveBackend",
    "DirectBackend",
    "ExaBackend",
    "ExaMCPBackend",
    "FetchBackend",
    "FetchResult",
    "JinaReaderBackend",
    "JinaSearchBackend",
    "MCPError",
    "MCPHttpClient",
    "ReadResult",
    "Reader",
    "SearchBackend",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearxngBackend",
    "SerperBackend",
    "StepBackend",
    "TavilyBackend",
    "WebToolkit",
    "afetch",
    "aread_url",
    "asearch",
    "cache_info",
    "clear_fetch_cache",
    "clear_web_cache",
    "fetch_chain",
    "find_feed_links",
    "get_reader",
    "html_to_markdown",
    "list_readers",
    "platform_search",
    "read_url",
    "register_reader",
    "resolve_web_tools",
    "search_chain",
    "search_platform",
    "unregister_reader",
    "web_fetch",
    "web_search",
]
