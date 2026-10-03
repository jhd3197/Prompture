"""Search inside platforms that offer public search: YouTube, GitHub, Hacker News, arXiv.

:func:`search_platform` returns :class:`SearchResult` hits; each hit's
``extra["served_by"]`` names the backend that produced it.

=========== ===================================================== ==========================
Platform    Chain                                                 Notes
=========== ===================================================== ==========================
youtube     ``yt-dlp ytsearchN:`` ▸ ``web_search`` (youtube.com)  no API key needed
github      REST search (``GITHUB_TOKEN`` optional) ▸ ``gh``      ``kind=repositories|issues|code``
hackernews  Algolia HN search                                     ``kind=story|comment|all``
arxiv       export.arxiv.org API                                  relevance-sorted
=========== ===================================================== ==========================
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

from ...capabilities.backends import BackendChain
from ...capabilities.errors import BackendUnavailableError
from . import _common
from ._common import ensure_error_rules, json_lines, run_command
from ._types import SearchResponse, SearchResult
from .html2md import html_to_text
from .readers.arxiv import API_URL as ARXIV_API
from .readers.arxiv import parse_arxiv_atom
from .readers.base import StepBackend, http_get, http_json
from .readers.github import GhTransport, RestTransport, github_token
from .search import web_search

PLATFORMS = ("youtube", "github", "hackernews", "arxiv")
GITHUB_KINDS = {
    "repositories": "repositories",
    "repos": "repositories",
    "repo": "repositories",
    "issues": "issues",
    "prs": "issues",
    "pulls": "issues",
    "code": "code",
}


def _yt_dlp_search(query: str, *, max_results: int = 10, **_: Any) -> list[SearchResult]:
    out = run_command(
        [
            "yt-dlp",
            f"ytsearch{max_results}:{query}",
            "--dump-json",
            "--flat-playlist",
            "--skip-download",
            "--no-warnings",
        ],
        timeout=90,
    )
    results = []
    for item in json_lines(out):
        vid = item.get("id")
        if not vid:
            continue
        url = (
            item.get("url") if str(item.get("url", "")).startswith("http") else f"https://www.youtube.com/watch?v={vid}"
        )
        results.append(
            SearchResult(
                title=item.get("title") or vid,
                url=url,
                snippet=(item.get("description") or "")[:400],
                extra={
                    "video_id": vid,
                    "channel": item.get("channel") or item.get("uploader"),
                    "duration": item.get("duration"),
                    "view_count": item.get("view_count"),
                },
            )
        )
    return results


def _youtube_via_web(query: str, *, max_results: int = 10, session: Any = None, **_: Any) -> list[SearchResult]:
    resp = web_search(
        f"{query} youtube video", max_results=max_results, include_domains=["youtube.com", "youtu.be"], session=session
    )
    return resp.results


def _github_search(transport: Any, query: str, *, max_results: int, kind: str) -> list[SearchResult]:
    endpoint = GITHUB_KINDS.get(kind.lower())
    if endpoint is None:
        raise ValueError(f"Unknown GitHub search kind {kind!r}. Use repositories, issues or code.")
    data = transport.get(f"search/{endpoint}", params={"q": query, "per_page": max_results})
    results = []
    for item in (data or {}).get("items", [])[:max_results]:
        if endpoint == "repositories":
            results.append(
                SearchResult(
                    title=item.get("full_name", ""),
                    url=item.get("html_url", ""),
                    snippet=item.get("description") or "",
                    extra={
                        "stars": item.get("stargazers_count"),
                        "language": item.get("language"),
                        "updated": item.get("pushed_at"),
                    },
                )
            )
        elif endpoint == "issues":
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("html_url", ""),
                    snippet=(item.get("body") or "")[:400],
                    extra={
                        "state": item.get("state"),
                        "is_pr": "pull_request" in item,
                        "comments": item.get("comments"),
                        "repo": (item.get("repository_url") or "").rsplit("repos/", 1)[-1],
                    },
                )
            )
        else:
            repo = (item.get("repository") or {}).get("full_name", "")
            results.append(
                SearchResult(
                    title=f"{repo}: {item.get('path', '')}",
                    url=item.get("html_url", ""),
                    snippet=item.get("name", ""),
                    extra={"repo": repo, "path": item.get("path")},
                )
            )
    return results


def _github_rest(
    query: str, *, max_results: int = 10, kind: str = "repositories", session: Any = None, **_: Any
) -> list[SearchResult]:
    if GITHUB_KINDS.get(kind.lower()) == "code" and not github_token():
        # Code search requires authentication; let the gh CLI try.

        raise BackendUnavailableError("GitHub code search needs GITHUB_TOKEN (or the gh CLI)")
    return _github_search(RestTransport(session=session), query, max_results=max_results, kind=kind)


def _github_gh(query: str, *, max_results: int = 10, kind: str = "repositories", **_: Any) -> list[SearchResult]:
    return _github_search(GhTransport(), query, max_results=max_results, kind=kind)


def _hackernews(
    query: str, *, max_results: int = 10, kind: str = "story", session: Any = None, **_: Any
) -> list[SearchResult]:
    params: dict[str, Any] = {"query": query, "hitsPerPage": max_results}
    if kind and kind != "all":
        params["tags"] = kind
    data = http_json("https://hn.algolia.com/api/v1/search", params=params, session=session, backend="hackernews")
    results = []
    for hit in data.get("hits", [])[:max_results]:
        oid = hit.get("objectID")
        hn_url = f"https://news.ycombinator.com/item?id={oid}"
        title = hit.get("title") or hit.get("story_title") or ""
        text = hit.get("story_text") or hit.get("comment_text") or ""

        results.append(
            SearchResult(
                title=title,
                url=hit.get("url") or hit.get("story_url") or hn_url,
                snippet=html_to_text(text)[:400] if text else "",
                extra={
                    "hn_url": hn_url,
                    "points": hit.get("points"),
                    "num_comments": hit.get("num_comments"),
                    "author": hit.get("author"),
                    "created_at": hit.get("created_at"),
                },
            )
        )
    return results


def _arxiv(query: str, *, max_results: int = 10, session: Any = None, **_: Any) -> list[SearchResult]:
    q = query if ":" in query else f"all:{query}"
    resp = http_get(
        ARXIV_API,
        params={"search_query": q, "start": 0, "max_results": max_results, "sortBy": "relevance"},
        session=session,
        backend="arxiv",
        timeout=30,
    )
    return [
        SearchResult(
            title=p["title"],
            url=p["url"],
            snippet=p["summary"][:400],
            extra={
                "authors": p["authors"],
                "published": p["published"],
                "categories": p["categories"],
                "pdf_url": p["pdf_url"],
            },
        )
        for p in parse_arxiv_atom(resp.content)[:max_results]
    ]


def _steps(platform: str) -> list[StepBackend]:
    def step(name: str, fn: Callable[..., list[SearchResult]], **kw: Any) -> StepBackend:
        return StepBackend(name, fn, **kw)

    if platform == "youtube":
        return [
            step(
                "yt_dlp",
                _yt_dlp_search,
                available=lambda: _common.binary_ok("yt-dlp"),
                requires=("yt-dlp",),
                hint=lambda: _common.binary_hint("yt-dlp"),
            ),
            step("web_search", _youtube_via_web),
        ]
    if platform == "github":
        return [
            step("rest", _github_rest),
            step(
                "gh",
                _github_gh,
                available=lambda: _common.binary_ok("gh"),
                requires=("gh",),
                hint=lambda: _common.binary_hint("gh"),
            ),
        ]
    if platform == "hackernews":
        return [step("algolia", _hackernews)]
    if platform == "arxiv":
        return [step("api", _arxiv)]
    raise ValueError(f"Unknown platform {platform!r}. Supported: {', '.join(PLATFORMS)}")


_ALIASES = {"yt": "youtube", "hn": "hackernews", "hacker_news": "hackernews", "gh": "github"}


def platform_chain(platform: str) -> BackendChain[list[SearchResult]]:
    ensure_error_rules()
    name = _ALIASES.get(platform.lower().strip(), platform.lower().strip())
    return BackendChain(
        _steps(name), override_env=f"PROMPTURE_PLATFORM_{name.upper()}_BACKENDS", name=f"search_platform:{name}"
    )


def platform_search(
    platform: str,
    query: str,
    *,
    max_results: int = 10,
    kind: str | None = None,
    session: requests.Session | None = None,
) -> SearchResponse:
    """Like :func:`search_platform` but returns a :class:`SearchResponse` with the route."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    max_results = max(1, min(int(max_results or 10), 50))
    chain = platform_chain(platform)
    kwargs: dict[str, Any] = {"max_results": max_results, "session": session}
    if kind:
        kwargs["kind"] = kind
    res = chain.run(query.strip(), **kwargs)
    results: list[SearchResult] = res.value
    for r in results:
        r.extra = {**(r.extra or {}), "served_by": res.served_by, "platform": chain.name.split(":", 1)[1]}
    return SearchResponse(
        query=query.strip(),
        results=results,
        served_by=f"{chain.name.split(':', 1)[1]}/{res.served_by}",
        route=res.route,
    )


def search_platform(
    platform: str,
    query: str,
    *,
    max_results: int = 10,
    kind: str | None = None,
    session: requests.Session | None = None,
) -> list[SearchResult]:
    """Search inside a platform.

    Args:
        platform: ``youtube``, ``github``, ``hackernews`` or ``arxiv``.
        query: Search query (arXiv accepts field syntax like ``ti:transformer``).
        max_results: Number of hits (1–50).
        kind: GitHub: ``repositories`` (default), ``issues`` or ``code``.
            Hacker News: ``story`` (default), ``comment`` or ``all``.
        session: Optional ``requests.Session``.
    """
    return platform_search(platform, query, max_results=max_results, kind=kind, session=session).results
