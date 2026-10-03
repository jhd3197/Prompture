"""Fetch a public URL as Markdown with failover, paging and a short cache.

:func:`web_fetch` validates the URL with
:func:`~prompture.capabilities.normalize_public_http_url` *before* any
backend sees it (a private address is never handed to a third-party
reader), then runs the fetch chain:

* ``jina_reader`` — ``r.jina.ai`` (keyless; ``JINA_API_KEY`` raises limits).
  Handles PDFs and script-rendered pages.
* ``direct`` — :func:`~prompture.capabilities.safe_get` plus HTML→Markdown
  (``trafilatura`` when installed, stdlib otherwise).

``PROMPTURE_FETCH_BACKENDS="direct,jina_reader"`` reorders the chain. A
challenge/captcha interstitial moves on to the next backend. Long pages are
returned in slices ending with ``[truncated — call again with start=N]``;
fetched pages go through the local web cache (:mod:`.cache`), so paging and
repeat reads don't refetch.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

from ...capabilities.backends import BackendChain, BaseBackend
from ...capabilities.challenge import is_challenge_page
from ...capabilities.errors import ChallengePageError, HTTPStatusError
from ...capabilities.http import safe_get
from ...capabilities.url_safety import normalize_public_http_url
from . import cache as web_cache
from ._common import (
    API_USER_AGENT,
    RequestRejectedError,
    config_value,
    default_session,
    ensure_error_rules,
    page,
)
from ._types import _route_footer
from .html2md import html_to_markdown, scan_head

FETCH_OVERRIDE_ENV = "PROMPTURE_FETCH_BACKENDS"
JINA_READER_URL = "https://r.jina.ai/"
DEFAULT_MAX_BYTES = 8 * 1024 * 1024


# Page titles that only interstitials use.
_CHALLENGE_TITLES = frozenset(
    {"just a moment...", "attention required! | cloudflare", "access denied", "security check", "are you a robot?"}
)


@dataclass
class FetchedPage:
    """Full page content produced by a fetch backend (before paging)."""

    url: str
    final_url: str
    title: str
    content: str
    content_type: str


@dataclass
class FetchResult:
    """One page (or one slice of a long page) fetched as Markdown.

    Attributes:
        url: URL as requested (normalized).
        final_url: URL after redirects.
        title: Page title when known.
        content: Markdown content of this slice.
        content_type: Media type of the source (``text/html``, ``application/pdf``, ...).
        served_by: Backend that fetched it.
        route: ``{served_by, fallback, attempts[]}``.
        truncated: ``True`` when more content follows.
        next_start: ``start`` value for the next slice, if any.
        total_chars: Length of the full Markdown content.
    """

    url: str
    final_url: str
    title: str
    content: str
    content_type: str = "text/html"
    served_by: str = ""
    route: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    next_start: int | None = None
    total_chars: int = 0
    cached: bool = False

    def to_markdown(self) -> str:
        head = f"# {self.title}\n\n" if self.title else ""
        footer = "\n\n" + _route_footer(self.served_by, self.route) if self.served_by else ""
        return f"{head}Source: <{self.final_url}>\n\n{self.content}{footer}"

    def __str__(self) -> str:
        return self.to_markdown()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class FetchBackend(BaseBackend):
    """Base class for fetch backends; ``run(url)`` returns a :class:`FetchedPage`."""

    category = "tools"

    def __init__(self, *, session: requests.Session | None = None, timeout: float = 20.0) -> None:
        self._session = session
        self.timeout = timeout

    @property
    def session(self) -> requests.Session:
        return self._session or default_session()

    def run(self, url: str) -> FetchedPage:  # pragma: no cover - abstract
        raise NotImplementedError

    def live_check(self) -> None:
        self.run("https://example.com/")


class JinaReaderBackend(FetchBackend):
    """``r.jina.ai`` reader: keyless, renders JS pages and PDFs to Markdown."""

    name = "jina_reader"
    keyless = True
    requires = ()

    def key(self) -> str | None:
        return config_value("jina_api_key", "JINA_API_KEY")

    def run(self, url: str) -> FetchedPage:
        headers = {"Accept": "application/json", "X-Return-Format": "markdown", "User-Agent": API_USER_AGENT}
        key = self.key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            resp = safe_get(
                JINA_READER_URL + url,
                session=self.session,
                headers=headers,
                timeout=max(self.timeout, 30.0),
                max_bytes=DEFAULT_MAX_BYTES,
                backend=self.name,
                check_challenge=False,
            )
        except HTTPStatusError as exc:
            # 4xx for this target (blocked, unprocessable) — another backend may still work.
            if exc.status_code in (400, 404, 409, 410, 422, 451):
                raise RequestRejectedError(self.name, f"status {exc.status_code}") from exc
            raise
        try:
            body = resp.json()
        except ValueError:
            body = None
        if isinstance(body, dict) and isinstance(body.get("data"), dict):
            data = body["data"]
            content = str(data.get("content") or "")
            title = str(data.get("title") or "")
            final_url = str(data.get("url") or url)
        else:
            content, title, final_url = _parse_jina_text(resp.text, url)
        if not content.strip():
            raise RequestRejectedError(self.name, "empty content")
        if is_challenge_page(content) or title.strip().lower() in _CHALLENGE_TITLES:
            raise ChallengePageError(f"{url} returned a bot challenge via {self.name}")
        ctype = "application/pdf" if url.lower().split("?", 1)[0].endswith(".pdf") else "text/html"
        return FetchedPage(url=url, final_url=final_url, title=title, content=content.strip(), content_type=ctype)


def _parse_jina_text(text: str, url: str) -> tuple[str, str, str]:
    """Parse Jina's plain-text format (``Title:``, ``URL Source:``, ``Markdown Content:``)."""
    title, final_url = "", url
    marker = "Markdown Content:"
    head, sep, body = text.partition(marker)
    if not sep:
        return text, title, final_url
    for line in head.splitlines():
        if line.startswith("Title:"):
            title = line[6:].strip()
        elif line.startswith("URL Source:"):
            final_url = line[11:].strip() or url
    return body.strip(), title, final_url


class DirectBackend(FetchBackend):
    """Fetch the page ourselves with :func:`safe_get` and convert it locally."""

    name = "direct"
    keyless = True
    requires = ()

    def run(self, url: str) -> FetchedPage:
        resp = safe_get(url, session=self.session, timeout=self.timeout, max_bytes=DEFAULT_MAX_BYTES, backend=self.name)
        ctype = resp.content_type or "text/html"
        if ctype in ("text/html", "application/xhtml+xml") or (
            not resp.content_type and b"<html" in resp.content[:2048].lower()
        ):
            html = resp.text
            info = scan_head(html, resp.url)
            content = html_to_markdown(html, base_url=resp.url)
            return FetchedPage(url, resp.url, info.title, content, "text/html")
        if ctype == "application/json" or ctype.endswith("+json"):
            try:
                content = "```json\n" + json.dumps(resp.json(), indent=2, ensure_ascii=False)[:2_000_000] + "\n```"
            except ValueError:
                content = resp.text
            return FetchedPage(url, resp.url, "", content, ctype)
        if ctype.startswith("text/") or ctype.endswith("+xml") or ctype in ("application/xml",):
            return FetchedPage(url, resp.url, "", resp.text, ctype)
        if ctype == "application/pdf":
            text = _pdf_text(resp.content)
            if text is None:
                raise RequestRejectedError(self.name, "PDF extraction needs `pip install pypdf`")
            return FetchedPage(url, resp.url, "", text, ctype)
        raise RequestRejectedError(self.name, f"content type {ctype!r} is not text")


def _pdf_text(data: bytes) -> str | None:
    try:
        import io

        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "").strip() for p in reader.pages).strip()
    except Exception:
        return None


FETCH_BACKENDS: dict[str, type[FetchBackend]] = {"jina_reader": JinaReaderBackend, "direct": DirectBackend}
DEFAULT_FETCH_ORDER = ("jina_reader", "direct")


def fetch_chain(
    *,
    session: requests.Session | None = None,
    timeout: float = 20.0,
    backends: Sequence[FetchBackend] | None = None,
) -> BackendChain[FetchedPage]:
    """The default fetch chain (``jina_reader`` ▸ ``direct``)."""
    ensure_error_rules()
    if backends is None:
        backends = [FETCH_BACKENDS[n](session=session, timeout=timeout) for n in DEFAULT_FETCH_ORDER]
    return BackendChain(list(backends), override_env=FETCH_OVERRIDE_ENV, name="web_fetch")


def _compress(text: str) -> str:
    from ...infra.compression import compress_messages

    out, _ = compress_messages([{"role": "tool", "content": text}], max_tool_chars=len(text) + 1)
    return str(out[0]["content"])


def clear_fetch_cache() -> None:
    """Drop the local web cache (pages, searches and reader results)."""
    web_cache.clear_web_cache()


def web_fetch(
    url: str,
    *,
    max_chars: int = 20000,
    start: int = 0,
    compress: bool = False,
    backends: Sequence[str] | None = None,
    session: requests.Session | None = None,
    timeout: float = 20.0,
    use_cache: bool = True,
    cache_ttl: float | None = None,
) -> FetchResult:
    """Fetch *url* and return its content as Markdown.

    Args:
        url: Public http(s) URL. Private / loopback / metadata targets are
            refused before any backend is called.
        max_chars: Characters per slice (``0`` = everything).
        start: Offset into the Markdown content (for paging).
        compress: Collapse blank-line runs and trailing whitespace
            (:mod:`prompture.infra.compression`).
        backends: Restrict to these backend names, in this order.
        session: Optional ``requests.Session``.
        timeout: Per-request timeout in seconds.
        use_cache: Reuse a recently fetched copy from the local web cache.
            Lifetime depends on the URL: 10 minutes for fast-moving sites,
            1 hour for ordinary pages, days for papers and PDFs.
        cache_ttl: Override the cache lifetime in seconds (``0`` = don't store).

    Raises:
        UnsafeURLError: The URL is not a public http(s) address.
        AllBackendsFailedError: Every backend failed.
    """
    safe_url = normalize_public_http_url(url)
    only: list[str] | None = None
    if backends is not None:
        only = [b.strip().lower() for b in ([backends] if isinstance(backends, str) else backends) if b.strip()]
        unknown = [b for b in only if b not in FETCH_BACKENDS]
        if unknown:
            raise ValueError(f"Unknown fetch backend(s) {unknown}. Known: {', '.join(FETCH_BACKENDS)}")
    key = web_cache.make_key("fetch", safe_url, only)
    stored = web_cache.get(key) if use_cache else None
    cached = stored is not None
    if stored is not None:
        value, age = stored
        fetched = FetchedPage(**value["page"])
        served_by, route = value["served_by"], web_cache.mark_cached(value["route"], age)
    else:
        res = fetch_chain(session=session, timeout=timeout).run(safe_url, only=only)
        fetched, served_by, route = res.value, res.served_by, res.route
        if use_cache and fetched.content:
            ttl = cache_ttl if cache_ttl is not None else web_cache.page_ttl(fetched.final_url or safe_url)
            web_cache.put(key, {"page": asdict(fetched), "served_by": served_by, "route": route}, ttl)
    content = _compress(fetched.content) if compress else fetched.content
    piece, truncated, next_start, total = page(content, start=start, max_chars=max_chars)
    return FetchResult(
        url=safe_url,
        final_url=fetched.final_url,
        title=fetched.title,
        content=piece,
        content_type=fetched.content_type,
        served_by=served_by,
        route=route,
        truncated=truncated,
        next_start=next_start,
        total_chars=total,
        cached=cached,
    )


async def afetch(url: str, **kwargs: Any) -> FetchResult:
    """Async :func:`web_fetch` (runs in a worker thread)."""
    return await asyncio.to_thread(web_fetch, url, **kwargs)
