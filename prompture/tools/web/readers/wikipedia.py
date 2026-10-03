"""Wikipedia reader: ``/wiki/<Title>`` → summary + article Markdown.

Uses the public REST API (``/api/rest_v1/page/summary`` and ``/page/html``)
of the article's language edition. ``read_url(url, full=False)`` returns the
summary only.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from .._common import RequestRejectedError, host_of
from ..html2md import html_to_markdown
from .base import BaseReader, ReadResult, StepBackend, http_get, http_json

_HOST_RE = re.compile(r"^([a-z][a-z0-9\-]*)\.(?:m\.)?wikipedia\.org$")
_NAMESPACES = (
    "special:",
    "file:",
    "talk:",
    "user:",
    "user_talk:",
    "wikipedia:",
    "help:",
    "category:",
    "template:",
    "portal:",
    "draft:",
    "module:",
    "mediawiki:",
    "image:",
)
_REF_RE = re.compile(r"<sup\b[^>]*class=\"[^\"]*(?:mw-ref|reference)[^\"]*\"[^>]*>.*?</sup>", re.IGNORECASE | re.DOTALL)
_REFLIST_RE = re.compile(
    r"<(ol|div)\b[^>]*class=\"[^\"]*(?:mw-references|reflist|navbox)[^\"]*\"[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)


def wikipedia_title(url: str) -> tuple[str, str] | None:
    """``(language, title)`` for an article URL, else ``None``."""
    m = _HOST_RE.match(host_of(url))
    if not m:
        return None
    try:
        path = urlsplit(url).path
    except ValueError:
        return None
    if not path.startswith("/wiki/") or len(path) <= 6:
        return None
    title = unquote(path[6:])
    if title.lower().startswith(_NAMESPACES):
        return None
    return m.group(1), title


class WikipediaReader(BaseReader):
    """Wikipedia articles in any language edition."""

    name = "wikipedia"
    description = "Wikipedia articles → summary + Markdown"

    def can_handle(self, url: str) -> bool:
        return wikipedia_title(url) is not None

    def steps(self) -> list[StepBackend]:
        return [
            StepBackend(
                "rest",
                self._via_rest,
                live=lambda: http_json(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/Python_(programming_language)",
                    backend="wikipedia",
                ),
            )
        ]

    def _via_rest(self, url: str, *, session: Any = None, full: bool = True, **_: Any) -> ReadResult:
        parsed = wikipedia_title(url)
        if parsed is None:
            raise RequestRejectedError("rest", "not a Wikipedia article URL")
        lang, title = parsed
        base = f"https://{lang}.wikipedia.org/api/rest_v1/page"
        slug = quote(title.replace(" ", "_"), safe="()_,'!-.")
        summary = http_json(f"{base}/summary/{slug}", session=session, backend="wikipedia")
        display = summary.get("title") or title.replace("_", " ")
        parts = []
        if summary.get("description"):
            parts.append(f"_{summary['description']}_")
        if summary.get("extract"):
            parts.append(summary["extract"])
        page_url = ((summary.get("content_urls") or {}).get("desktop") or {}).get("page") or url
        if full and summary.get("type") != "disambiguation":
            html = http_get(f"{base}/html/{slug}", session=session, backend="wikipedia", timeout=30).text
            html = _REFLIST_RE.sub("", _REF_RE.sub("", html))
            article = html_to_markdown(html, base_url=page_url, use_trafilatura=False)
            if article:
                parts.append("## Article\n\n" + article)
        meta = {
            "language": lang,
            "page_title": display,
            "description": summary.get("description"),
            "type": summary.get("type"),
            "page_url": page_url,
            "thumbnail": (summary.get("thumbnail") or {}).get("source"),
            "last_modified": summary.get("timestamp"),
        }
        return ReadResult(url, display, "\n\n".join(parts), self.name, "article", meta)
