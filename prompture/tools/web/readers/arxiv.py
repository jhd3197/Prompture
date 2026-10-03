"""arXiv reader: abs / pdf URLs → title, authors, abstract (optionally full text).

Chain: ``api`` (``export.arxiv.org`` Atom API) ▸ ``web_fetch`` (the abs page).
``read_url(url, full_text=True)`` appends the paper body fetched through
:func:`web_fetch` (the Jina reader converts the PDF).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET  # nosec B405 - fixed trusted host, entities rejected
from typing import Any
from urllib.parse import urlsplit

from .._common import RequestRejectedError, host_of
from ..fetch import web_fetch
from .base import BaseReader, ReadResult, StepBackend, http_get

API_URL = "https://export.arxiv.org/api/query"
_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$")
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


def arxiv_id(url: str) -> str | None:
    """Paper id (with version when present) from abs / pdf / html URLs."""
    host = host_of(url).removeprefix("www.")
    if host not in ("arxiv.org", "export.arxiv.org"):
        return None
    try:
        path = urlsplit(url).path
    except ValueError:
        return None
    m = re.match(r"^/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?$", path)
    if not m:
        return None
    candidate = m.group(1)
    return candidate if _ID_RE.match(candidate) else None


def parse_arxiv_atom(data: bytes) -> list[dict[str, Any]]:
    """Parse an arXiv API Atom response into paper dicts."""
    if b"<!ENTITY" in data[:65536]:
        raise RequestRejectedError("arxiv", "response declares XML entities")
    root = ET.fromstring(data)  # nosec B314
    papers = []
    for entry in root.findall(f"{_ATOM}entry"):
        entry_id = (entry.findtext(f"{_ATOM}id") or "").strip()
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        if not entry_id or "api/errors" in entry_id or title == "Error":
            continue
        pdf = ""
        for link in entry.findall(f"{_ATOM}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf = link.get("href", "")
        papers.append(
            {
                "id": entry_id.rsplit("/abs/", 1)[-1],
                "url": entry_id,
                "title": title,
                "summary": " ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
                "authors": [
                    " ".join((a.findtext(f"{_ATOM}name") or "").split()) for a in entry.findall(f"{_ATOM}author")
                ],
                "published": (entry.findtext(f"{_ATOM}published") or "")[:10],
                "updated": (entry.findtext(f"{_ATOM}updated") or "")[:10],
                "categories": [c.get("term", "") for c in entry.findall(f"{_ATOM}category")],
                "primary_category": (
                    entry.find(f"{_ARXIV}primary_category").get("term", "")
                    if entry.find(f"{_ARXIV}primary_category") is not None
                    else ""
                ),
                "doi": (entry.findtext(f"{_ARXIV}doi") or "").strip() or None,
                "comment": (entry.findtext(f"{_ARXIV}comment") or "").strip() or None,
                "pdf_url": pdf,
            }
        )
    return papers


class ArxivReader(BaseReader):
    """arxiv.org abstract and PDF links."""

    name = "arxiv"
    description = "arXiv papers → metadata + abstract (full text optional)"

    def can_handle(self, url: str) -> bool:
        return arxiv_id(url) is not None

    def steps(self) -> list[StepBackend]:
        return [
            StepBackend(
                "api", self._via_api, live=lambda: http_get(API_URL, params={"id_list": "1706.03762"}, backend="arxiv")
            ),
            StepBackend("web_fetch", self._via_web_fetch),
        ]

    def _full_text(self, paper_id: str, session: Any) -> str:
        fr = web_fetch(f"https://arxiv.org/pdf/{paper_id}", max_chars=0, session=session)
        return fr.content

    def _via_api(self, url: str, *, session: Any = None, full_text: bool = False, **_: Any) -> ReadResult:
        pid = arxiv_id(url) or ""
        resp = http_get(
            API_URL, params={"id_list": pid, "max_results": 1}, session=session, backend="arxiv", timeout=30
        )
        papers = parse_arxiv_atom(resp.content)
        if not papers:
            raise RequestRejectedError("api", f"paper {pid} not found")
        p = papers[0]
        lines = [
            f"**Authors:** {', '.join(p['authors'])}",
            f"**Published:** {p['published']}"
            + (f" · updated {p['updated']}" if p["updated"] != p["published"] else ""),
            f"**Categories:** {', '.join(p['categories'])}",
        ]
        if p.get("doi"):
            lines.append(f"**DOI:** {p['doi']}")
        if p.get("comment"):
            lines.append(f"**Comment:** {p['comment']}")
        lines.append(f"**PDF:** <{p['pdf_url'] or f'https://arxiv.org/pdf/{pid}'}>")
        content = "\n".join(lines) + "\n\n## Abstract\n\n" + p["summary"]
        if full_text:
            content += "\n\n## Full text\n\n" + self._full_text(pid, session)
        return ReadResult(
            url,
            p["title"],
            content,
            self.name,
            "paper",
            {k: v for k, v in p.items() if k != "summary"} | {"abstract": p["summary"]},
        )

    def _via_web_fetch(self, url: str, *, session: Any = None, full_text: bool = False, **_: Any) -> ReadResult:
        pid = arxiv_id(url) or ""
        fr = web_fetch(f"https://arxiv.org/abs/{pid}", max_chars=0, session=session)
        content = fr.content
        if full_text:
            content += "\n\n## Full text\n\n" + self._full_text(pid, session)
        return ReadResult(url, fr.title, content, self.name, "paper", {"id": pid, "page_served_by": fr.served_by})
