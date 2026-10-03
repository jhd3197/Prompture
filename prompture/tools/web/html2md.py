"""HTML → Markdown conversion with the standard library (``trafilatura`` when installed).

:func:`html_to_markdown` keeps headings, paragraphs, links, lists, code
blocks, quotes and simple tables, drops scripts/styles/navigation, and
prefers the page's ``<article>`` / ``<main>`` region when it holds real
content. :func:`scan_head` pulls the title, meta tags, ``<link>`` tags and
media sources used by the readers (feed discovery, podcast enclosures).
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

_SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "svg", "template", "iframe", "canvas", "form", "button", "select", "textarea"}
)
_CHROME_TAGS = frozenset({"nav", "footer", "aside"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "figure",
        "figcaption",
        "dl",
        "dt",
        "dd",
        "table",
        "details",
        "summary",
        "address",
        "hr",
    }
)
_VOID_TAGS = frozenset({"br", "hr", "img", "meta", "link", "input", "source", "wbr", "area", "base", "col", "embed"})
_WS_RE = re.compile(r"[ \t\r\n\f\v]+")
_FEED_TYPES = ("application/rss+xml", "application/atom+xml", "application/feed+json", "application/xml", "text/xml")


class _MarkdownConverter(HTMLParser):
    def __init__(self, base_url: str | None = None, drop_chrome: bool = True) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.drop_chrome = drop_chrome
        self.out: list[str] = []
        self.skip_depth = 0
        self.skip_tag = ""
        self.pre_depth = 0
        self.lists: list[list[Any]] = []  # [kind, counter]
        self.links: list[tuple[int, str | None]] = []
        self.quotes: list[int] = []
        self.row_cells = 0
        self.rows_in_table = 0
        self.header_row = False
        self.title = ""
        self._in_title = False

    # -- helpers -------------------------------------------------------
    def _emit(self, text: str) -> None:
        self.out.append(text)

    def _block(self) -> None:
        self.out.append("\n\n")

    def _line(self) -> None:
        self.out.append("\n")

    # -- parser hooks --------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
            return
        if self.skip_depth:
            if tag == self.skip_tag:
                self.skip_depth += 1
            return
        a = dict(attrs)
        hidden = "hidden" in a or (a.get("aria-hidden") or "").lower() == "true"
        if tag in _SKIP_TAGS or (self.drop_chrome and tag in _CHROME_TAGS) or hidden:
            if tag not in _VOID_TAGS:
                self.skip_tag = tag
                self.skip_depth = 1
            return
        if re.fullmatch(r"h[1-6]", tag):
            self._block()
            self._emit("#" * int(tag[1]) + " ")
        elif tag in _BLOCK_TAGS:
            self._block()
            if tag == "hr":
                self._emit("---")
                self._block()
        elif tag == "br":
            self._line()
        elif tag in ("ul", "ol"):
            if not self.lists:
                self._block()
            self.lists.append([tag, 0])
        elif tag == "li":
            self._line()
            depth = max(len(self.lists), 1)
            kind = self.lists[-1] if self.lists else ["ul", 0]
            kind[1] += 1
            marker = f"{kind[1]}. " if kind[0] == "ol" else "- "
            self._emit("  " * (depth - 1) + marker)
        elif tag == "a":
            href = a.get("href")
            if href and not href.startswith(("javascript:", "#", "mailto:")):
                href = urljoin(self.base_url, href) if self.base_url else href
            else:
                href = None
            self.links.append((len(self.out), href))
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("_")
        elif tag == "pre":
            self.pre_depth += 1
            self._block()
            self._emit("```\n")
        elif tag == "code" and not self.pre_depth:
            self._emit("`")
        elif tag == "blockquote":
            self._block()
            self.quotes.append(len(self.out))
        elif tag == "tr":
            self._line()
            self._emit("| ")
            self.row_cells = 0
            self.header_row = False
        elif tag in ("td", "th"):
            if tag == "th":
                self.header_row = True
        elif tag == "img":
            alt = (a.get("alt") or "").strip()
            if alt and len(alt) > 3:
                self._emit(f" {alt} ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            return
        if self.skip_depth:
            if tag == self.skip_tag:
                self.skip_depth -= 1
            return
        if re.fullmatch(r"h[1-6]", tag) or tag in _BLOCK_TAGS:
            self._block()
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._block() if not self.lists else self._line()
        elif tag == "a":
            if not self.links:
                return
            start, href = self.links.pop()
            text = _WS_RE.sub(" ", "".join(self.out[start:])).strip()
            if href and text and href.startswith(("http://", "https://")):
                del self.out[start:]
                self._emit(f"[{text}]({href})")
        elif tag in ("strong", "b"):
            self._emit("**")
        elif tag in ("em", "i"):
            self._emit("_")
        elif tag == "pre":
            self.pre_depth = max(0, self.pre_depth - 1)
            self._emit("\n```")
            self._block()
        elif tag == "code" and not self.pre_depth:
            self._emit("`")
        elif tag == "blockquote":
            if self.quotes:
                start = self.quotes.pop()
                text = "".join(self.out[start:]).strip()
                del self.out[start:]
                if text:
                    self._emit("\n".join("> " + line if line.strip() else ">" for line in text.splitlines()))
                self._block()
        elif tag in ("td", "th"):
            self._emit(" | ")
            self.row_cells += 1
        elif tag == "tr":
            self.rows_in_table += 1
            if self.header_row and self.row_cells:
                self._line()
                self._emit("|" + " --- |" * self.row_cells)
        elif tag == "table":
            self.rows_in_table = 0

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self.skip_depth:
            return
        if self.pre_depth:
            self._emit(data)
            return
        text = _WS_RE.sub(" ", data)
        if text.strip():
            self._emit(text)
        elif text and self.out and not self.out[-1].endswith((" ", "\n")):
            self._emit(" ")

    def markdown(self) -> str:
        lines = "".join(self.out).splitlines()
        cleaned: list[str] = []
        in_code = False
        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                cleaned.append(line.strip())
                continue
            if in_code:
                cleaned.append(line.rstrip())
                continue
            m = re.match(r"^( *)((?:[-*]|\d+\.) )", line)
            indent = m.group(1) if m else ""
            body = re.sub(r" {2,}", " ", line.strip())
            cleaned.append(indent + body)
        text = "\n".join(cleaned)
        text = re.sub(r"\*\*\s*\*\*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _main_region(html: str) -> str | None:
    """The ``<main>`` region, else the ``<article>`` region(s), or ``None``.

    Listing pages with several articles get all of them, in order.
    """
    mains = re.findall(r"<main\b[^>]*>.*?</main\s*>", html, flags=re.IGNORECASE | re.DOTALL)
    if mains:
        return max(mains, key=len)
    articles = re.findall(r"<article\b[^>]*>.*?</article\s*>", html, flags=re.IGNORECASE | re.DOTALL)
    if articles:
        return "\n".join(articles)
    return None


def _stdlib_markdown(html: str, base_url: str | None, drop_chrome: bool = True) -> tuple[str, str]:
    conv = _MarkdownConverter(base_url, drop_chrome=drop_chrome)
    try:
        conv.feed(html)
        conv.close()
    except Exception:  # malformed markup — keep what we have
        pass
    return conv.markdown(), html_lib.unescape(_WS_RE.sub(" ", conv.title)).strip()


def _trafilatura_markdown(html: str, base_url: str | None) -> str | None:
    try:
        import trafilatura  # type: ignore[import-not-found]
    except ImportError:
        return None
    for fmt in ("markdown", "txt"):
        try:
            out = trafilatura.extract(
                html,
                url=base_url,
                output_format=fmt,
                include_links=True,
                include_tables=True,
                include_comments=False,
            )
        except (TypeError, ValueError):
            continue
        except Exception:
            return None
        if out:
            return out.strip()
    return None


def html_to_markdown(html: str, *, base_url: str | None = None, use_trafilatura: bool = True) -> str:
    """Convert an HTML document to readable Markdown.

    Args:
        html: The HTML source.
        base_url: Used to absolutize relative links.
        use_trafilatura: Use ``trafilatura`` main-content extraction when it
            is installed (``pip install prompture[web]``).
    """
    if use_trafilatura:
        extracted = _trafilatura_markdown(html, base_url)
        if extracted and len(extracted) > 200:
            return extracted
    region = _main_region(html)
    if region:
        md, _ = _stdlib_markdown(region, base_url)
        if len(md) >= 200:
            return md
    md, _ = _stdlib_markdown(html, base_url)
    if len(md) < 80:  # chrome removal left nothing — keep everything
        md_all, _ = _stdlib_markdown(html, base_url, drop_chrome=False)
        if len(md_all) > len(md):
            return md_all
    return md


# ---------------------------------------------------------------------------
# <head> scanning
# ---------------------------------------------------------------------------


@dataclass
class PageInfo:
    """Title, meta tags, ``<link>`` tags and media sources of a page."""

    title: str = ""
    meta: dict[str, str] = field(default_factory=dict)
    links: list[dict[str, str]] = field(default_factory=list)
    media: list[str] = field(default_factory=list)


class _HeadScanner(HTMLParser):
    def __init__(self, base_url: str | None) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.info = PageInfo()
        self._in_title = False
        self._title_done = False

    def _abs(self, href: str) -> str:
        return urljoin(self.base_url, href) if self.base_url else href

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()
        if tag == "title" and not self._title_done:
            self._in_title = True
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and "content" in a and key not in self.info.meta:
                self.info.meta[key] = html_lib.unescape(a["content"]).strip()
        elif tag == "link" and a.get("href"):
            self.info.links.append(
                {
                    "rel": a.get("rel", "").lower(),
                    "type": a.get("type", "").lower(),
                    "href": self._abs(a["href"]),
                    "title": a.get("title", ""),
                }
            )
        elif tag in ("audio", "video", "source", "enclosure") and (a.get("src") or a.get("url")):
            self.info.media.append(self._abs(a.get("src") or a.get("url") or ""))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title" and self._in_title:
            self._in_title = False
            self._title_done = True

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.info.title += data


def scan_head(html: str, base_url: str | None = None) -> PageInfo:
    """Collect title, meta tags, link tags and media sources from *html*."""
    scanner = _HeadScanner(base_url)
    try:
        scanner.feed(html)
        scanner.close()
    except Exception:
        pass
    info = scanner.info
    info.title = _WS_RE.sub(" ", info.title).strip()
    if not info.title:
        info.title = info.meta.get("og:title") or info.meta.get("twitter:title") or ""
    return info


def extract_title(html: str) -> str:
    """Page title (``<title>``, else ``og:title``)."""
    return scan_head(html).title


def find_feed_links(html: str, base_url: str | None = None) -> list[str]:
    """Feed URLs advertised with ``<link rel="alternate" type="application/rss+xml">`` (or Atom/JSON)."""
    out: list[str] = []
    for link in scan_head(html, base_url).links:
        rels = link["rel"].split()
        if "alternate" in rels and link["type"] in _FEED_TYPES and link["href"] not in out:
            out.append(link["href"])
    return out


def html_to_text(fragment: str) -> str:
    """Plain text of a short HTML fragment (feed summaries, comments)."""
    if "<" not in fragment:
        return html_lib.unescape(fragment).strip()
    md, _ = _stdlib_markdown(fragment, None, drop_chrome=False)
    return md
