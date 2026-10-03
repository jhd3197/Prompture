"""Hacker News reader: item pages → story + comment thread.

Chain: ``firebase`` (official ``hacker-news.firebaseio.com`` API; top-level
comments fetched in parallel) ▸ ``algolia`` (``hn.algolia.com`` items API,
whole tree in one request).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .._common import RequestRejectedError, host_of
from ..html2md import html_to_text
from .base import BaseReader, ReadResult, StepBackend, http_json

FIREBASE = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
ALGOLIA_ITEM = "https://hn.algolia.com/api/v1/items/{id}"


def hn_item_id(url: str) -> int | None:
    """Item id from ``news.ycombinator.com/item?id=N``."""
    if host_of(url) not in ("news.ycombinator.com", "ycombinator.com"):
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.path.rstrip("/") != "/item":
        return None
    raw = (parse_qs(parts.query).get("id") or [""])[0]
    return int(raw) if raw.isdigit() else None


def _render(story: dict[str, Any], comments: list[dict[str, Any]], item_id: int) -> tuple[str, str, dict[str, Any]]:
    title = story.get("title") or f"HN item {item_id}"
    head = [
        f"**{story.get('type', 'story')}** by {story.get('by') or story.get('author') or 'unknown'} · "
        f"{story.get('score', story.get('points', 0)) or 0} points · {story.get('descendants', len(comments))} comments",
    ]
    if story.get("url"):
        head.append(f"Link: <{story['url']}>")
    body = html_to_text(story.get("text") or "") if story.get("text") else ""
    parts = ["\n".join(head)]
    if body:
        parts.append(body)
    if comments:
        parts.append("## Comments")
        for c in comments:
            indent = "> " * c.get("depth", 0)
            text = html_to_text(c.get("text") or "")
            lines = [f"{indent}**{c.get('by') or 'unknown'}**:"] + [
                f"{indent}{line}" if line else indent.rstrip() for line in text.splitlines()
            ]
            parts.append("\n".join(lines))
    meta = {
        "item_id": item_id,
        "by": story.get("by") or story.get("author"),
        "score": story.get("score", story.get("points")),
        "link": story.get("url"),
        "comment_count": story.get("descendants"),
        "comments_shown": len(comments),
        "hn_url": f"https://news.ycombinator.com/item?id={item_id}",
    }
    return title, "\n\n".join(parts), meta


class HackerNewsReader(BaseReader):
    """news.ycombinator.com item pages."""

    name = "hackernews"
    description = "Hacker News threads → story + comments"

    def can_handle(self, url: str) -> bool:
        return hn_item_id(url) is not None

    def steps(self) -> list[StepBackend]:
        return [
            StepBackend(
                "firebase", self._via_firebase, live=lambda: http_json(FIREBASE.format(id=1), backend="hackernews")
            ),
            StepBackend("algolia", self._via_algolia),
        ]

    def _via_firebase(
        self, url: str, *, session: Any = None, max_comments: int = 30, replies: int = 2, **_: Any
    ) -> ReadResult:
        item_id = hn_item_id(url)
        if item_id is None:
            raise RequestRejectedError("firebase", "not an HN item URL")
        story = http_json(FIREBASE.format(id=item_id), session=session, backend="hackernews")
        if not story:
            raise RequestRejectedError("firebase", f"item {item_id} not found")

        def get(cid: int) -> dict[str, Any] | None:
            try:
                return http_json(FIREBASE.format(id=cid), session=session, backend="hackernews")
            except Exception:
                return None

        kids = list(story.get("kids") or [])[:max_comments]
        with ThreadPoolExecutor(max_workers=8) as pool:
            top = list(pool.map(get, kids))
            reply_ids = [(i, k) for i, c in enumerate(top) if c for k in (c.get("kids") or [])[:replies]]
            reply_items = list(pool.map(get, [k for _, k in reply_ids]))
        replies_by_parent: dict[int, list[dict[str, Any]]] = {}
        for (parent_idx, _), item in zip(reply_ids, reply_items, strict=False):
            if item and not item.get("deleted") and not item.get("dead"):
                replies_by_parent.setdefault(parent_idx, []).append({**item, "depth": 1})
        comments: list[dict[str, Any]] = []
        for i, c in enumerate(top):
            if not c or c.get("deleted") or c.get("dead"):
                continue
            comments.append({**c, "depth": 0})
            comments.extend(replies_by_parent.get(i, []))
        title, content, meta = _render(story, comments, item_id)
        return ReadResult(url, title, content, self.name, "thread", meta)

    def _via_algolia(
        self, url: str, *, session: Any = None, max_comments: int = 30, replies: int = 2, **_: Any
    ) -> ReadResult:
        item_id = hn_item_id(url)
        if item_id is None:
            raise RequestRejectedError("algolia", "not an HN item URL")
        data = http_json(ALGOLIA_ITEM.format(id=item_id), session=session, backend="hackernews")
        comments: list[dict[str, Any]] = []
        for child in (data.get("children") or [])[:max_comments]:
            if child.get("text"):
                comments.append({"by": child.get("author"), "text": child.get("text"), "depth": 0})
            for reply in (child.get("children") or [])[:replies]:
                if reply.get("text"):
                    comments.append({"by": reply.get("author"), "text": reply.get("text"), "depth": 1})
        story = {
            "title": data.get("title"),
            "by": data.get("author"),
            "score": data.get("points"),
            "url": data.get("url"),
            "text": data.get("text"),
            "type": data.get("type"),
            "descendants": len(data.get("children") or []),
        }
        title, content, meta = _render(story, comments, item_id)
        return ReadResult(url, title, content, self.name, "thread", meta)
