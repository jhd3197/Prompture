"""Result types shared by search, fetch, readers and platform search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


def _route_footer(served_by: str, route: dict[str, Any], verb: str = "served by") -> str:
    footer = f"_{verb} {served_by}"
    failed = [a.get("backend") for a in route.get("attempts", []) if a.get("status") == "error"]
    if route.get("fallback") and failed:
        footer += f" (fallback after {', '.join(dict.fromkeys(str(f) for f in failed))} failed)"
    return footer + "_"


@dataclass
class SearchResponse:
    """Search results plus the route that produced them.

    Attributes:
        query: The query as sent.
        results: Deduplicated hits.
        served_by: Backend that answered.
        route: ``{served_by, fallback, attempts[]}`` from the backend chain.
        answer: Synthesized answer when the backend provides one.
    """

    query: str
    results: list[SearchResult] = field(default_factory=list)
    served_by: str = ""
    route: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None

    def to_markdown(self, *, include_answer: bool = True, footer: bool = True) -> str:
        """Render as a numbered Markdown list with a ``served by <backend>`` footer."""
        if not self.results:
            text = f'No results for "{self.query}".'
            if footer and self.served_by:
                text += "\n\n" + _route_footer(self.served_by, self.route)
            return text
        lines: list[str] = []
        if include_answer and self.answer:
            lines.append(f"**Answer:** {self.answer}\n")
        lines.append(f'### Search results for "{self.query}"')
        for i, r in enumerate(self.results, 1):
            snippet = " ".join((r.snippet or "").split())
            if len(snippet) > 500:
                snippet = snippet[:497].rstrip() + "..."
            lines.append(f"{i}. **{r.title or r.url}** — <{r.url}>\n   {snippet}")
        if footer and self.served_by:
            lines.append("\n" + _route_footer(self.served_by, self.route))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()
