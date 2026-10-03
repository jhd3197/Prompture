"""AniList reader: anime / manga / character pages → structured cast Markdown.

anilist.co renders in the browser, so fetching the page returns an app shell.
This reader asks AniList's public GraphQL API instead (keyless, rate-limited
per IP). A title page (``/anime/<id>``, ``/manga/<id>``, with or without
``/characters``) becomes the title's details plus its characters in role
order, each with gender, age, description and voice actors; a
``/character/<id>`` page becomes that character with the titles it appears in.
``meta`` carries the same data as plain dicts for programmatic use.

Reader options: ``max_characters`` (default 50), ``spoilers`` (default
``False`` — AniList ``~!spoiler!~`` passages are dropped) and
``voice_language`` (``"JAPANESE"`` by default; ``None`` for every language).

:func:`search_anilist` finds titles by name, for callers that start from a
title rather than a URL.
"""

from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import urlsplit

import requests

from ....capabilities.http import proxies_for
from .._common import API_USER_AGENT, BackendUnavailableError, RequestRejectedError, default_session, host_of
from .._common import raise_for_status as _raise_for_status
from .base import BaseReader, ReadResult, StepBackend, clip

API = "https://graphql.anilist.co"
SITE = "https://anilist.co"
_PATH_RE = re.compile(r"^/(anime|manga|character)/(\d+)(?:/|$)")
_SPOILER_RE = re.compile(r"~!.*?!~", re.DOTALL)
_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
PAGE_SIZE = 25

_CHARACTER_FIELDS = """
  id siteUrl gender age
  name { full native alternative }
  image { medium }
  description(asHtml: false)
"""

_MEDIA_QUERY = (
    """
query ($id: Int, $page: Int, $perPage: Int, $lang: StaffLanguage) {
  Media(id: $id) {
    id type format status episodes chapters seasonYear siteUrl
    title { romaji english native }
    description(asHtml: false)
    characters(page: $page, perPage: $perPage, sort: [ROLE, RELEVANCE, ID]) {
      pageInfo { hasNextPage }
      edges {
        role
        node {"""
    + _CHARACTER_FIELDS
    + """}
        voiceActors(language: $lang) { name { full native } languageV2 }
      }
    }
  }
}
"""
)

_CHARACTER_QUERY = (
    """
query ($id: Int) {
  Character(id: $id) {"""
    + _CHARACTER_FIELDS
    + """
    media(perPage: 25, sort: [POPULARITY_DESC]) {
      edges {
        characterRole
        node { id type format seasonYear siteUrl title { romaji english native } }
        voiceActors { name { full native } languageV2 }
      }
    }
  }
}
"""
)

_SEARCH_QUERY = """
query ($search: String, $type: MediaType, $perPage: Int) {
  Page(perPage: $perPage) {
    media(search: $search, type: $type, sort: [SEARCH_MATCH]) {
      id type format episodes seasonYear siteUrl
      title { romaji english native }
      synonyms
    }
  }
}
"""


def anilist_id(url: str) -> tuple[str, int] | None:
    """``(kind, id)`` for an AniList title or character URL, else ``None``.

    *kind* is ``"anime"``, ``"manga"`` or ``"character"``.
    """
    if host_of(url) not in ("anilist.co", "www.anilist.co"):
        return None
    try:
        m = _PATH_RE.match(urlsplit(url).path)
    except ValueError:
        return None
    return (m.group(1), int(m.group(2))) if m else None


def graphql(query: str, variables: dict[str, Any], *, session: Any = None, timeout: float = 20.0) -> dict[str, Any]:
    """POST one query to the AniList API and return its ``data``."""
    sess = session or default_session()
    resp = sess.post(
        API,
        json={"query": query, "variables": variables},
        headers={"Accept": "application/json", "Content-Type": "application/json", "User-Agent": API_USER_AGENT},
        timeout=timeout,
        proxies=proxies_for("anilist"),
    )
    _raise_for_status(resp, "anilist")
    body = resp.json() or {}
    if body.get("errors"):
        message = str(body["errors"][0].get("message") or "unknown error")
        if "not found" in message.lower():
            raise RequestRejectedError("anilist", message)
        raise BackendUnavailableError(f"AniList API error: {message}")
    return body.get("data") or {}


def clean_description(text: str | None, *, spoilers: bool = False) -> str:
    """AniList description text without HTML entities and (by default) spoilers."""
    if not text:
        return ""
    if not spoilers:
        text = _SPOILER_RE.sub("", text)
    text = html.unescape(_TAG_RE.sub("\n", text.replace("~!", "").replace("!~", "")))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _title(media: dict[str, Any]) -> str:
    t = media.get("title") or {}
    return str(t.get("english") or t.get("romaji") or t.get("native") or "")


def _voices(entries: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    return [
        {
            "name": str((v.get("name") or {}).get("full") or ""),
            "native": str((v.get("name") or {}).get("native") or ""),
            "language": str(v.get("languageV2") or ""),
        }
        for v in entries or []
    ]


def _character(node: dict[str, Any], *, spoilers: bool) -> dict[str, Any]:
    name = node.get("name") or {}
    return {
        "id": node.get("id"),
        "name": str(name.get("full") or ""),
        "native": str(name.get("native") or ""),
        "alternative": [a for a in name.get("alternative") or [] if a],
        "gender": node.get("gender"),
        "age": node.get("age"),
        "description": clean_description(node.get("description"), spoilers=spoilers),
        "image": (node.get("image") or {}).get("medium"),
        "url": node.get("siteUrl"),
    }


def _character_lines(c: dict[str, Any], head: str) -> list[str]:
    facts = [f for f in (c.get("role"), c.get("gender"), f"age {c['age']}" if c.get("age") else None) if f]
    lines = [f"{head} {c['name']}" + (f" ({c['native']})" if c.get("native") else "")]
    if facts:
        lines.append("- " + " · ".join(str(f) for f in facts))
    if c.get("voice_actors"):
        lines.append(
            "- Voice: "
            + ", ".join(v["name"] + (f" ({v['language']})" if v["language"] else "") for v in c["voice_actors"])
        )
    if c.get("description"):
        lines.append("")
        lines.append(clip(c["description"], 600))
    lines.append("")
    return lines


def search_anilist(
    query: str,
    *,
    media_type: str | None = "ANIME",
    max_results: int = 5,
    session: Any = None,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """Titles matching *query*, best match first.

    Each item has ``id``, ``type``, ``format``, ``episodes``, ``year``,
    ``title`` (display), ``titles`` (romaji / english / native), ``synonyms``
    and ``url`` — pass the ``url`` to ``read_url`` for the cast.
    """
    variables: dict[str, Any] = {"search": query, "perPage": max(1, min(max_results, 25))}
    if media_type:
        variables["type"] = media_type.upper()
    data = graphql(_SEARCH_QUERY, variables, session=session, timeout=timeout)
    return [
        {
            "id": m.get("id"),
            "type": m.get("type"),
            "format": m.get("format"),
            "episodes": m.get("episodes"),
            "year": m.get("seasonYear"),
            "title": _title(m),
            "titles": m.get("title") or {},
            "synonyms": m.get("synonyms") or [],
            "url": m.get("siteUrl"),
        }
        for m in (data.get("Page") or {}).get("media") or []
    ]


class AniListReader(BaseReader):
    """AniList anime, manga and character pages via the public GraphQL API."""

    name = "anilist"
    description = "AniList titles and characters → cast with roles, gender, age and voice actors"

    def can_handle(self, url: str) -> bool:
        return anilist_id(url) is not None

    def steps(self) -> list[StepBackend]:
        return [
            StepBackend(
                "graphql",
                self._via_graphql,
                live=lambda: graphql("query { Media(id: 1) { id } }", {}),
            )
        ]

    def _via_graphql(
        self,
        url: str,
        *,
        session: requests.Session | None = None,
        max_characters: int = 50,
        spoilers: bool = False,
        voice_language: str | None = "JAPANESE",
        **_: Any,
    ) -> ReadResult:
        parsed = anilist_id(url)
        if parsed is None:
            raise RequestRejectedError("graphql", "not an AniList title or character URL")
        kind, item_id = parsed
        if kind == "character":
            return self._read_character(url, item_id, session=session, spoilers=spoilers)
        return self._read_media(
            url,
            item_id,
            session=session,
            max_characters=max_characters,
            spoilers=spoilers,
            voice_language=voice_language,
        )

    def _read_media(
        self,
        url: str,
        media_id: int,
        *,
        session: Any,
        max_characters: int,
        spoilers: bool,
        voice_language: str | None,
    ) -> ReadResult:
        media: dict[str, Any] = {}
        characters: list[dict[str, Any]] = []
        page_no = 1
        more = True
        while more and len(characters) < max_characters:
            variables: dict[str, Any] = {"id": media_id, "page": page_no, "perPage": PAGE_SIZE}
            if voice_language:
                variables["lang"] = voice_language.upper()
            media = graphql(_MEDIA_QUERY, variables, session=session).get("Media") or {}
            if not media:
                raise RequestRejectedError("graphql", f"AniList has no title {media_id}")
            block = media.get("characters") or {}
            for edge in block.get("edges") or []:
                c = _character(edge.get("node") or {}, spoilers=spoilers)
                c["role"] = edge.get("role")
                c["voice_actors"] = _voices(edge.get("voiceActors"))
                characters.append(c)
            more = bool((block.get("pageInfo") or {}).get("hasNextPage"))
            page_no += 1
        characters = characters[:max_characters]

        title = _title(media)
        facts = [media.get("format"), media.get("seasonYear")]
        if media.get("episodes"):
            facts.append(f"{media['episodes']} episodes")
        lines = ["_" + " · ".join(str(f) for f in facts if f) + "_", ""]
        about = clean_description(media.get("description"), spoilers=spoilers)
        if about:
            lines += [clip(about, 1200), ""]
        lines += [f"## Characters ({len(characters)}{'+' if more else ''})", ""]
        for c in characters:
            lines += _character_lines(c, "###")
        meta = {
            "id": media.get("id"),
            "type": media.get("type"),
            "format": media.get("format"),
            "status": media.get("status"),
            "episodes": media.get("episodes"),
            "chapters": media.get("chapters"),
            "year": media.get("seasonYear"),
            "titles": media.get("title") or {},
            "page_url": media.get("siteUrl") or url,
            "characters": characters,
            "more_characters": more,
        }
        return ReadResult(url, title, "\n".join(lines).strip(), self.name, "cast", meta)

    def _read_character(self, url: str, character_id: int, *, session: Any, spoilers: bool) -> ReadResult:
        node = graphql(_CHARACTER_QUERY, {"id": character_id}, session=session).get("Character") or {}
        if not node:
            raise RequestRejectedError("graphql", f"AniList has no character {character_id}")
        c = _character(node, spoilers=spoilers)
        appearances = []
        for edge in (node.get("media") or {}).get("edges") or []:
            m = edge.get("node") or {}
            appearances.append(
                {
                    "id": m.get("id"),
                    "title": _title(m),
                    "type": m.get("type"),
                    "format": m.get("format"),
                    "year": m.get("seasonYear"),
                    "role": edge.get("characterRole"),
                    "url": m.get("siteUrl"),
                    "voice_actors": _voices(edge.get("voiceActors")),
                }
            )
        lines = _character_lines(c, "##")[1:]  # the title already names the character
        if appearances:
            lines += ["## Appears in", ""]
            for a in appearances:
                facts = ", ".join(str(f) for f in (a["format"], a["year"], a["role"]) if f)
                lines.append(f"- {a['title']}" + (f" ({facts})" if facts else ""))
        meta = {**c, "page_url": c.get("url") or url, "appearances": appearances}
        return ReadResult(url, c["name"], "\n".join(lines).strip(), self.name, "character", meta)
