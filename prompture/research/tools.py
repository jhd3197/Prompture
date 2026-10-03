"""Injectable gathering functions for :class:`~prompture.research.ResearchAgent`.

:class:`ResearchTools` bundles every outside call the agent makes — web
search, platform search, page reading, transcription and domain packs. Each
field defaults to ``None``, which binds lazily to the real implementation in
:mod:`prompture.tools.web`, :mod:`prompture.media.understand` or
:mod:`prompture.tools.packs` on first use. Pass your own callables to swap a
backend or to test without network::

    tools = ResearchTools(
        search=lambda q, **kw: my_search(q),
        read=lambda url, **kw: my_reader(url),
    )
    ResearchAgent("openai/gpt-4o-mini", tools=tools).run("...")

Signatures follow the web-tools contract:

* ``search(query, *, max_results, ...) -> SearchResponse`` (``.results``,
  ``.served_by``, ``.route``) — a bare list of results is accepted too.
* ``search_platform(platform, query, *, max_results) -> list[SearchResult]``.
* ``read(url) -> ReadResult`` (``.content``, ``.title``, ``.reader``, ``.kind``).
* ``fetch(url, *, max_chars) -> FetchResult`` (``.content``, ``.served_by``).
* ``transcribe(source) -> Transcript`` (``.text`` / ``.to_markdown()``).
* ``pack_tools(name) -> list[ToolDefinition]``.
* ``pack_runner(pack, question) -> (text, usage)`` — replaces the default
  runner, which lets a small tool-calling :class:`~prompture.agents.Agent`
  use the pack's tools to gather facts for *question*.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ResearchToolUnavailable(RuntimeError):
    """A gathering capability isn't installed or importable."""


def _load(module: str, attr: str) -> Callable[..., Any]:
    try:
        return getattr(importlib.import_module(module), attr)  # type: ignore[no-any-return]
    except (ImportError, AttributeError) as exc:
        raise ResearchToolUnavailable(f"{module}.{attr} is not available: {exc}") from exc


#: Platforms :func:`search_platform` supports.
PLATFORMS = ("github", "hackernews", "arxiv", "youtube")

#: Domain packs the agent may consult.
PACKS = ("finance", "news", "dev", "places")


@dataclass
class ResearchTools:
    """Gathering functions; ``None`` fields bind to the built-in implementations."""

    search: Callable[..., Any] | None = None
    search_platform: Callable[..., Any] | None = None
    read: Callable[..., Any] | None = None
    fetch: Callable[..., Any] | None = None
    transcribe: Callable[..., Any] | None = None
    transcription_available: Callable[[], bool] | None = None
    pack_tools: Callable[[str], list[Any]] | None = None
    pack_runner: Callable[[str, str], tuple[str, dict[str, Any]]] | None = None
    enable_platforms: bool = True
    enable_packs: bool = True
    enable_transcription: bool = True

    # -- calls --------------------------------------------------------------

    def do_search(self, query: str, **kwargs: Any) -> Any:
        fn = self.search or _load("prompture.tools.web", "web_search")
        return fn(query, **kwargs)

    def do_search_platform(self, platform: str, query: str, **kwargs: Any) -> Any:
        if not self.enable_platforms:
            raise ResearchToolUnavailable("platform search disabled")
        fn = self.search_platform or _load("prompture.tools.web", "search_platform")
        return fn(platform, query, **kwargs)

    def do_read(self, url: str) -> Any:
        """Open *url* with the routed reader, falling back to ``fetch`` when needed."""
        if self.read is not None:
            return self.read(url)
        if self.fetch is not None:
            return self.fetch(url)
        try:
            fn = _load("prompture.tools.web", "read_url")
        except ResearchToolUnavailable:
            fn = _load("prompture.tools.web", "web_fetch")
        return fn(url)

    def do_fetch(self, url: str, **kwargs: Any) -> Any:
        fn = self.fetch or _load("prompture.tools.web", "web_fetch")
        return fn(url, **kwargs)

    def can_transcribe(self) -> bool:
        if not self.enable_transcription:
            return False
        if self.transcribe is not None:
            return True if self.transcription_available is None else bool(self.transcription_available())
        try:
            available = self.transcription_available or _load("prompture.media.understand", "transcription_available")
            return bool(available())
        except Exception:
            return False

    def do_transcribe(self, source: str) -> Any:
        fn = self.transcribe or _load("prompture.media.understand", "transcribe")
        return fn(source)

    def do_pack_tools(self, name: str) -> list[Any]:
        if not self.enable_packs:
            raise ResearchToolUnavailable("domain packs disabled")
        fn = self.pack_tools or _load("prompture.tools.packs", "resolve_pack_tools")
        return list(fn(name))


__all__ = ["PACKS", "PLATFORMS", "ResearchToolUnavailable", "ResearchTools"]
