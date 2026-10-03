"""Reader protocol, result type and helpers shared by the URL readers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import requests

from ....capabilities.backends import BackendChain, BaseBackend
from ....capabilities.health import HealthStatus
from ....capabilities.http import safe_get
from .._common import API_USER_AGENT, default_session, ensure_error_rules
from .._types import _route_footer


@dataclass
class ReadResult:
    """Content read from a URL by a specialized reader (or ``web_fetch``).

    Attributes:
        url: The URL that was read.
        title: Title of the item (video, repo, paper, thread, ...).
        content: Markdown content (one slice when paged).
        reader: Reader that produced it (``youtube``, ``github``, ``web_fetch``, ...).
        kind: Item kind (``video``, ``repo``, ``issue``, ``feed``, ``paper``, ``page``, ...).
        meta: Structured fields (ids, authors, dates, counts, entries).
        route: ``{served_by, reader, fallback, attempts[]}``.
        truncated: ``True`` when more content follows.
        next_start: ``start`` value for the next slice, if any.
        total_chars: Length of the full content.
    """

    url: str
    title: str
    content: str
    reader: str
    kind: str = "page"
    meta: dict[str, Any] = field(default_factory=dict)
    route: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    next_start: int | None = None
    total_chars: int = 0

    def to_markdown(self) -> str:
        head = f"# {self.title}\n\n" if self.title else ""
        served = self.route.get("served_by") or self.reader
        footer = "\n\n" + _route_footer(str(served), self.route, verb="read with") if served else ""
        return f"{head}Source: <{self.url}>\n\n{self.content}{footer}"

    def __str__(self) -> str:
        return self.to_markdown()


@runtime_checkable
class Reader(Protocol):
    """A URL-routed reader.

    ``can_handle`` must be cheap and offline (URL pattern only). ``read``
    returns a :class:`ReadResult` or raises. ``check`` reports health and is
    offline unless *live*.
    """

    name: str

    def can_handle(self, url: str) -> bool: ...

    def read(self, url: str, **kwargs: Any) -> ReadResult: ...

    def check(self, live: bool = False) -> HealthStatus: ...


class StepBackend(BaseBackend):
    """A reader step wrapped as a chain backend.

    Args:
        name: Step name used in route records (``rest``, ``gh``, ``yt_dlp``, ...).
        fn: ``fn(url, **kwargs) -> ReadResult``.
        available: Offline availability check.
        requires: Human-readable requirements.
        hint: Fix hint when unavailable (string or zero-argument callable).
        live: Callable for live health checks.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[..., ReadResult],
        *,
        available: Callable[[], bool] | None = None,
        requires: Sequence[str] = (),
        hint: str | Callable[[], str | None] | None = None,
        live: Callable[[], Any] | None = None,
        keyless: bool = True,
    ) -> None:
        self.name = name
        self._fn = fn
        self._available = available
        self.requires = tuple(requires)
        self._hint = hint
        self._live = live
        self.keyless = keyless

    def available(self) -> bool:
        return True if self._available is None else bool(self._available())

    def unavailable_hint(self) -> str | None:
        hint = self._hint() if callable(self._hint) else self._hint
        return hint or super().unavailable_hint()

    def run(self, url: str, **kwargs: Any) -> ReadResult:
        return self._fn(url, **kwargs)

    def live_check(self) -> None:
        if self._live is not None:
            self._live()


class BaseReader:
    """Convenience base: subclass, set ``name`` and implement ``can_handle`` + ``steps``.

    ``steps()`` returns the ordered :class:`StepBackend` list; ``read`` runs
    them as a :class:`BackendChain` (override env
    ``PROMPTURE_READER_<NAME>_BACKENDS``) and fills the route.
    """

    name = "reader"
    description = ""

    def can_handle(self, url: str) -> bool:  # pragma: no cover - abstract
        return False

    def steps(self) -> list[StepBackend]:  # pragma: no cover - abstract
        return []

    def chain(self) -> BackendChain[ReadResult]:
        ensure_error_rules()
        return BackendChain(
            self.steps(),
            override_env=f"PROMPTURE_READER_{self.name.upper()}_BACKENDS",
            name=f"read_url:{self.name}",
        )

    def available(self) -> bool:
        return self.chain().active_backend() is not None

    def read(self, url: str, **kwargs: Any) -> ReadResult:
        res = self.chain().run(url, **kwargs)
        result: ReadResult = res.value
        result.reader = self.name
        result.route = {**res.route, "reader": self.name}
        return result

    def check(self, live: bool = False) -> HealthStatus:
        return self.chain().check(live, category="tools")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


# ---------------------------------------------------------------------------
# HTTP helpers for fixed API hosts
# ---------------------------------------------------------------------------


def http_get(
    url: str,
    *,
    session: requests.Session | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
    backend: str | None = None,
    max_bytes: int = 8 * 1024 * 1024,
    check_challenge: bool = False,
    api: bool | None = None,
) -> Any:
    """``safe_get`` with reader-friendly defaults; returns the :class:`SafeResponse`.

    API calls (the default unless *check_challenge* is set, i.e. an ordinary
    page) send an identifying User-Agent instead of the browser one.
    """
    if api if api is not None else not check_challenge:
        headers = {"User-Agent": API_USER_AGENT, **(headers or {})}
    return safe_get(
        url,
        session=session or default_session(),
        params=params,
        headers=headers,
        timeout=timeout,
        backend=backend,
        max_bytes=max_bytes,
        check_challenge=check_challenge,
    )


def http_json(url: str, **kwargs: Any) -> Any:
    """GET *url* and decode JSON."""
    headers = {"Accept": "application/json", **(kwargs.pop("headers", None) or {})}
    return http_get(url, headers=headers, **kwargs).json()


def clip(text: str, limit: int) -> str:
    """Shorten *text* to *limit* characters on a word boundary."""
    text = " ".join(text.split()) if "\n" not in text else text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip() + "…"


def fmt_timestamp(seconds: float) -> str:
    """``mm:ss`` or ``h:mm:ss``."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"
