"""Safe HTTP GET for tools that fetch arbitrary URLs.

:func:`safe_get` follows redirects by hand so every hop goes back through
:func:`normalize_public_http_url`, streams the body and aborts past
``max_bytes``, detects challenge interstitials, and raises typed errors that
``classify_error`` understands. Proxies come from ``PROMPTURE_<BACKEND>_PROXY``
or ``PROMPTURE_PROXY``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import requests

from .challenge import is_challenge_page
from .errors import ChallengePageError, HTTPStatusError, ResponseTooLargeError, UnsafeURLError
from .url_safety import normalize_public_http_url

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
DEFAULT_HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_TIMEOUT = 15.0
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def resolve_proxy(backend: str | None = None) -> str | None:
    """Return the proxy URL for *backend*: ``PROMPTURE_<BACKEND>_PROXY`` then ``PROMPTURE_PROXY``."""
    if backend:
        specific = os.environ.get(f"PROMPTURE_{backend.upper().replace('-', '_')}_PROXY")
        if specific:
            return specific.strip()
    general = os.environ.get("PROMPTURE_PROXY")
    return general.strip() if general else None


def proxies_for(backend: str | None = None) -> dict[str, str] | None:
    """``requests``-style proxies mapping for *backend*, or ``None``."""
    proxy = resolve_proxy(backend)
    return {"http": proxy, "https": proxy} if proxy else None


@dataclass
class SafeResponse:
    """Body and metadata of a completed :func:`safe_get`."""

    url: str
    status_code: int
    headers: dict[str, str]
    content: bytes
    encoding: str | None = None
    history: list[str] = field(default_factory=list)
    truncated: bool = False

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";", 1)[0].strip().lower()

    @property
    def text(self) -> str:
        return self.content.decode(self.encoding or "utf-8", errors="replace")

    def json(self) -> Any:
        import json

        return json.loads(self.text)


def _encoding_of(resp: requests.Response) -> str | None:
    enc = resp.encoding
    ctype = resp.headers.get("content-type", "").lower()
    # requests defaults text/* without a charset to ISO-8859-1; UTF-8 is a better guess today.
    if enc and enc.lower() == "iso-8859-1" and "charset" not in ctype:
        return "utf-8"
    return enc


def safe_get(
    url: str,
    *,
    session: requests.Session | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_redirects: int = 5,
    allow_private: bool | None = None,
    check_challenge: bool = True,
    truncate: bool = False,
    backend: str | None = None,
    raise_for_status: bool = True,
) -> SafeResponse:
    """GET a public URL safely.

    Args:
        url: Target URL; validated with :func:`normalize_public_http_url`.
        session: Optional ``requests.Session`` (connection reuse / tests).
        headers: Extra headers merged over a browser-like default set.
        params: Query parameters for the first request.
        timeout: Per-request timeout in seconds.
        max_bytes: Body size cap. Exceeding it raises
            :class:`ResponseTooLargeError` unless ``truncate`` is set.
        max_redirects: Redirect hops allowed; each one is re-validated.
        allow_private: Permit private targets (default: env
            ``PROMPTURE_WEB_ALLOW_PRIVATE``).
        check_challenge: Raise :class:`ChallengePageError` on interstitials.
        truncate: Return the first ``max_bytes`` instead of raising.
        backend: Backend name used to pick a per-backend proxy.
        raise_for_status: Raise :class:`HTTPStatusError` on non-2xx.
    """
    http = session or requests.Session()
    merged = {**DEFAULT_HEADERS, **(headers or {})}
    proxies = proxies_for(backend)
    current = normalize_public_http_url(url, allow_private=allow_private)
    history: list[str] = []
    query = params

    for _hop in range(max_redirects + 1):
        resp = http.get(
            current,
            headers=merged,
            params=query,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
            proxies=proxies,
        )
        query = None
        try:
            if resp.status_code in _REDIRECT_STATUSES and resp.headers.get("location"):
                history.append(current)
                target = urljoin(current, resp.headers["location"])
                try:
                    current = normalize_public_http_url(target, allow_private=allow_private)
                except UnsafeURLError as exc:
                    raise UnsafeURLError(f"redirect to blocked target refused ({exc})") from exc
                continue

            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes and not truncate:
                raise ResponseTooLargeError(f"{declared} bytes exceeds the {max_bytes}-byte cap", limit=max_bytes)

            chunks: list[bytes] = []
            size = 0
            truncated = False
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                if size + len(chunk) > max_bytes:
                    if not truncate:
                        raise ResponseTooLargeError(f"body exceeds the {max_bytes}-byte cap", limit=max_bytes)
                    chunks.append(chunk[: max_bytes - size])
                    truncated = True
                    break
                chunks.append(chunk)
                size += len(chunk)
            body = b"".join(chunks)
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}

            if check_challenge and is_challenge_page(body, status=resp.status_code, headers=resp_headers):
                raise ChallengePageError(f"{current} returned a bot challenge", status_code=resp.status_code)
            if raise_for_status and not 200 <= resp.status_code < 300:
                snippet = body[:200].decode("utf-8", errors="replace").strip()
                raise HTTPStatusError(
                    f"{resp.reason or 'error'} fetching {current} {snippet}".strip(),
                    status_code=resp.status_code,
                    headers=resp_headers,
                    url=current,
                )
            return SafeResponse(
                url=current,
                status_code=resp.status_code,
                headers=resp_headers,
                content=body,
                encoding=_encoding_of(resp),
                history=history,
                truncated=truncated,
            )
        finally:
            resp.close()

    raise UnsafeURLError(f"too many redirects (>{max_redirects})")
