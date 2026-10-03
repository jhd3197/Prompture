"""Shared helpers for the web tools: config lookup, sessions, caching, subprocess.

Nothing here talks to the network on import. Keys are read on every call
(environment first, then :data:`prompture.infra.settings.settings`) so a key
added at runtime is picked up without a restart.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from ...capabilities.errors import BackendUnavailableError, CapabilityError
from ...capabilities.probe import cached_probe, install_hint, probe_env
from ...infra.settings import settings
from ...security.redaction import scrub_secrets

DEFAULT_TIMEOUT = 15.0

# Identifying User-Agent for official APIs (Wikimedia, GitHub, Jina, ...).
# A browser UA is reserved for fetching ordinary pages.
API_USER_AGENT = "prompture-web/1.0 (+https://pypi.org/project/prompture/)"

_session: requests.Session | None = None
_session_lock = threading.Lock()


def ensure_error_rules() -> None:
    """Make sure the capability error markers are registered with ``classify_error``.

    Idempotent and cheap; called whenever a chain is built so a
    ``reset_error_rules()`` elsewhere cannot turn an unsafe URL into a
    plain failover.
    """
    from ...capabilities.errors import register_capability_error_rules

    register_capability_error_rules()


def config_value(attr: str | None, env: str | None = None) -> str | None:
    """Return a configured value: env var *env* first, then ``settings.<attr>``."""
    if env:
        value = os.environ.get(env)
        if value and value.strip():
            return value.strip()
    if attr:
        value = getattr(settings, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def default_session() -> requests.Session:
    """Process-wide ``requests.Session`` used when the caller passes none."""
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
        return _session


def error_text(exc: BaseException) -> str:
    """Scrubbed one-line description of *exc* for tool results."""
    msg = str(exc).strip() or type(exc).__name__
    return scrub_secrets(f"{type(exc).__name__}: {msg}")[:800]


class RequestRejectedError(CapabilityError):
    """A backend refused this particular request (4xx that another backend may accept).

    The message says "does not support", which ``classify_error`` maps to
    *failover* instead of the fatal ``bad_request`` action a bare 400 gets.
    """

    def __init__(self, backend: str, detail: str) -> None:
        super().__init__(f"{backend} does not support this request ({detail})")


def raise_for_status(resp: Any, backend: str, *, reject: Sequence[int] = (400, 404, 422)) -> None:
    """``resp.raise_for_status()`` with per-request rejections mapped to failover."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int) and status in reject:
            raise RequestRejectedError(backend, f"status {status}") from exc
        raise


def header(resp: Any, name: str) -> str | None:
    """Case-insensitive header lookup that also works on plain-dict fakes."""
    headers = getattr(resp, "headers", None) or {}
    try:
        value = headers.get(name)
        if value is None:
            value = headers.get(name.lower())
        if value is None:
            for k, v in headers.items():
                if str(k).lower() == name.lower():
                    return str(v)
        return None if value is None else str(value)
    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# TTL cache
# ---------------------------------------------------------------------------


class TTLCache:
    """Tiny thread-safe TTL + size bounded cache."""

    def __init__(self, ttl: float = 600.0, maxsize: int = 128) -> None:
        self.ttl = ttl
        self.maxsize = maxsize
        self._data: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        now = time.monotonic()
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            if now - hit[0] > self.ttl:
                self._data.pop(key, None)
                return None
            return hit[1]

    def set(self, key: Any, value: Any) -> None:
        with self._lock:
            if len(self._data) >= self.maxsize and key not in self._data:
                oldest = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest, None)
            self._data[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# URLs and domains
# ---------------------------------------------------------------------------

_TRACKING_PARAMS = re.compile(r"^(utm_[a-z]+|fbclid|gclid|mc_cid|mc_eid|ref_src|igshid)$", re.IGNORECASE)


def dedupe_key(url: str) -> str:
    """Normalize *url* for duplicate detection (case, ``www.``, slash, tracking params)."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)))
    return urlunsplit(("", host, path, query, ""))


def host_of(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def clean_domain(domain: str) -> str:
    d = domain.strip().lower()
    if "://" in d:
        d = host_of(d)
    return d.split("/", 1)[0].removeprefix("www.").lstrip(".")


def domain_matches(url: str, domains: Sequence[str]) -> bool:
    """True when the host of *url* equals or is a subdomain of any of *domains*."""
    host = host_of(url).removeprefix("www.")
    if not host:
        return False
    for raw in domains:
        d = clean_domain(raw)
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def parse_date(value: Any) -> datetime | None:
    """Best-effort parse of ISO-8601 / RFC-822 / epoch dates to an aware datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    iso = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        dt = None
    if dt is None:
        from email.utils import parsedate_to_datetime

        try:
            dt = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            dt = None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Subprocess (no shell)
# ---------------------------------------------------------------------------


class CommandError(CapabilityError):
    """An external command failed. ``status_code`` stays unset so routing uses the text."""


def run_command(argv: Sequence[str], *, timeout: float = 60.0, max_output: int = 20 * 1024 * 1024) -> str:
    """Run *argv* without a shell and return stdout as text.

    Raises :class:`BackendUnavailableError` when the binary is missing and
    :class:`CommandError` on a non-zero exit or timeout.
    """
    if not argv:
        raise ValueError("empty command")
    path = shutil.which(argv[0])
    if path is None:
        raise BackendUnavailableError(f"`{argv[0]}` is not installed. {install_hint(argv[0])}")
    try:
        proc = subprocess.run(  # nosec B603 - argv list, no shell
            [path, *argv[1:]],
            capture_output=True,
            timeout=timeout,
            env=probe_env(),
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise CommandError(f"`{argv[0]}` timed out after {timeout:g}s") from exc
    except OSError as exc:
        raise CommandError(f"`{argv[0]}` could not be started: {exc}") from exc
    out = (proc.stdout or b"")[:max_output].decode("utf-8", errors="replace")
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        last = err.splitlines()[-1] if err else f"exit code {proc.returncode}"
        raise CommandError(f"`{argv[0]}` failed: {last[:300]}")
    return out


def has_module(name: str) -> bool:
    """``True`` when the optional package *name* is importable (no import side effects)."""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def binary_ok(cmd: str) -> bool:
    """``True`` when *cmd* is on PATH **and** actually runs (cached ``--version`` probe)."""
    if shutil.which(cmd) is None:
        return False
    return cached_probe(cmd, ttl=300.0).ok


def binary_hint(cmd: str) -> str:
    """Install / reinstall hint for *cmd* based on its probe result."""
    if shutil.which(cmd) is None:
        return install_hint(cmd)
    return cached_probe(cmd, ttl=300.0).hint or install_hint(cmd)


def json_lines(text: str) -> list[dict[str, Any]]:
    """Parse newline-delimited JSON, skipping blank or invalid lines."""
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def page(content: str, *, start: int = 0, max_chars: int = 20000) -> tuple[str, bool, int | None, int]:
    """Slice *content* for paging.

    Returns ``(piece, truncated, next_start, total_chars)``. A truncated piece
    ends with a ``[truncated — call again with start=N]`` marker.
    """
    total = len(content)
    start = max(0, min(int(start or 0), total))
    if max_chars is None or max_chars <= 0:
        return content[start:], False, None, total
    end = start + int(max_chars)
    piece = content[start:end]
    if end < total:
        return f"{piece}\n\n[truncated — call again with start={end}]", True, end, total
    return piece, False, None, total
