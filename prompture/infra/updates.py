"""Update awareness: is a newer Prompture on PyPI, and what changed since mine?

:func:`check_for_update` makes one PyPI JSON call (plus a best-effort GitHub
releases call for changelog highlights), caches the answer for ~24 hours in
``~/.prompture/update_check.json`` and never raises: offline, it falls back to
the last cached answer or reports the error in :attr:`UpdateInfo.error`.

The same state file backs the "mention an update once" rule for agents:
:func:`should_announce` is true until :func:`mark_announced` records a version.
That file is the only thing this module writes, always atomically.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("prompture.updates")

PACKAGE = "prompture"
PYPI_URL = "https://pypi.org/pypi/prompture/json"
RELEASES_URL = "https://api.github.com/repos/jhd3197/prompture/releases"
STATE_FILE = Path.home() / ".prompture" / "update_check.json"
CACHE_TTL_SECONDS = 24 * 3600
DEFAULT_TIMEOUT = 5.0
UPGRADE_COMMAND = "pip install -U prompture"

_MAX_RELEASES = 10
_MAX_NOTES_PER_RELEASE = 5

#: ``fetch_json(url, timeout) -> parsed JSON``; injectable for tests.
FetchJSON = Callable[[str, float], Any]


@dataclass
class UpdateInfo:
    """Result of an update check.

    Attributes:
        installed: Installed Prompture version.
        latest: Latest version on PyPI (``None`` when unknown).
        update_available: ``latest`` is newer than ``installed``.
        checked_at: Unix time the PyPI answer was fetched.
        source: ``network`` (fresh), ``cache`` (within TTL or offline fallback)
            or ``offline`` (no answer at all).
        error: Why the network check failed, if it did.
        highlights: ``[{"version", "date", "url", "notes": [...]}]`` for
            releases newer than ``installed`` (newest first, best-effort).
        release_url: PyPI page of the latest version.
    """

    installed: str
    latest: str | None = None
    update_available: bool = False
    checked_at: float | None = None
    source: str = "offline"
    error: str | None = None
    highlights: list[dict[str, Any]] = field(default_factory=list)
    release_url: str | None = None

    @property
    def upgrade_command(self) -> str | None:
        return UPGRADE_COMMAND if self.update_available else None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema"] = "prompture.update/1"
        data["upgrade_command"] = self.upgrade_command
        return data

    def summary(self) -> str:
        """One line for humans."""
        if self.latest is None:
            reason = f" ({self.error})" if self.error else ""
            return f"prompture {self.installed} installed; latest version unknown{reason}"
        if self.update_available:
            return f"prompture {self.latest} is available (installed {self.installed}); run: {UPGRADE_COMMAND}"
        return f"prompture {self.installed} is up to date (latest {self.latest})"


# ── Versions ──────────────────────────────────────────────────────────────


def installed_version() -> str:
    """The installed Prompture version (``"0"`` when it can't be determined)."""
    try:
        from importlib.metadata import version

        return version(PACKAGE)
    except Exception:
        try:
            from .. import __version__  # type: ignore[attr-defined]

            return str(__version__)
        except Exception:
            return "0"


_PRE_TAGS = {"dev": -4, "a": -3, "alpha": -3, "b": -2, "beta": -2, "rc": -1, "c": -1, "pre": -1, "preview": -1}


def _fallback_key(text: str) -> tuple[tuple[int, ...], int, int]:
    """Sort key used when ``packaging`` is unavailable: (release, pre-tag rank, pre number)."""
    text = text.strip().lower().lstrip("v").split("+", 1)[0]
    m = re.match(r"(\d+(?:\.\d+)*)(.*)", text)
    if not m:
        return ((0,), 0, 0)
    release = tuple(int(p) for p in m.group(1).split("."))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]
    rest = m.group(2)
    tag = re.search(r"(dev|alpha|beta|preview|pre|rc|a|b|c)\.?(\d*)", rest)
    post = re.search(r"post\.?(\d*)", rest)
    if tag:
        return (release, _PRE_TAGS[tag.group(1)], int(tag.group(2) or 0))
    if post:
        return (release, 1, int(post.group(1) or 0))
    return (release, 0, 0)


def compare_versions(a: str, b: str) -> int:
    """``-1`` if *a* < *b*, ``0`` if equal, ``1`` if *a* > *b* (PEP 440 when ``packaging`` exists)."""
    try:
        from packaging.version import InvalidVersion, Version

        try:
            va, vb = Version(a.lstrip("vV")), Version(b.lstrip("vV"))
            return (va > vb) - (va < vb)
        except InvalidVersion:
            pass
    except ImportError:
        pass
    ka, kb = _fallback_key(a), _fallback_key(b)
    return (ka > kb) - (ka < kb)


def is_newer(candidate: str, current: str) -> bool:
    return compare_versions(candidate, current) > 0


def _is_prerelease(text: str) -> bool:
    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(text.lstrip("vV")).is_prerelease
        except InvalidVersion:
            return False
    except ImportError:
        return _fallback_key(text)[1] < 0


# ── State file ────────────────────────────────────────────────────────────


def _read_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_state(path: Path, data: dict[str, Any]) -> None:
    """Atomically replace the state file; failures are logged, never raised."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".update_check.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            if os.name == "posix":
                os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    except OSError as exc:
        logger.debug("could not write update state %s: %s", path, exc)


def should_announce(version: str | None, *, state_path: Path | None = None) -> bool:
    """True when *version* hasn't been announced yet (the "mention once" rule)."""
    if not version:
        return False
    announced = _read_state(state_path or STATE_FILE).get("announced") or []
    return version not in announced


def mark_announced(version: str, *, state_path: Path | None = None) -> None:
    """Record that *version* was mentioned so :func:`should_announce` stays quiet for it."""
    path = state_path or STATE_FILE
    state = _read_state(path)
    announced = [v for v in state.get("announced") or [] if isinstance(v, str)]
    if version not in announced:
        announced.append(version)
    state["announced"] = announced[-20:]
    _write_state(path, state)


# ── Fetching ──────────────────────────────────────────────────────────────


def _default_fetch_json(url: str, timeout: float) -> Any:
    from ..capabilities.http import safe_get

    headers = {"Accept": "application/json", "User-Agent": f"prompture/{installed_version()} (update-check)"}
    resp = safe_get(url, headers=headers, timeout=timeout, max_bytes=8 * 1024 * 1024, check_challenge=False)
    return resp.json()


def _clean_note(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)  # [text](url) -> text
    line = re.sub(r"[*_`]{1,3}", "", line)
    return line.strip()[:200]


def release_notes(body: str | None, limit: int = _MAX_NOTES_PER_RELEASE) -> list[str]:
    """Pull up to *limit* highlight lines out of a markdown release body (bullets first)."""
    if not body:
        return []
    lines = [ln.rstrip() for ln in body.splitlines()]
    bullets = [_clean_note(ln) for ln in lines if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+\S", ln)]
    notes = [b for b in bullets if b and not _BOILERPLATE.match(b)]
    if not notes:
        notes = [_clean_note(ln) for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", "<!--", "|"))]
        notes = [n for n in notes if n and not _BOILERPLATE.match(n)]
    return notes[:limit]


# Release-bot lines that say nothing about what changed.
_BOILERPLATE = re.compile(r"^(?:version|pr|pr url|full changelog|commit|commits|compare)\s*:", re.IGNORECASE)


def changelog_since(releases: Any, installed: str, latest: str) -> list[dict[str, Any]]:
    """Highlights of stable releases with ``installed < version <= latest`` (newest first)."""
    out: list[dict[str, Any]] = []
    if not isinstance(releases, list):
        return out
    for rel in releases:
        if not isinstance(rel, dict) or rel.get("draft") or rel.get("prerelease"):
            continue
        tag = str(rel.get("tag_name") or rel.get("name") or "").strip()
        version = tag.lstrip("vV")
        if not version or not re.match(r"\d", version):
            continue
        if not is_newer(version, installed) or is_newer(version, latest):
            continue
        out.append(
            {
                "version": version,
                "date": (rel.get("published_at") or "")[:10] or None,
                "url": rel.get("html_url"),
                "notes": release_notes(rel.get("body")),
            }
        )
    out.sort(key=functools.cmp_to_key(lambda a, b: compare_versions(a["version"], b["version"])), reverse=True)
    return out[:_MAX_RELEASES]


def _info_from_state(state: dict[str, Any], installed: str, source: str, error: str | None) -> UpdateInfo:
    latest = state.get("latest")
    highlights = state.get("highlights") if state.get("installed") == installed else []
    return UpdateInfo(
        installed=installed,
        latest=latest,
        update_available=bool(latest) and is_newer(latest, installed),
        checked_at=state.get("checked_at"),
        source=source,
        error=error,
        highlights=highlights or [],
        release_url=state.get("release_url"),
    )


def check_for_update(
    *,
    installed: str | None = None,
    force: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    include_changelog: bool = True,
    include_prereleases: bool = False,
    cache_ttl: float = CACHE_TTL_SECONDS,
    state_path: Path | None = None,
    fetch_json: FetchJSON | None = None,
    now: float | None = None,
) -> UpdateInfo:
    """Compare the installed Prompture with PyPI's latest; never raises.

    Args:
        installed: Version to compare (defaults to the installed package).
        force: Ignore a fresh cache and ask PyPI again.
        timeout: Per-request timeout in seconds.
        include_changelog: Also fetch GitHub release notes (best-effort).
        include_prereleases: Consider pre-releases on PyPI as "latest".
        cache_ttl: Seconds a cached answer stays fresh (default 24 h).
        state_path: Cache/state file (default ``~/.prompture/update_check.json``).
        fetch_json: ``(url, timeout) -> JSON`` override (tests, proxies).
        now: Clock override (tests).
    """
    path = state_path or STATE_FILE
    installed = installed or installed_version()
    clock = time.time() if now is None else now
    state = _read_state(path)

    fresh = (
        not force
        and state.get("latest")
        and state.get("installed") == installed
        and isinstance(state.get("checked_at"), (int, float))
        and 0 <= clock - state["checked_at"] < cache_ttl
    )
    if fresh:
        return _info_from_state(state, installed, "cache", None)

    fetch = fetch_json or _default_fetch_json
    try:
        data = fetch(PYPI_URL, timeout)
        latest = _latest_from_pypi(data, include_prereleases)
    except Exception as exc:  # offline, DNS, 5xx, bad JSON
        from ..security.redaction import scrub_secrets

        error = scrub_secrets(f"{type(exc).__name__}: {exc}")[:300]
        if state.get("latest"):
            return _info_from_state(state, installed, "cache", error)
        return UpdateInfo(installed=installed, source="offline", error=error)

    highlights: list[dict[str, Any]] = []
    if include_changelog and is_newer(latest, installed):
        try:
            highlights = changelog_since(fetch(RELEASES_URL, timeout), installed, latest)
        except Exception as exc:
            logger.debug("changelog fetch failed: %s", exc)

    release_url = f"https://pypi.org/project/{PACKAGE}/{latest}/"
    state.update(
        {
            "checked_at": clock,
            "installed": installed,
            "latest": latest,
            "highlights": highlights,
            "release_url": release_url,
        }
    )
    _write_state(path, state)
    return UpdateInfo(
        installed=installed,
        latest=latest,
        update_available=is_newer(latest, installed),
        checked_at=clock,
        source="network",
        highlights=highlights,
        release_url=release_url,
    )


def _latest_from_pypi(data: Any, include_prereleases: bool) -> str:
    if not isinstance(data, dict):
        raise ValueError("unexpected PyPI response")
    info_version = str((data.get("info") or {}).get("version") or "")
    candidates = [info_version] if info_version and (include_prereleases or not _is_prerelease(info_version)) else []
    releases = data.get("releases")
    if isinstance(releases, dict):
        for version, files in releases.items():
            yanked = isinstance(files, list) and files and all(isinstance(f, dict) and f.get("yanked") for f in files)
            if not yanked and (include_prereleases or not _is_prerelease(version)):
                candidates.append(version)
    if not candidates and info_version:
        candidates = [info_version]
    if not candidates:
        raise ValueError("PyPI response has no version")
    return max(candidates, key=functools.cmp_to_key(compare_versions))


__all__ = [
    "CACHE_TTL_SECONDS",
    "PYPI_URL",
    "RELEASES_URL",
    "STATE_FILE",
    "UPGRADE_COMMAND",
    "UpdateInfo",
    "changelog_since",
    "check_for_update",
    "compare_versions",
    "installed_version",
    "is_newer",
    "mark_announced",
    "release_notes",
    "should_announce",
]
