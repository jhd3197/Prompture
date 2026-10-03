"""Built-in ``binaries`` health checks (``ffmpeg``, ``ffprobe``, ``yt-dlp``, ``gh``, ``node``)."""

from __future__ import annotations

from .health import HealthStatus, register_capability
from .probe import cached_probe

# binary → (probe args, what it unlocks)
BINARIES: dict[str, tuple[tuple[str, ...], str]] = {
    "ffmpeg": (("-version",), "audio/video transcription (chunking, downmix)"),
    "ffprobe": (("-version",), "media duration probing"),
    "yt-dlp": (("--version",), "YouTube/podcast audio + subtitles, platform search"),
    "gh": (("--version",), "GitHub CLI fallback reader and `gh` tool"),
    "node": (("--version",), "npx-launched MCP servers"),
}

_STATUS_MAP = {"ok": "ok", "missing": "missing", "broken": "broken", "timeout": "timeout", "error": "error"}


def binary_status(name: str) -> HealthStatus:
    args, unlocks = BINARIES.get(name, (("--version",), ""))
    probe = cached_probe(name, args)
    if probe.ok:
        message = probe.version or "ok"
    else:
        message = probe.output.splitlines()[0] if probe.output else probe.status
    return HealthStatus(
        name,
        _STATUS_MAP[probe.status],  # type: ignore[arg-type]
        category="binaries",
        active_backend=probe.path if probe.ok else None,
        message=message[:200],
        fix_hint=None if probe.ok else probe.hint,
        details={"path": probe.path, "unlocks": unlocks, "exit_code": probe.exit_code},
    )


def _make_check(name: str):
    def _check(live: bool) -> HealthStatus:
        return binary_status(name)

    return _check


for _name in BINARIES:
    register_capability(_name, "binaries", _make_check(_name), description=BINARIES[_name][1])
