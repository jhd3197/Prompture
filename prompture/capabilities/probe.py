"""Real health probes for external binaries.

A binary on ``PATH`` is not proof that it runs: a venv shim left behind by a
Python upgrade resolves fine and then fails with "No Python at ...".
:func:`probe_command` executes a side-effect-free command (``--version`` by
default) and classifies the outcome as ``ok``, ``missing``, ``broken``,
``timeout`` or ``error``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from ..security.redaction import scrub_secrets

ProbeStatus = Literal["ok", "missing", "broken", "timeout", "error"]

# Output that means "the launcher exists but its target does not".
_BROKEN_MARKERS = (
    "no python at",
    "bad interpreter",
    "cannot execute",
    "is not recognized as an internal or external command",
    "the system cannot find the path specified",
    "the system cannot find the file specified",
    "no such file or directory",
    "error while loading shared libraries",
    "modulenotfounderror",
    "importerror",
    "dyld: library not loaded",
    "unable to create process",
)

# Variables that leak a parent interpreter / venv into the child.
DEFAULT_STRIP_ENV = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__")

_INSTALL_HINTS = {
    "ffmpeg": "Install ffmpeg (https://ffmpeg.org/download.html, `winget install ffmpeg`, `brew install ffmpeg`, `apt install ffmpeg`).",
    "ffprobe": "ffprobe ships with ffmpeg — install ffmpeg.",
    "yt-dlp": "Install yt-dlp: `pip install -U yt-dlp` (or your package manager).",
    "gh": "Install the GitHub CLI: https://cli.github.com/ and run `gh auth login`.",
    "node": "Install Node.js: https://nodejs.org/.",
    "npx": "Install Node.js (npx ships with npm): https://nodejs.org/.",
    "uvx": "Install uv: https://docs.astral.sh/uv/.",
}


@dataclass
class ProbeResult:
    """Outcome of :func:`probe_command`."""

    status: ProbeStatus
    command: str
    path: str | None = None
    output: str = ""
    hint: str | None = None
    exit_code: int | None = None
    elapsed_ms: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def version(self) -> str | None:
        """First non-empty output line — usually the version banner."""
        for line in self.output.splitlines():
            if line.strip():
                return line.strip()
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def install_hint(cmd: str) -> str:
    """Return a short install hint for *cmd*."""
    return _INSTALL_HINTS.get(
        os.path.basename(cmd).lower().removesuffix(".exe"), f"Install `{cmd}` and make sure it is on PATH."
    )


def probe_env(strip: Iterable[str] = DEFAULT_STRIP_ENV, extra: dict[str, str] | None = None) -> dict[str, str]:
    """A child environment with UTF-8 I/O and *strip* variables removed."""
    env = {k: v for k, v in os.environ.items() if k not in set(strip)}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if extra:
        env.update(extra)
    return env


def _looks_broken(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _BROKEN_MARKERS)


def probe_command(
    cmd: str,
    args: Sequence[str] = ("--version",),
    *,
    timeout: float = 10.0,
    retries: int = 0,
    strip_env: Iterable[str] = DEFAULT_STRIP_ENV,
    env: dict[str, str] | None = None,
    hint: str | None = None,
    max_output: int = 2000,
) -> ProbeResult:
    """Run ``cmd args`` without a shell and classify the result.

    Args:
        cmd: Binary name or path.
        args: Side-effect-free arguments (default ``--version``).
        timeout: Seconds before the attempt counts as ``timeout``.
        retries: Extra attempts after a timeout.
        strip_env: Inherited variables removed from the child environment.
        env: Extra variables for the child.
        hint: Override the install / reinstall hint.
        max_output: Characters of combined output kept on the result.
    """
    path = shutil.which(cmd)
    fix = hint or install_hint(cmd)
    if path is None:
        return ProbeResult("missing", cmd, hint=fix)

    child_env = probe_env(strip_env, env)
    attempts = max(1, retries + 1)
    last: ProbeResult | None = None
    for _ in range(attempts):
        start = time.monotonic()
        try:
            proc = subprocess.run(  # nosec B603 - argv list, no shell
                [path, *args],
                capture_output=True,
                timeout=timeout,
                env=child_env,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired:
            last = ProbeResult(
                "timeout",
                cmd,
                path=path,
                hint=f"`{cmd}` did not answer within {timeout:g}s.",
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )
            continue
        except OSError as exc:
            return ProbeResult(
                "broken",
                cmd,
                path=path,
                output=scrub_secrets(str(exc))[:max_output],
                hint=f"`{path}` exists but cannot be executed — reinstall it. {fix}",
                elapsed_ms=int((time.monotonic() - start) * 1000),
            )

        elapsed = int((time.monotonic() - start) * 1000)
        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        combined = scrub_secrets((out + ("\n" + err if err.strip() else "")).strip())[:max_output]
        if proc.returncode == 0 and not (not out.strip() and _looks_broken(err)):
            return ProbeResult("ok", cmd, path=path, output=combined, exit_code=0, elapsed_ms=elapsed)
        if proc.returncode in (126, 127) or _looks_broken(combined):
            return ProbeResult(
                "broken",
                cmd,
                path=path,
                output=combined,
                hint=f"`{path}` is on PATH but fails to start (stale shim or missing runtime?) — reinstall it. {fix}",
                exit_code=proc.returncode,
                elapsed_ms=elapsed,
            )
        return ProbeResult(
            "error",
            cmd,
            path=path,
            output=combined,
            hint=f"`{cmd} {' '.join(args)}` exited with code {proc.returncode}.",
            exit_code=proc.returncode,
            elapsed_ms=elapsed,
        )
    assert last is not None
    return last


_cache: dict[tuple[str, tuple[str, ...]], tuple[float, ProbeResult]] = {}
_cache_lock = threading.Lock()


def cached_probe(cmd: str, args: Sequence[str] = ("--version",), *, ttl: float = 60.0, **kwargs: Any) -> ProbeResult:
    """:func:`probe_command` with a short TTL cache so several checks can share one probe."""
    key = (cmd, tuple(args))
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    result = probe_command(cmd, args, **kwargs)
    with _cache_lock:
        _cache[key] = (now, result)
    return result


def clear_probe_cache() -> None:
    with _cache_lock:
        _cache.clear()
