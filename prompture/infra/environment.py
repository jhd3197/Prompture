"""Detect whether Prompture runs on a personal machine or a server.

:func:`detect_environment` looks for container, CI, SSH and headless markers
and returns an :class:`EnvironmentInfo` with ``kind`` (``"local"`` or
``"server"``) plus suggestions tailored to that setting — used by
``prompture setup`` to pick sensible defaults (e.g. offering a proxy on
servers, whose datacenter IPs some providers block).

Detection only reads environment variables and a few well-known files; it
never makes network calls or writes anything.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

__all__ = ["EnvironmentInfo", "detect_environment"]

_CI_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "BUILDKITE",
    "JENKINS_URL",
    "TF_BUILD",
    "TEAMCITY_VERSION",
    "BITBUCKET_BUILD_NUMBER",
    "CODEBUILD_BUILD_ID",
    "DRONE",
    "APPVEYOR",
)
_FALSY = frozenset({"", "0", "false", "no", "off"})
_CGROUP_MARKERS = ("docker", "kubepods", "containerd", "libpod", "lxc", "podman")


@dataclass
class EnvironmentInfo:
    """Where Prompture is running and what that implies for configuration."""

    kind: Literal["local", "server"]
    platform: str
    container: bool = False
    ci: bool = False
    ssh: bool = False
    headless: bool = False
    markers: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def is_server(self) -> bool:
        return self.kind == "server"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in _FALSY


def _read_small(path: Path, limit: int = 65536) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return ""


def detect_environment(
    environ: Mapping[str, str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    platform: str | None = None,
) -> EnvironmentInfo:
    """Classify the current machine as ``local`` or ``server``.

    Args:
        environ: Environment mapping to inspect (default ``os.environ``).
        root: Filesystem root used for marker files such as ``/.dockerenv``
            and ``/proc/1/cgroup`` (default ``/``; tests point it elsewhere).
        platform: ``sys.platform``-style value (default ``sys.platform``).
    """
    env = os.environ if environ is None else environ
    plat = platform or sys.platform
    base = Path(root) if root is not None else Path("/")
    markers: list[str] = []

    # Containers --------------------------------------------------------
    container = False
    if plat.startswith("linux") or root is not None:
        if (base / ".dockerenv").exists():
            container = True
            markers.append("container: /.dockerenv")
        if (base / "run" / ".containerenv").exists():
            container = True
            markers.append("container: /run/.containerenv")
        cgroup = _read_small(base / "proc" / "1" / "cgroup").lower()
        hit = next((m for m in _CGROUP_MARKERS if m in cgroup), None)
        if hit:
            container = True
            markers.append(f"container: cgroup mentions {hit}")
    if env.get("KUBERNETES_SERVICE_HOST"):
        container = True
        markers.append("container: KUBERNETES_SERVICE_HOST")
    if env.get("container"):
        container = True
        markers.append(f"container: container={env.get('container')}")

    # CI ----------------------------------------------------------------
    ci_hits = [name for name in _CI_VARS if _truthy(env.get(name))]
    ci = bool(ci_hits)
    markers.extend(f"ci: {name}" for name in ci_hits)

    # Remote shell / headless ------------------------------------------
    ssh_hits = [name for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY") if env.get(name)]
    ssh = bool(ssh_hits)
    if ssh:
        markers.append(f"ssh: {ssh_hits[0]}")
    headless = False
    if plat.startswith("linux") and not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        headless = True
        markers.append("headless: no DISPLAY or WAYLAND_DISPLAY")

    kind: Literal["local", "server"] = "server" if (container or ci or ssh or headless) else "local"

    suggestions: list[str] = []
    if kind == "server":
        suggestions.append(
            "Servers often get blocked by some providers from datacenter IPs; "
            "consider PROMPTURE_PROXY (or PROMPTURE_<BACKEND>_PROXY for a single backend)."
        )
    if ci:
        suggestions.append("In CI, pass keys as masked environment secrets rather than a credential file.")
    if container:
        suggestions.append(
            "Inside a container ~/.prompture is usually ephemeral; pass keys as environment "
            "variables or mount the credential file read-only."
        )
    if ssh or headless:
        suggestions.append("No local browser is available here; paste API keys directly when prompted.")
    if kind == "local":
        suggestions.append(
            "Keys saved by `prompture setup` go to ~/.prompture/credentials.yaml (owner-only); "
            "environment variables and .env still take precedence."
        )

    return EnvironmentInfo(
        kind=kind,
        platform=plat,
        container=container,
        ci=ci,
        ssh=ssh,
        headless=headless,
        markers=markers,
        suggestions=suggestions,
    )
