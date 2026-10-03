"""The Prompture agent skill: a ``SKILL.md`` plus reference pages that teach a
coding agent how to search, read, transcribe, call models and mount MCP
servers through Prompture.

:func:`install_skill` copies the bundled skill into an explicit target only:

* ``claude`` — ``~/.claude/skills/prompture/``
* ``project`` — ``./.claude/skills/prompture/`` (current directory)
* ``path`` — ``<path>/prompture/``

A marker file records ownership, so :func:`uninstall_skill` only ever removes
a directory this installer created.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

SKILL_NAME = "prompture"
MARKER_FILE = ".prompture-skill"
TARGETS = ("claude", "project", "path")


def skill_source_dir() -> Path:
    """Directory holding the bundled ``SKILL.md`` and ``references/``."""
    return Path(__file__).resolve().parent / "data"


def skill_files() -> list[Path]:
    """Bundled skill files, relative to :func:`skill_source_dir`."""
    root = skill_source_dir()
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file() and p.suffix == ".md")


def resolve_target(target: str = "claude", path: str | os.PathLike[str] | None = None) -> Path:
    """Return the skill directory for *target*."""
    if target == "claude":
        return Path.home() / ".claude" / "skills" / SKILL_NAME
    if target == "project":
        return Path.cwd() / ".claude" / "skills" / SKILL_NAME
    if target == "path":
        if not path:
            raise ValueError("target='path' needs a path")
        return Path(path).expanduser().resolve() / SKILL_NAME
    raise ValueError(f"Unknown target {target!r}; expected one of {', '.join(TARGETS)}")


@dataclass
class SkillPlan:
    """What an install or uninstall did (or would do, with ``dry_run``)."""

    action: str
    target_dir: Path
    files: list[str] = field(default_factory=list)
    dry_run: bool = False
    skipped_reason: str | None = None

    def describe(self) -> str:
        if self.skipped_reason:
            return f"Nothing to do: {self.skipped_reason}"
        if self.dry_run:
            head = f"[dry-run] Would {self.action} the Prompture skill at {self.target_dir}"
        else:
            done = "Installed" if self.action == "install" else "Removed"
            head = f"{done} the Prompture skill at {self.target_dir}"
        return "\n".join([head, *(f"  {f}" for f in self.files)])


def _version() -> str:
    try:
        from .. import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def install_skill(
    target: str = "claude",
    *,
    path: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> SkillPlan:
    """Copy the bundled skill into *target*.

    An existing directory is replaced only when it carries this installer's
    marker (an upgrade) or ``force`` is set.
    """
    dest = resolve_target(target, path)
    files = [str(f).replace(os.sep, "/") for f in skill_files()]
    plan = SkillPlan("install", dest, files, dry_run)
    if dest.is_symlink():
        raise ValueError(f"{dest} is a symlink; refusing to write through it")
    if dest.exists() and not (dest / MARKER_FILE).exists() and not force:
        raise FileExistsError(f"{dest} exists and was not created by `prompture skill install` (use --force)")
    if dry_run:
        return plan
    src = skill_source_dir()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel in skill_files():
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, out)
    (dest / MARKER_FILE).write_text(f"prompture {_version()}\n", encoding="utf-8")
    return plan


def uninstall_skill(
    target: str = "claude",
    *,
    path: str | os.PathLike[str] | None = None,
    dry_run: bool = False,
) -> SkillPlan:
    """Remove a skill directory previously written by :func:`install_skill`."""
    dest = resolve_target(target, path)
    if not dest.exists():
        return SkillPlan("uninstall", dest, dry_run=dry_run, skipped_reason=f"{dest} does not exist")
    if dest.is_symlink() or not (dest / MARKER_FILE).exists():
        raise PermissionError(f"{dest} was not created by `prompture skill install`; leaving it alone")
    files = sorted(str(p.relative_to(dest)).replace(os.sep, "/") for p in dest.rglob("*") if p.is_file())
    plan = SkillPlan("uninstall", dest, files, dry_run)
    if not dry_run:
        shutil.rmtree(dest)
    return plan


__all__ = [
    "MARKER_FILE",
    "SKILL_NAME",
    "TARGETS",
    "SkillPlan",
    "install_skill",
    "resolve_target",
    "skill_files",
    "skill_source_dir",
    "uninstall_skill",
]
