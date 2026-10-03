"""``prompture skill install|uninstall|show`` — manage the bundled agent skill."""

from __future__ import annotations

import click

_TARGET = click.option(
    "--target",
    type=click.Choice(["claude", "project", "path"]),
    default="claude",
    show_default=True,
    help="claude = ~/.claude/skills, project = ./.claude/skills, path = --path DIR.",
)
_PATH = click.option(
    "--path", "path", default=None, type=click.Path(file_okay=False), help="Directory for --target path."
)
_DRY = click.option("--dry-run", is_flag=True, help="Print the plan without changing anything.")


@click.group()
def skill() -> None:
    """Install the Prompture skill so coding agents know how to use it."""


@skill.command("install")
@_TARGET
@_PATH
@_DRY
@click.option("--force", is_flag=True, help="Replace a directory not created by this command.")
def skill_install(target: str, path: str | None, dry_run: bool, force: bool) -> None:
    """Copy SKILL.md and its references into the target."""
    from ..skill import install_skill

    try:
        plan = install_skill(target, path=path, dry_run=dry_run, force=force)
    except (FileExistsError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(plan.describe())


@skill.command("uninstall")
@_TARGET
@_PATH
@_DRY
def skill_uninstall(target: str, path: str | None, dry_run: bool) -> None:
    """Remove a skill directory this command installed."""
    from ..skill import uninstall_skill

    try:
        plan = uninstall_skill(target, path=path, dry_run=dry_run)
    except (PermissionError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(plan.describe())


@skill.command("show")
@click.option("--file", "name", default="SKILL.md", show_default=True, help="Bundled file to print.")
def skill_show(name: str) -> None:
    """Print a bundled skill file (SKILL.md or references/<topic>.md)."""
    from ..skill import skill_files, skill_source_dir

    available = {str(f).replace("\\", "/") for f in skill_files()}
    if name not in available:
        raise click.ClickException(f"Unknown file {name!r}. Available: {', '.join(sorted(available))}")
    click.echo((skill_source_dir() / name).read_text(encoding="utf-8"))


COMMANDS = [skill]
