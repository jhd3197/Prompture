"""``prompture doctor``, ``prompture check-update`` and ``prompture watch``.

Exit codes:

* ``doctor`` — always ``0``; it reports, it doesn't gate (``2`` only for bad
  options, as click does). Use ``watch`` in scripts.
* ``check-update`` — always ``0``, including offline.
* ``watch`` — ``0`` healthy; ``1`` at least one ``broken``/``error`` row;
  ``2`` an update is available *and* ``--fail-on-update`` was given (``1``
  wins when both apply). An offline update check never fails the run.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import click

from ..capabilities.health import CATEGORIES

_MAX_RELEASES_SHOWN = 5
_ONLY_HELP = "Only these categories (repeatable): " + ", ".join(CATEGORIES) + "."


def _echo_update(info, *, highlights: bool = True) -> None:  # type: ignore[no-untyped-def]
    click.echo(info.summary())
    if info.error and info.source == "cache":
        click.echo(f"  (offline: showing the answer cached earlier; {info.error})")
    if highlights and info.highlights:
        click.echo("")
        click.echo("Changes since your version:")
        for rel in info.highlights[:_MAX_RELEASES_SHOWN]:
            date = f" ({rel['date']})" if rel.get("date") else ""
            click.echo(f"  {rel['version']}{date}")
            for note in rel.get("notes") or []:
                click.echo(f"    - {note}")
            if not rel.get("notes") and rel.get("url"):
                click.echo(f"    {rel['url']}")
        older = len(info.highlights) - _MAX_RELEASES_SHOWN
        if older > 0:
            click.echo(f"  ... and {older} older release(s); --json lists them all")


@contextmanager
def _quiet_driver_logs() -> Iterator[None]:
    """Keep driver warnings off the terminal; doctor already folds them into rows."""
    from ..doctor.providers import DRIVER_LOGGERS

    logs = [logging.getLogger(name) for name in DRIVER_LOGGERS]
    previous = [log.propagate for log in logs]
    for log in logs:
        log.propagate = False
    try:
        yield
    finally:
        for log, value in zip(logs, previous, strict=True):
            log.propagate = value


@click.command("doctor")
@click.option(
    "--live", is_flag=True, help="Also make cheap real calls (model lists, one query). Never a paid generation."
)
@click.option("--json", "json_output", is_flag=True, help="Print the prompture.doctor/1 JSON document.")
@click.option("--only", multiple=True, type=click.Choice(CATEGORIES), help=_ONLY_HELP)
@click.option("--verbose", "-v", is_flag=True, help="List every provider, including unconfigured ones.")
@click.option("--timeout", default=6.0, show_default=True, type=float, help="Per-provider timeout for --live.")
def doctor(live: bool, json_output: bool, only: tuple[str, ...], verbose: bool, timeout: float) -> None:
    """Show what works on this machine, what's active and how to fix the rest.

    Offline by default: no network calls, no writes. Exits 0 even when
    something is broken (use `prompture watch` to gate on health).
    """
    from ..doctor import check_all

    with _quiet_driver_logs():
        report = check_all(live=live, only=list(only) or None, verbose=verbose, timeout=timeout)
    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        click.echo(report.to_table())


@click.command("check-update")
@click.option("--json", "json_output", is_flag=True, help="Print the prompture.update/1 JSON document.")
@click.option("--force", is_flag=True, help="Ignore the 24h cache and ask PyPI now.")
@click.option("--timeout", default=5.0, show_default=True, type=float, help="Network timeout in seconds.")
def check_update(json_output: bool, force: bool, timeout: float) -> None:
    """Check PyPI for a newer Prompture and show what changed since yours.

    One PyPI call (plus GitHub release notes, best-effort), cached for 24h in
    ~/.prompture/update_check.json. Exits 0, also when offline.
    """
    from ..infra.updates import check_for_update

    info = check_for_update(force=force, timeout=timeout)
    if json_output:
        click.echo(json.dumps(info.to_dict(), indent=2))
    else:
        _echo_update(info)


@click.command("watch")
@click.option("--json", "json_output", is_flag=True, help="Print the prompture.watch/1 JSON document.")
@click.option("--no-update-check", is_flag=True, help="Skip the PyPI update check.")
@click.option("--fail-on-update", is_flag=True, help="Exit 2 when a newer version is available.")
@click.option("--only", multiple=True, type=click.Choice(CATEGORIES), help=_ONLY_HELP)
def watch(json_output: bool, no_update_check: bool, fail_on_update: bool, only: tuple[str, ...]) -> None:
    """Offline doctor + update check in one quick run, for cron/scheduled tasks.

    Exit codes: 0 healthy; 1 something is broken (a broken/error row);
    2 update available with --fail-on-update (1 wins when both apply).
    """
    from ..doctor import run_watch

    result = run_watch(update_check=not no_update_check, fail_on_update=fail_on_update, only=list(only) or None)
    if json_output:
        click.echo(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        report = result.report
        counts = ", ".join(f"{n} {s}" for s, n in report.summary()["counts"].items()) or "no checks"
        click.echo(f"health: {'ok' if report.ok else 'BROKEN'} (worst {report.worst}; {counts})")
        for row in report.failing:
            fix = f" - fix: {row.fix_hint}" if row.fix_hint else ""
            click.echo(f"  [{row.category}] {row.name}: {row.status} - {row.message}{fix}")
        if result.update is not None:
            click.echo(f"update: {result.update.summary()}")
    raise SystemExit(result.exit_code)


COMMANDS = [doctor, check_update, watch]

__all__ = ["COMMANDS", "check_update", "doctor", "watch"]
