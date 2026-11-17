"""Scenarios a driver is known to fail, as ``(target id, test name)``.

Each entry is an ``xfail(strict=True)``: the suite stays green while the gap
is open, and turns red the moment it is fixed so the entry gets deleted. An
empty table is the goal.
"""

from __future__ import annotations


def _gaps(test: str, reason: str, targets: str) -> dict[tuple[str, str], str]:
    return {(target, test): reason for target in targets.split()}


KNOWN_GAPS: dict[tuple[str, str], str] = {}
