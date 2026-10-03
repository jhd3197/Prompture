"""``prompture doctor``: what works on this machine, what's active, how to fix the rest.

>>> from prompture.doctor import check_all
>>> report = check_all()            # offline: no network, no writes
>>> report.worst, report.ok
>>> print(report.to_table())
>>> report.to_dict()["schema"]      # "prompture.doctor/1"

Importing this package registers the ``providers`` capability; the tools,
media, MCP and binaries capabilities register themselves when their modules
load (see :data:`prompture.capabilities.health.BUILTIN_CAPABILITY_MODULES`).
"""

from .core import (
    EXIT_BROKEN,
    EXIT_OK,
    EXIT_UPDATE,
    FAILING_STATUSES,
    SCHEMA,
    SUMMARY_SCHEMA,
    WATCH_SCHEMA,
    DoctorReport,
    WatchResult,
    capabilities_summary,
    check_all,
    clear_summary_cache,
    run_watch,
    watch_exit_code,
)
from .providers import provider_rows

__all__ = [
    "EXIT_BROKEN",
    "EXIT_OK",
    "EXIT_UPDATE",
    "FAILING_STATUSES",
    "SCHEMA",
    "SUMMARY_SCHEMA",
    "WATCH_SCHEMA",
    "DoctorReport",
    "WatchResult",
    "capabilities_summary",
    "check_all",
    "clear_summary_cache",
    "provider_rows",
    "run_watch",
    "watch_exit_code",
]
