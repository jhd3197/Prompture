"""The companion API: what desktop companions (e.g. Prompture Desk) read.

Two servers speak it:

- **prompture-hub** — the gateway, which sees every call routed through it and
  adds keys, projects, alert rules and controls.
- **the local companion** (``prompture companion``) — no hub needed; reports
  this machine's Prompture usage from the usage ledger, rate-limit headroom and
  provider account balances, plus the usage local coding agents (Claude Code,
  Codex, Kimi Code, Gemini CLI, Qwen Code, OpenCode, Cline, …) log on disk,
  and runs queued coding-agent steps one after another (automations).

This package holds what both share — the live event bus, the spend / limits
aggregations and the API version — plus the local server itself.
"""

from .automations import Automations
from .coding_tools import CodingToolSource
from .live import LiveBus, get_bus, new_request_id, sse_event, visible
from .local import LedgerSource
from .server import CompanionServer, read_state, running_instance
from .summary import (
    COMPANION_API_VERSION,
    PERIODS,
    UsageRow,
    account_limits,
    provider_limits,
    summarize_spend,
    window_end,
    window_start,
)

__all__ = [
    "COMPANION_API_VERSION",
    "PERIODS",
    "Automations",
    "CodingToolSource",
    "CompanionServer",
    "LedgerSource",
    "LiveBus",
    "UsageRow",
    "account_limits",
    "get_bus",
    "new_request_id",
    "provider_limits",
    "read_state",
    "running_instance",
    "sse_event",
    "summarize_spend",
    "visible",
    "window_end",
    "window_start",
]
