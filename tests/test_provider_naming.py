"""Provider attribution in auto-recorded usage events.

When a driver's ``model`` carries no ``provider/`` prefix, the provider is
derived from the class name. Async twins must resolve to the same provider as
their sync counterparts — ``AsyncClaudeDriver`` is still ``claude``, not
``asyncclaude`` (which split one provider's spend into two rows).
"""

from __future__ import annotations

import asyncio
from typing import Any

from prompture.drivers.async_base import AsyncDriver
from prompture.drivers.base import Driver
from prompture.infra.tracker import UsageEvent, configure_tracker


class ClaudeStubDriver(Driver):
    def __init__(self) -> None:
        super().__init__()
        self.model = "claude-haiku-4-5"

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "ok", "meta": {"total_tokens": 3, "cost": 0.001}}


class AsyncClaudeStubDriver(AsyncDriver):
    def __init__(self) -> None:
        super().__init__()
        self.model = "claude-haiku-4-5"

    async def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return {"text": "ok", "meta": {"total_tokens": 3, "cost": 0.001}}


def _capture_events(tmp_path) -> list[UsageEvent]:
    events: list[UsageEvent] = []
    configure_tracker(db_path=str(tmp_path / "u.db"), persist=False, sinks=[events.append])
    return events


def test_sync_driver_provider_from_class_name(tmp_path):
    events = _capture_events(tmp_path)

    ClaudeStubDriver().generate_with_hooks("hi", {})

    assert events, "driver hook should auto-record a usage event"
    assert events[0].provider == "claudestub"
    assert events[0].model_name == "claudestub/claude-haiku-4-5"


def test_async_driver_strips_async_prefix(tmp_path):
    events = _capture_events(tmp_path)

    asyncio.run(AsyncClaudeStubDriver().generate_with_hooks("hi", {}))

    assert events, "driver hook should auto-record a usage event"
    # Same provider as the sync twin — never "asyncclaudestub".
    assert events[0].provider == "claudestub"
    assert events[0].model_name == "claudestub/claude-haiku-4-5"


def test_model_with_prefix_wins_over_class_name(tmp_path):
    events = _capture_events(tmp_path)

    drv = ClaudeStubDriver()
    drv.model = "claude/claude-haiku-4-5"
    drv.generate_with_hooks("hi", {})

    assert events[0].provider == "claude"
    assert events[0].model_name == "claude/claude-haiku-4-5"
