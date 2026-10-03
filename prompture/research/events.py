"""Progress events emitted by :class:`~prompture.research.ResearchAgent`.

Events follow the same frozen-dataclass + ``event_type`` shape as
:mod:`prompture.agents.live_events`, so UIs can dispatch on one field::

    def show(ev):
        print(f"[{ev.event_type}] {ev.message}")

    ResearchAgent("openai/gpt-4o-mini", on_event=show).run("...")

Event types, in the order a run produces them:

* ``plan`` — sub-questions decided (``data["sub_questions"]``).
* ``search`` — one search finished (``query``, ``platform``, ``results``, ``served_by``).
* ``fetch`` — one page opened or failed (``url``, ``ok``, ``reader``).
* ``synthesize`` — synthesis started (``sources``) or skipped (``reason``).
* ``warning`` — something degraded (budget limit, backend failure).
* ``done`` — the run finished; ``data["report"]`` holds the :class:`ResearchReport`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ResearchEventType = Literal["plan", "search", "fetch", "synthesize", "warning", "done"]


@dataclass(frozen=True)
class ResearchEvent:
    """One progress update from a research run.

    Attributes:
        event_type: See the module docstring.
        message: Short human-readable line.
        data: Structured details for the event type.
        elapsed_s: Seconds since the run started.
    """

    event_type: ResearchEventType
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = {k: v for k, v in self.data.items() if k != "report"}
        return {
            "event_type": self.event_type,
            "message": self.message,
            "data": data,
            "elapsed_s": round(self.elapsed_s, 2),
        }


__all__ = ["ResearchEvent", "ResearchEventType"]
