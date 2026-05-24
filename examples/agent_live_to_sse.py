#!/usr/bin/env python3
"""Forward a live agent stream to Server-Sent Events.

Drop-in pattern for any web app: every LiveEvent is one ``data:`` frame.
Frontend reads them with ``EventSource`` and renders text deltas + tool
calls as they arrive. No buffering, no callbacks required.

Run::

    OPENAI_API_KEY=sk-... python examples/agent_live_to_sse.py
    # then in another shell:
    curl -N http://127.0.0.1:8765/chat

This example uses ``http.server`` so it has zero web-framework deps. The
event-shaping logic in :func:`event_to_sse` is what you'd lift into a
FastAPI / Starlette / Flask / aiohttp handler.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from prompture import Agent
from prompture.agents.live_events import (
    AssistantTurnStart,
    LiveEvent,
    MessageStop,
    TextDelta,
    ThinkingDelta,
    ToolInputDelta,
    ToolResult,
    ToolUseStart,
    ToolUseStop,
    TurnComplete,
)


def get_weather(city: str) -> str:
    """Return a fake weather report for *city*."""
    fake = {"Paris": "18°C, cloudy", "Tokyo": "27°C, sunny", "London": "12°C, rainy"}
    return fake.get(city, f"No data for {city}.")


def event_to_sse(event: LiveEvent) -> str:
    """Convert one ``LiveEvent`` into an SSE ``data:`` frame.

    Each variant is serialised as a typed JSON object so the frontend can
    ``switch (msg.type)`` cleanly. We do not include the raw dataclass
    because some fields (like ``usage``) are nested dicts.
    """
    payload: dict[str, Any]
    match event:
        case AssistantTurnStart():
            payload = {"type": "turn_start", "turn": event.turn_index}
        case TextDelta():
            payload = {"type": "text", "text": event.text}
        case ThinkingDelta():
            payload = {"type": "thinking", "text": event.text}
        case ToolUseStart():
            payload = {"type": "tool_start", "id": event.id, "name": event.name}
        case ToolInputDelta():
            payload = {"type": "tool_input_delta", "id": event.id, "fragment": event.fragment}
        case ToolUseStop():
            payload = {
                "type": "tool_input_ready",
                "id": event.id,
                "name": event.name,
                "input": event.input,
            }
        case ToolResult():
            payload = {
                "type": "tool_result",
                "id": event.id,
                "name": event.name,
                "output": event.output,
                "is_error": event.is_error,
            }
        case MessageStop():
            payload = {"type": "turn_stop", "stop_reason": event.stop_reason, "usage": event.usage}
        case TurnComplete():
            payload = {"type": "done", "usage": event.usage}
        case _:
            payload = {"type": "unknown", "repr": repr(event)}
            if dataclasses.is_dataclass(event):
                payload["data"] = dataclasses.asdict(event)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/chat":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        agent = Agent(
            "openai/gpt-4o-mini",
            system_prompt="Use the weather tool when asked about cities. Narrate what you're doing.",
            tools=[get_weather],
        )
        streamed = agent.run_live("What's the weather in Paris and Tokyo? Compare them.")
        for event in streamed:
            try:
                self.wfile.write(event_to_sse(event).encode("utf-8"))
                self.wfile.flush()
            except BrokenPipeError:
                break

    def log_message(self, *_a: Any, **_kw: Any) -> None:
        # Silence default access log for cleaner demo output.
        pass


def main() -> int:
    addr = ("127.0.0.1", 8765)
    print(f"SSE demo listening at http://{addr[0]}:{addr[1]}/chat")
    print("In another shell:  curl -N http://127.0.0.1:8765/chat")
    server = HTTPServer(addr, _Handler)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
