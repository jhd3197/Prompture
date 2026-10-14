"""Game turn progress: events, a deadline and a Stop button for a model playing Connect Four.

This script demonstrates:
1. Bounding a turn with ``deadline`` and stopping it with a ``CancelToken``.
2. Consuming sequenced ``TurnEvent``s from a bounded ``EventBuffer`` on another thread.
3. Reading the turn record: attempts, outcome and honest cost completeness.

By default a local scripted driver streams the replies, so nothing is sent to a
provider and nothing is billed. Pass ``--model provider/model`` to play a real
model instead (that run makes paid calls if the provider charges).

Run it::

    python examples/game_progress_example.py
    python examples/game_progress_example.py --stop-after 0.2
    python examples/game_progress_example.py --model ollama/llama3.2:3b
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prompture.drivers.base import Driver
from prompture.games import Connect4, Connect4Board, GameHarness, TurnCancelled
from prompture.games.control import CancelToken, EventBuffer


class LocalStreamingDriver(Driver):
    """Streams a fixed answer a few characters at a time. No network, no cost."""

    supports_streaming = True
    supports_json_mode = True
    model = "local/scripted"

    def generate(self, prompt, options):  # streaming only
        raise NotImplementedError

    def generate_messages_stream(self, messages, options):
        text = json.dumps(
            {"reason": "The center column joins the most lines.", "plan": "Build on column 4.", "move": "4"}
        )
        for start in range(0, len(text), 6):
            time.sleep(0.03)
            yield {"type": "delta", "text": text[start : start + 6]}
        yield {
            "type": "done",
            "text": text,
            "meta": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        }


def consume(buffer: EventBuffer, done: threading.Event) -> None:
    """A slow UI: print events as they arrive; deltas coalesce if it falls behind."""
    while not (done.is_set() and not len(buffer)):
        event = buffer.get(timeout=0.1)
        if event is None:
            continue
        if event.kind == "output_delta":
            print(f"  [{event.seq:>3}] delta {event.data['text']!r}")
        else:
            print(f"  [{event.seq:>3}] {event.kind} {json.dumps(event.data, default=str)[:100]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="play a real model, e.g. ollama/llama3.2:3b (may make paid calls)")
    parser.add_argument("--stop-after", type=float, help="press Stop after this many seconds")
    parser.add_argument("--deadline", type=float, default=30.0, help="total seconds per turn")
    args = parser.parse_args()

    game = Connect4()
    token = CancelToken()
    buffer = EventBuffer(maxsize=64)
    driver = None if args.model else LocalStreamingDriver()
    harness = GameHarness(
        game,
        args.model,
        driver=driver,
        level=1,
        concepts="all",
        deadline=args.deadline,
        cancel=token,
        on_event=buffer.put,
    )
    done = threading.Event()
    printer = threading.Thread(target=consume, args=(buffer, done))
    printer.start()
    if args.stop_after is not None:
        threading.Timer(args.stop_after, token.cancel, args=("Stop pressed",)).start()

    print(f"Red ({harness.model}) to move; harness fingerprint {harness.fingerprint()}")
    try:
        decision = harness.choose(Connect4Board())
    except TurnCancelled as exc:
        done.set()
        printer.join()
        print(f"\nStopped: {exc}. No move was played.")
        print(f"Attempts: {len(exc.record.attempts)}; cost {exc.record.accounting.cost_label}")
        return
    done.set()
    printer.join()
    record = decision.record
    print(f"\nMove {decision.name} ({record.outcome}); reason: {decision.reason}")
    print(f"Plan: {decision.plan}")
    print(
        f"Attempts: {len(record.attempts)}; tokens {record.accounting.total_tokens:.0f}; cost {record.accounting.cost_label}"
    )
    print(f"Dropped events (slow consumer): {buffer.dropped}")


if __name__ == "__main__":
    main()
