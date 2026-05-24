"""Streaming Extraction Example

Demonstrates :func:`prompture.extraction.streaming.stream_extract` and
:meth:`prompture.Conversation.ask_stream_model`: each yields a
progressively-filled Pydantic model as the LLM streams its JSON
response, so a UI can render fields the moment they arrive instead of
blocking on the full payload.

The example uses Ollama by default. Set ``OLLAMA_MODEL`` to a
streaming-capable instruction model (default ``llama3.1:8b``). When
the live LLM is unreachable, a scripted-driver fallback simulates the
stream so the example still demonstrates the API.

Run it
~~~~~~

::

    ollama pull llama3.1:8b
    OLLAMA_MODEL=llama3.1:8b python examples/streaming_extraction_example.py
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel, Field

from prompture import Conversation
from prompture.extraction.streaming import stream_extract

# =============================================================================
# Schema
# =============================================================================


class JobApplicant(BaseModel):
    """A small applicant record with enough fields to make streaming visible."""

    full_name: str = Field(..., description="Full legal name in 'First Last' order")
    current_role: str = Field(..., description="Current job title")
    years_experience: int = Field(..., description="Years of relevant work experience")
    skills: list[str] = Field(default_factory=list, description="Notable technical skills")
    location_city: str | None = Field(None, description="City the applicant is based in")


SAMPLE_TEXT = (
    "Alex Chen has been working in software for 9 years. Currently a senior "
    "platform engineer at Stripe based in San Francisco. Strengths: Python, "
    "Rust, distributed systems, observability, and Kubernetes."
)


# =============================================================================
# Scripted-driver fallback (when Ollama isn't reachable)
# =============================================================================


class _ScriptedStreamingDriver:
    """Simulates an LLM streaming a JSON object character-by-character.

    Emits realistic-shaped chunks (not just one big string) so the
    example shows progressive partial parsing in action even without
    a real LLM.
    """

    supports_json_schema = True
    model = "scripted/demo"
    model_name = "scripted/demo"

    _FULL_JSON = (
        '{\n  "full_name": "Alex Chen",\n  "current_role": "Senior Platform Engineer",\n'
        '  "years_experience": 9,\n  "skills": ["Python", "Rust", "Distributed Systems",'
        ' "Observability", "Kubernetes"],\n  "location_city": "San Francisco"\n}'
    )

    def generate_messages_stream(
        self, messages: list[dict[str, Any]], options: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        # Stream in roughly 6-char chunks with a tiny sleep so the visual is realistic.
        for i in range(0, len(self._FULL_JSON), 6):
            piece = self._FULL_JSON[i : i + 6]
            time.sleep(0.04)
            yield {"type": "delta", "text": piece}
        yield {"type": "done", "text": self._FULL_JSON}


def _pick_driver():
    """Return a real Ollama driver when reachable, otherwise the scripted fallback."""
    try:
        from prompture.drivers.ollama_driver import OllamaDriver

        driver = OllamaDriver(
            endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"),
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        )
        # Quick reachability probe.
        driver.generate("ping", {"temperature": 0.0, "max_tokens": 4})
        print(f"[LLM: ollama/{driver.model}]\n")
        return driver
    except Exception as e:
        print(f"[LLM: scripted fallback — Ollama not reachable: {type(e).__name__}]\n")
        return _ScriptedStreamingDriver()


# =============================================================================
# Pretty-print a partial model live (one line, updates in place)
# =============================================================================


def _render_snapshot(model: BaseModel, prev_text: str = "") -> str:
    """Print a partial-model snapshot, returning the text we rendered."""
    fields_set = {}
    for name in type(model).model_fields:
        try:
            val = getattr(model, name)
        except AttributeError:
            continue
        # model_construct leaves missing fields unset rather than None.
        if val is None and name in model.model_fields_set:
            fields_set[name] = None
        elif val is not None:
            fields_set[name] = val
    text = ", ".join(f"{k}={v!r}" for k, v in fields_set.items())
    if text != prev_text:
        print(f"    snapshot: {text}")
    return text


# =============================================================================
# Examples
# =============================================================================


def example_stream_extract_function() -> None:
    print("=" * 70)
    print("EXAMPLE 1: stream_extract() — direct, driver-level API")
    print("=" * 70)

    driver = _pick_driver()
    prompt = (
        "Extract the applicant fields from the text below as JSON matching "
        "the documented schema.\n\nTEXT:\n" + SAMPLE_TEXT
    )

    prev = ""
    final: JobApplicant | None = None
    for partial in stream_extract(driver, prompt, JobApplicant, options={"json_mode": True}):
        prev = _render_snapshot(partial, prev)
        final = partial

    print()
    if final is not None:
        print(f"  FINAL: {final.model_dump_json(indent=2)}")
    print()


def example_conversation_ask_stream_model() -> None:
    print("=" * 70)
    print("EXAMPLE 2: Conversation.ask_stream_model() — higher-level wrapper")
    print("=" * 70)

    driver = _pick_driver()
    conv = Conversation(driver=driver, system_prompt="You are a careful résumé parser.")

    prev = ""
    final: JobApplicant | None = None
    for partial in conv.ask_stream_model(
        "Extract the applicant record from this résumé snippet:\n" + SAMPLE_TEXT,
        JobApplicant,
    ):
        prev = _render_snapshot(partial, prev)
        final = partial

    print()
    if final is not None:
        print(f"  FINAL: {final.model_dump_json(indent=2)}")
    print()
    print(f"  Conversation history size: {len(conv._messages)} messages")


def main() -> None:
    example_stream_extract_function()
    example_conversation_ask_stream_model()


if __name__ == "__main__":
    main()
