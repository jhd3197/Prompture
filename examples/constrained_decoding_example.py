"""Constrained-decoding example for vLLM / LMStudio / OpenRouter.

Demonstrates the Phase 8A-lite ``guided_decoding`` and ``extra_body``
options. Any OpenAI-compatible driver — ``OpenAICompatibleDriver``,
``OpenRouterDriver``, ``LMStudioDriver`` (sync + async) — accepts these
options and ships them on the request:

* ``options={"guided_decoding": True}`` adds vLLM's ``guided_json``
  field at the top level. The server picks its own backend (Outlines
  by default on vLLM 0.5+).
* ``options={"guided_decoding": "outlines"}`` (or ``"xgrammar"`` /
  ``"lm-format-enforcer"``) explicitly pins the backend.
* ``options={"extra_body": {...}}`` mirrors the OpenAI SDK's
  ``extra_body`` escape hatch — drop any vendor-specific fields here.

Why this matters
~~~~~~~~~~~~~~~~

Modern vLLM honours ``response_format`` natively, so most users don't
need ``guided_decoding`` at all. But:

* Older vLLM versions (<0.5) only respect ``guided_json``.
* Some models / servers handle FSM-level decoding only via the
  ``guided_*`` family, not ``response_format``.
* Some backends (xgrammar) are dramatically faster than the default
  Outlines backend and can be selected explicitly.

This example assembles a request payload via Prompture's stack
without sending it anywhere — it prints the body that would hit the
server, so you can see exactly what flows under the hood.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import BaseModel

from prompture.drivers._openai_compat import apply_structured_output


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class Person(BaseModel):
    name: str
    age: int
    address: Address


def _print_body(label: str, body: dict[str, Any]) -> None:
    print(f"  {label}:")
    print(json.dumps(body, indent=2, default=str))
    print()


def main() -> None:
    schema = Person.model_json_schema()

    print("=" * 72)
    print("EXAMPLE 1: response_format only (every OpenAI-compatible server)")
    print("=" * 72)
    body = {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "..."}]}
    apply_structured_output(body, options={"json_mode": True, "json_schema": schema})
    _print_body("payload", body)

    print("=" * 72)
    print("EXAMPLE 2: + guided_decoding=True (vLLM picks its own backend)")
    print("=" * 72)
    body = {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "..."}]}
    apply_structured_output(
        body,
        options={"json_mode": True, "json_schema": schema, "guided_decoding": True},
    )
    _print_body("payload", body)

    print("=" * 72)
    print("EXAMPLE 3: pin to xgrammar (fast lattice FSM)")
    print("=" * 72)
    body = {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "..."}]}
    apply_structured_output(
        body,
        options={"json_mode": True, "json_schema": schema, "guided_decoding": "xgrammar"},
    )
    _print_body("payload", body)

    print("=" * 72)
    print("EXAMPLE 4: extra_body escape hatch for vendor-specific fields")
    print("=" * 72)
    body = {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "..."}]}
    apply_structured_output(
        body,
        options={
            "json_mode": True,
            "json_schema": schema,
            "guided_decoding": "outlines",
            # vLLM sampling-knob examples; OpenRouter provider-preference example.
            "extra_body": {
                "min_p": 0.05,
                "repetition_penalty": 1.05,
                "provider": {"order": ["Hyperbolic", "Together"]},
            },
        },
    )
    _print_body("payload", body)

    print("=" * 72)
    print("How to use in practice")
    print("=" * 72)
    print(
        "  from prompture import extract_with_model\n"
        "  from prompture.drivers.openai_compatible_driver import OpenAICompatibleDriver\n"
        "\n"
        '  driver = OpenAICompatibleDriver(endpoint="http://my-vllm:8000/v1", model="...")\n'
        "\n"
        "  result = extract_with_model(\n"
        "      Person,\n"
        '      "Alice Smith, 32, lives at 1 Main St, Springfield, IL 62701",\n'
        '      model_name="openai_compatible/local-vllm",\n'
        '      options={"guided_decoding": "outlines"},  # FSM-constrained sampling\n'
        "  )\n"
    )


if __name__ == "__main__":
    main()
