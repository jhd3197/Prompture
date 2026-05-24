"""Shared request-shaping helpers for OpenAI-compatible drivers.

The OpenAI Chat Completions wire format is the de-facto standard for
local inference servers (vLLM, LMStudio, Ollama-in-OpenAI-mode,
Aphrodite, OpenRouter, Together, Fireworks, ...). Most of those
servers also accept *vendor-specific* extra fields for advanced
features the OpenAI spec doesn't cover — most importantly,
**logit-level constrained decoding** for guaranteed schema validity.

This module centralises the "translate Prompture-side options into
the right OpenAI-compatible request payload" logic so every driver
applies the same defaults and respects the same escape hatches.

Two knobs callers care about:

* ``guided_decoding`` (``bool`` | ``str``) — when truthy and a
  ``json_schema`` is provided, also writes vLLM's ``guided_json`` (and
  optionally ``guided_decoding_backend``) at the top level of the
  request body. Servers that don't recognise the keys ignore them, so
  this is safe to send to any OpenAI-compatible endpoint.

* ``extra_body`` (``dict``) — caller-supplied dict merged into the
  request body verbatim. The OpenAI Python SDK exposes the same
  ``extra_body`` escape hatch; Prompture mirrors it so users can pass
  any vendor field we don't know about yet
  (``min_p``, ``repetition_penalty``, ``guided_grammar``, ...).
"""

from __future__ import annotations

from typing import Any

from ..infra.cost_mixin import prepare_strict_schema

#: vLLM-style guided-decoding backends. ``True`` accepts the server
#: default; passing a string pins a specific implementation.
_VLLM_BACKENDS: frozenset[str] = frozenset({"outlines", "lm-format-enforcer", "xgrammar"})


def build_response_format(json_schema: dict[str, Any] | None) -> dict[str, Any]:
    """Build the OpenAI ``response_format`` object for a strict json_schema mode.

    Returns the dict to assign to ``data["response_format"]``. When
    ``json_schema`` is ``None`` the caller should fall back to
    ``{"type": "json_object"}`` for loose JSON mode (we don't return
    that here because it's a one-liner at the call site).
    """
    schema_copy = prepare_strict_schema(json_schema) if json_schema is not None else {}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "extraction",
            "strict": True,
            "schema": schema_copy,
        },
    }


def apply_guided_decoding(
    payload: dict[str, Any],
    *,
    json_schema: dict[str, Any] | None,
    guided_decoding: Any,
) -> None:
    """Inject vLLM-style top-level ``guided_json`` fields when requested.

    Args:
        payload: The request body to mutate in place. Must be the
            outer dict the driver sends (where ``model`` /
            ``messages`` already live).
        json_schema: The schema the user wants to enforce. ``None`` is
            a no-op.
        guided_decoding: User-supplied flag.

            * ``False`` / ``None`` — no-op, leaves ``payload`` alone.
            * ``True`` — adds ``guided_json=<schema>`` (server picks
              its own backend).
            * One of ``"outlines"`` / ``"lm-format-enforcer"`` /
              ``"xgrammar"`` — also sets
              ``guided_decoding_backend`` to that string.

    Unknown string values are accepted and passed through verbatim —
    vLLM releases occasionally add new backends, and Prompture
    shouldn't have to ship a release every time.
    """
    if not guided_decoding or json_schema is None:
        return
    payload["guided_json"] = json_schema
    if isinstance(guided_decoding, str):
        # Honour explicit backend choice. We don't reject unknown
        # strings; future vLLM versions can introduce new backends.
        payload["guided_decoding_backend"] = guided_decoding


def merge_extra_body(payload: dict[str, Any], options: dict[str, Any]) -> None:
    """Mirror the OpenAI SDK's ``extra_body`` escape hatch.

    Spreads ``options["extra_body"]`` (when it's a dict) into
    ``payload``. User-supplied keys win — this is intentional so the
    escape hatch can override anything Prompture set above, including
    ``response_format``.
    """
    extra = options.get("extra_body")
    if isinstance(extra, dict):
        payload.update(extra)


def apply_structured_output(
    payload: dict[str, Any],
    options: dict[str, Any],
) -> None:
    """One-call wrapper that's safe to drop at the end of every driver's
    request-shaping block.

    Reads ``json_mode`` / ``json_schema`` / ``guided_decoding`` /
    ``extra_body`` out of ``options``, mutates ``payload`` in place.
    Existing drivers can keep their own ``response_format`` logic if
    they prefer; this helper is opt-in.
    """
    json_schema = options.get("json_schema")
    if options.get("json_mode"):
        if json_schema is not None:
            payload["response_format"] = build_response_format(json_schema)
        else:
            payload["response_format"] = {"type": "json_object"}
    apply_guided_decoding(
        payload,
        json_schema=json_schema,
        guided_decoding=options.get("guided_decoding"),
    )
    merge_extra_body(payload, options)


__all__ = [
    "apply_guided_decoding",
    "apply_structured_output",
    "build_response_format",
    "merge_extra_body",
]
