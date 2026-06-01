"""Negative-prompt composition — route avoidance guidance to wherever the
target image model can actually use it.

Some image providers accept a native ``negative_prompt`` option (their driver
class declares ``supports_negative_prompt = True``); the rest only consume the
positive text prompt. :func:`compose_negative_prompt` sets the native option
when it's supported and otherwise folds the guidance into the prompt as an
``Avoid: …`` clause, so the negative intent is always expressed — on any
provider, with no per-provider branching in caller code.
"""

from __future__ import annotations

import re
from typing import Any


def model_supports_negative_prompt(model: str) -> bool:
    """Whether ``model``'s driver declares native negative-prompt support.

    Reads the driver *class* attribute without instantiating it (no API key
    needed). Unknown or unresolvable models return ``False`` — the safe default,
    since the caller then folds the negative into the prompt text.
    """
    if not model:
        return False
    try:
        from ..infra.discovery import _img_gen_driver_class

        provider = model.split("/", 1)[0]
        cls = _img_gen_driver_class(provider)
        return bool(getattr(cls, "supports_negative_prompt", False)) if cls is not None else False
    except Exception:
        return False


def compose_negative_prompt(
    prompt: str,
    negative: str,
    options: dict[str, Any] | None = None,
    *,
    model: str | None = None,
    native_supported: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Route negative guidance and return ``(prompt, options)`` ready to generate.

    - When the target supports a native negative prompt, set
      ``options['negative_prompt']`` (merged with any existing value) and leave
      the prompt unchanged.
    - Otherwise append an ``Avoid: …`` clause to the prompt.

    Args:
        prompt: The positive image prompt.
        negative: Things to avoid. An empty value is a no-op.
        options: Existing generation options (not mutated; a copy is returned).
        native_supported: Force the routing decision. When ``None`` it is
            resolved from ``model`` via :func:`model_supports_negative_prompt`,
            defaulting to the prompt-fold path when no model is given.
    """
    opts: dict[str, Any] = dict(options or {})
    neg = (negative or "").strip()
    if not neg:
        return prompt, opts

    if native_supported is None:
        native_supported = model_supports_negative_prompt(model) if model else False

    if native_supported:
        existing = str(opts.get("negative_prompt") or "").strip()
        opts["negative_prompt"] = f"{existing}, {neg}".strip(", ") if existing else neg
        return prompt, opts

    base = (prompt or "").rstrip()
    if base and base[-1] not in ".!?":
        base += "."
    clause = f"Avoid: {neg.rstrip('.')}."
    final = re.sub(r"\s+", " ", f"{base} {clause}".strip())
    return final, opts
