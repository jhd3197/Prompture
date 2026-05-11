"""
Example: image generation across providers — Runway (gpt_image_2) and OpenAI (gpt-image-1).

Demonstrates:
1. Picking an image-gen driver by ``"provider/model"`` string.
2. Calling Runway ``POST /v1/text_to_image`` with the ``gpt_image_2`` model,
   ``ratio=1920:1088``, and ``quality=high`` — i.e. the exact contract from the
   curl example in the Runway docs.
3. Calling OpenAI's GPT image API (``gpt-image-1`` today; the same code accepts
   ``gpt-image-2`` automatically when OpenAI ships it).
4. Saving each result to a PNG next to this script.

Setup:
    RUNWAY_API_KEY     — Runway API key (RUNWAYML_API_SECRET also accepted).
    OPENAI_API_KEY     — OpenAI API key.

By default this runs whichever providers are configured. Skip the second
provider by unsetting its key; the script will tell you what it skipped.
"""

from __future__ import annotations

import base64
import os
import sys
from datetime import datetime
from pathlib import Path

from prompture.drivers.img_gen_registry import get_img_gen_driver_for_model

PROMPT = (
    "A cinematic wide shot of a neon-lit Tokyo alleyway at night in the rain, "
    "reflections on wet pavement, moody atmosphere, shot on 35mm film"
)

OUT_DIR = Path(__file__).parent / "runway_image_outputs"


# ── Section: helpers ────────────────────────────────────────────────────────


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _save_image(image, path: Path) -> Path:
    """Persist an ``ImageContent`` to disk.

    Runway returns URL-backed images; OpenAI returns base64-backed images.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.source_type == "url" and image.url:
        import urllib.request

        with urllib.request.urlopen(image.url) as resp:
            path.write_bytes(resp.read())
    else:
        path.write_bytes(base64.b64decode(image.data))

    return path


def _print_result(label: str, result: dict) -> None:
    meta = result["meta"]
    print(f"\n── {label} ──")
    print(f"  model:    {meta['model_name']}")
    print(f"  size:     {meta.get('size')}")
    print(f"  count:    {meta['image_count']}")
    print(f"  cost:     ${meta['cost']:.4f}")
    if meta.get("task_id"):
        print(f"  task_id:  {meta['task_id']}")
    if meta.get("status"):
        print(f"  status:   {meta['status']}")


# ── Section: Runway (gpt_image_2) ───────────────────────────────────────────


def run_runway() -> None:
    if not (os.getenv("RUNWAY_API_KEY") or os.getenv("RUNWAYML_API_SECRET")):
        print("• Runway: SKIPPED (set RUNWAY_API_KEY or RUNWAYML_API_SECRET to enable)")
        return

    print("• Runway: generating with gpt_image_2 (ratio=1920:1088, quality=high)...")
    driver = get_img_gen_driver_for_model("runway/gpt_image_2")
    result = driver.generate_image(
        PROMPT,
        {
            "ratio": "1920:1088",
            "quality": "high",
            # poll_interval/timeout default to 5s / 600s — fine for this demo.
        },
    )
    _print_result("Runway / gpt_image_2", result)

    for i, image in enumerate(result["images"], start=1):
        out = OUT_DIR / f"runway-gpt_image_2-{_timestamp()}-{i}.png"
        saved = _save_image(image, out)
        print(f"  saved:    {saved}")


# ── Section: OpenAI (gpt-image-1, with gpt-image-2 forward-compat) ──────────


def run_openai() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("• OpenAI: SKIPPED (set OPENAI_API_KEY to enable)")
        return

    # gpt-image-2 is not yet published in OpenAI's public catalog; use
    # gpt-image-1, which has identical request semantics. Override via
    # PROMPTURE_OPENAI_IMAGE_MODEL=gpt-image-2 the day OpenAI ships it.
    model = os.getenv("PROMPTURE_OPENAI_IMAGE_MODEL", "gpt-image-1")

    print(f"• OpenAI: generating with {model} (size=1024x1024, quality=high)...")
    driver = get_img_gen_driver_for_model(f"openai/{model}")
    result = driver.generate_image(
        PROMPT,
        {
            "size": "1024x1024",
            "quality": "high",
            "n": 1,
        },
    )
    _print_result(f"OpenAI / {model}", result)

    for i, image in enumerate(result["images"], start=1):
        out = OUT_DIR / f"openai-{model}-{_timestamp()}-{i}.png"
        saved = _save_image(image, out)
        print(f"  saved:    {saved}")


# ── Section: entry point ────────────────────────────────────────────────────


def main() -> None:
    print(f"Prompt: {PROMPT!r}")
    print(f"Output directory: {OUT_DIR}")

    try:
        run_runway()
    except Exception as exc:
        print(f"  Runway error: {exc}", file=sys.stderr)

    try:
        run_openai()
    except Exception as exc:
        print(f"  OpenAI error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
