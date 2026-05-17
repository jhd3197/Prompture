"""
Example: Runway video generation — text/image/video → video.

Demonstrates all three Runway video endpoints through a single driver:

1. ``text_to_video``  with ``gen4.5``         — a cinematic prompt.
2. ``image_to_video`` with ``gen4_turbo``     — animate a still image.
3. ``video_to_video`` with ``gen4_aleph``     — restyle a video.

The script chains them: the v2v step uses the t2v output (a Runway-signed
URL, valid for 24–48 h) as its input, so the whole demo is self-contained.

Setup:
    RUNWAY_API_KEY (or RUNWAYML_API_SECRET) must be set.

Each scenario waits for the task to finish and downloads the MP4. Set
``PROMPTURE_RUNWAY_SCENARIO`` to ``t2v``, ``i2v``, or ``v2v`` to run just one.
If you run ``v2v`` solo, override ``PROMPTURE_RUNWAY_VIDEO_URL`` to point at
an existing 720p MP4 — there's no prior t2v output to chain from.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from prompture.drivers.runway_video_gen_driver import RunwayVideoGenDriver
from prompture.drivers.video_gen_registry import get_video_gen_driver_for_model

OUT_DIR = Path(__file__).parent / "runway_video_outputs"

IMAGE_URL = os.getenv(
    "PROMPTURE_RUNWAY_IMAGE_URL",
    "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=1280",
)
# Fallback only — when running v2v in isolation, no chained output exists.
FALLBACK_VIDEO_URL = os.getenv(
    "PROMPTURE_RUNWAY_VIDEO_URL",
    "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4",
)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _save_video(url: str, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}-{_ts()}.mp4"
    with urllib.request.urlopen(url) as resp:
        path.write_bytes(resp.read())
    return path


def _estimate_cost(model: str, duration: int | None) -> float:
    pricing = RunwayVideoGenDriver.VIDEO_PRICING.get(model, {})
    return round(pricing.get("per_second", 0.0) * (duration or 0), 4)


def _print(label: str, result: dict) -> None:
    meta = result["meta"]
    print(f"── {label} ──")
    print(f"  model:    {meta['model_name']}")
    print(f"  mode:     {meta.get('mode')}")
    print(f"  duration: {meta.get('duration_seconds')}s")
    print(f"  ratio:    {meta.get('aspect_ratio')}")
    print(f"  task:     {meta.get('task_id')}")
    print(f"  status:   {meta.get('status')}")
    print(f"  cost:     ${meta['cost']:.4f}")


# State that scenarios can pass to each other (e.g. the t2v output URL
# becomes the v2v input).
chained: dict[str, str] = {}


def run_text_to_video() -> None:
    model, duration = "gen4.5", 5
    print(f"\n• {model} / text_to_video  (estimated ${_estimate_cost(model, duration):.4f})")
    driver = get_video_gen_driver_for_model(f"runway/{model}")
    result = driver.generate_video(
        "wide cinematic shot of a crystal-powered rocket launching from red desert dunes, "
        "slow camera push-in, golden hour light",
        {"ratio": "1280:720", "duration": duration},
    )
    _print("text_to_video / gen4.5", result)
    if result["videos"]:
        url = result["videos"][0].url
        chained["t2v_url"] = url
        path = _save_video(url, "t2v-gen4.5")
        print(f"  saved:    {path}")


def run_image_to_video() -> None:
    model, duration = "gen4_turbo", 5
    print(f"\n• {model} / image_to_video  (estimated ${_estimate_cost(model, duration):.4f})")
    driver = get_video_gen_driver_for_model(f"runway/{model}")
    result = driver.generate_video(
        "slow dolly-in across the scene, soft drifting clouds, cinematic",
        {"image": IMAGE_URL, "ratio": "1280:720", "duration": duration},
    )
    _print("image_to_video / gen4_turbo", result)
    if result["videos"]:
        url = result["videos"][0].url
        chained.setdefault("i2v_url", url)
        path = _save_video(url, "i2v-gen4_turbo")
        print(f"  saved:    {path}")


def run_video_to_video() -> None:
    # Prefer the t2v output (Runway → Runway always fetches cleanly), then
    # i2v output, then the external fallback.
    source = chained.get("t2v_url") or chained.get("i2v_url") or FALLBACK_VIDEO_URL
    source_label = (
        "t2v output"
        if source == chained.get("t2v_url")
        else "i2v output"
        if source == chained.get("i2v_url")
        else "external URL"
    )

    per_sec = RunwayVideoGenDriver.VIDEO_PRICING.get("gen4_aleph", {}).get("per_second", 0.0)
    print(f"\n• gen4_aleph / video_to_video  (≈ ${per_sec:.2f}/sec of input)")
    print(f"  input:    {source_label}")

    driver = get_video_gen_driver_for_model("runway/gen4_aleph")
    result = driver.generate_video(
        "restyle the scene to a snow-covered winter version, soft falling snow",
        {"video": source},
    )
    _print("video_to_video / gen4_aleph", result)
    if result["videos"]:
        path = _save_video(result["videos"][0].url, "v2v-gen4_aleph")
        print(f"  saved:    {path}")


SCENARIOS = {
    "t2v": run_text_to_video,
    "i2v": run_image_to_video,
    "v2v": run_video_to_video,
}


def main() -> None:
    if not (os.getenv("RUNWAY_API_KEY") or os.getenv("RUNWAYML_API_SECRET")):
        raise RuntimeError("Set RUNWAY_API_KEY (or RUNWAYML_API_SECRET) before running.")

    print(f"Output: {OUT_DIR}")
    print("Each scenario polls until SUCCEEDED — this can take 30s–3min per video.")

    only = os.getenv("PROMPTURE_RUNWAY_SCENARIO", "").lower()
    targets = [only] if only in SCENARIOS else list(SCENARIOS)

    for name in targets:
        try:
            SCENARIOS[name]()
        except Exception as exc:
            print(f"  {name} error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
