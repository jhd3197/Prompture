"""
Example: Grok video generation.

This script demonstrates:
1. Selecting a video generation driver by provider/model string.
2. Starting a text-to-video request with xAI Grok Imagine Video.
3. Running a fast local smoke test by returning the request ID without polling.

Setup:
    Set GROK_API_KEY or XAI_API_KEY in your environment or .env file.
    Set PROMPTURE_VIDEO_POLL=true to wait for the finished video URL.
"""

from __future__ import annotations

import os

from prompture import get_video_gen_driver_for_model

MODEL = "grok/grok-imagine-video"
PROMPT = "wide cinematic shot of a crystal-powered rocket launching from red desert dunes"


def main() -> None:
    if not (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")):
        raise RuntimeError("Set GROK_API_KEY or XAI_API_KEY before running this example.")

    poll = os.getenv("PROMPTURE_VIDEO_POLL", "false").lower() == "true"
    driver = get_video_gen_driver_for_model(MODEL)
    result = driver.generate_video(
        PROMPT,
        {
            "duration": 8,
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "poll": poll,
        },
    )

    meta = result["meta"]
    print("Video request metadata:")
    print(f"Request ID: {meta['request_id']}")
    print(f"Status: {meta['status']}")
    print(f"Model: {meta['model_name']}")
    print(f"Estimated cost: ${meta['cost']:.6f}")

    if result["videos"]:
        print(f"Video URL: {result['videos'][0].url}")
    else:
        print("Polling disabled. Re-run with PROMPTURE_VIDEO_POLL=true to wait for the video URL.")


if __name__ == "__main__":
    main()
