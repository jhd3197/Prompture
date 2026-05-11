"""
Example: Runway audio — TTS, sound effects, and audio-transform endpoints.

Demonstrates four endpoints, each polling to completion:

1. ``POST /v1/text_to_speech``    — Maya voice speaks a line.
2. ``POST /v1/sound_effect``      — generate a 5-second rain sound effect.
3. ``POST /v1/voice_dubbing``     — dub the sample clip to Spanish.
4. ``POST /v1/voice_isolation``   — isolate voice from the sample clip.

Setup:
    RUNWAY_API_KEY (or RUNWAYML_API_SECRET) must be set.

Override the sample audio used by #3 and #4 with
``PROMPTURE_RUNWAY_AUDIO_URL=<https-url-to-mp3>``. The default is a public
~15 s clip; for meaningful dubbing, point this at a speech recording.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

from prompture.drivers import RunwayAudioTransformDriver
from prompture.drivers.audio_registry import get_tts_driver_for_model
from prompture.drivers.runway_tts_driver import RunwayTTSDriver

OUT_DIR = Path(__file__).parent / "runway_audio_outputs"

# Public sample clip used by voice_dubbing and voice_isolation. Stable URL,
# small file, well under Runway's input limits.
AUDIO_URL = os.getenv(
    "PROMPTURE_RUNWAY_AUDIO_URL",
    "https://download.samplelib.com/mp3/sample-15s.mp3",
)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _ext_for(media_type: str) -> str:
    if "wav" in media_type:
        return "wav"
    if "ogg" in media_type:
        return "ogg"
    if "mp4" in media_type:
        return "m4a"
    return "mp3"


def _save(audio: bytes, media_type: str, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}-{_ts()}.{_ext_for(media_type)}"
    path.write_bytes(audio)
    return path


def _print(label: str, result: dict) -> None:
    meta = result["meta"]
    print(f"── {label} ──")
    print(f"  model:    {meta['model_name']}")
    print(f"  task:     {meta.get('task_id')}")
    print(f"  status:   {meta.get('status')}")
    if "cost" in meta:
        print(f"  cost:     ${meta['cost']:.4f}")


def _estimate_tts_cost(model: str, *, characters: int = 0, duration: float = 0.0) -> float:
    pricing = RunwayTTSDriver.AUDIO_PRICING.get(model, {})
    cost = characters * pricing.get("per_character", 0.0)
    cost += duration * pricing.get("per_second", 0.0)
    return round(cost, 4)


def run_text_to_speech() -> None:
    text = "Hello! This is the Runway text-to-speech demo wired into Prompture."
    est = _estimate_tts_cost("eleven_multilingual_v2", characters=len(text))
    print(f"\n• text_to_speech (Maya)  (estimated ${est:.4f} for {len(text)} chars)")
    driver = get_tts_driver_for_model("runway/eleven_multilingual_v2")
    result = driver.synthesize(text, {"voice": "Maya"})
    _print("text_to_speech / Maya", result)
    if result["audio"]:
        path = _save(result["audio"], result["media_type"], "tts-maya")
        print(f"  saved:    {path}")


def run_sound_effect() -> None:
    duration = 5
    est = _estimate_tts_cost("eleven_text_to_sound_v2", duration=duration)
    print(f"\n• sound_effect  (estimated ${est:.4f} for {duration}s)")
    driver = get_tts_driver_for_model("runway/eleven_text_to_sound_v2")
    result = driver.synthesize(
        "Heavy tropical rain on a metal roof with distant thunder",
        {"duration": duration, "loop": False},
    )
    _print("sound_effect", result)
    if result["audio"]:
        path = _save(result["audio"], result["media_type"], "sfx-rain")
        print(f"  saved:    {path}")


def run_voice_dubbing() -> None:
    # Voice dubbing is priced per-second of input; the clip duration isn't
    # known until Runway fetches it, so we just note the input URL.
    print("\n• voice_dubbing → es")
    print(f"  input:    {AUDIO_URL}")
    driver = RunwayAudioTransformDriver()
    result = driver.dub(AUDIO_URL, target_lang="es", drop_background_audio=False)
    _print("voice_dubbing → es", result)
    if result["audio"]:
        path = _save(result["audio"], result["media_type"], "dub-es")
        print(f"  saved:    {path}")


def run_voice_isolation() -> None:
    print("\n• voice_isolation")
    print(f"  input:    {AUDIO_URL}")
    driver = RunwayAudioTransformDriver()
    result = driver.isolate(AUDIO_URL)
    _print("voice_isolation", result)
    if result["audio"]:
        path = _save(result["audio"], result["media_type"], "isolated")
        print(f"  saved:    {path}")


def main() -> None:
    if not (os.getenv("RUNWAY_API_KEY") or os.getenv("RUNWAYML_API_SECRET")):
        raise RuntimeError("Set RUNWAY_API_KEY (or RUNWAYML_API_SECRET) before running.")

    print(f"Output: {OUT_DIR}")
    print("Each step polls until SUCCEEDED — total runtime can be a few minutes.")

    for step in (run_text_to_speech, run_sound_effect, run_voice_dubbing, run_voice_isolation):
        try:
            step()
        except Exception as exc:
            print(f"  {step.__name__} error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
