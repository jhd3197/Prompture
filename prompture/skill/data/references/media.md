# Video and audio: transcripts and summaries

## CLI

```bash
prompture transcribe "https://www.youtube.com/watch?v=<id>"     # Markdown with [HH:MM:SS]
prompture transcribe talk.mp3 --out talk.json                    # full JSON
prompture transcribe <url> --summary [--summary-model openai/gpt-4o-mini] [--focus "pricing"]
```

## Python

```python
from prompture.media.understand import transcribe, summarize_media, transcription_available

if transcription_available():
    t = transcribe("https://example.com/episode.mp3")      # URL or local file
    print(t.provider, t.duration_s, t.cost)
    print(t.to_markdown())
    s = summarize_media(t, model="openai/gpt-4o-mini")
    for kp in s.key_points:
        print(kp.timestamp, kp.text)
```

Agent tools: `transcribe_media`, `summarize_media` (`tools=["web:media"]` or
`prompture.media.agent_tools.register_media_tools(registry)`).

## Provider chain and privacy

`model="auto"` uses the **first** configured STT key:
`GROQ_API_KEY` (Groq Whisper) ▸ `OPENAI_API_KEY` (Whisper) ▸ `ELEVENLABS_API_KEY`.
Reorder with `PROMPTURE_STT_PROVIDERS=openai,groq`.

Audio goes to **one** provider only. A second provider is tried only with
`allow_provider_fallback=True` / `--allow-provider-fallback` — ask the user
first.

## Pipeline and requirements

- Direct audio URLs and local files: just an STT key. Small files in a
  supported format skip ffmpeg.
- Long media: `ffmpeg` downmixes to 16 kHz mono and splits into ~10-minute
  chunks; timestamps are stitched. Caps: source size, ~4 h, total upload bytes.
- Video/podcast pages: need a working `yt-dlp` (`pip install "prompture[media]"`).
- For YouTube, prefer `read_url(url)` first: it uses the free transcript API
  and subtitles before paying for transcription.

## When it fails

`prompture doctor --only media --only binaries` shows `transcription`,
`media_download`, `ffmpeg`, `ffprobe` and `yt-dlp`. A `broken` yt-dlp means a
stale shim on PATH — reinstall it (`pip install -U yt-dlp`). An invalid STT key
shows in the row with its env var; fix it or reorder with
`PROMPTURE_STT_PROVIDERS`.
