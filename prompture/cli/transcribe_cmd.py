"""``prompture transcribe`` — URL / file → transcript (and optional summary)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ..security.redaction import scrub_secrets


@click.command("transcribe")
@click.argument("source")
@click.option("--model", default="auto", show_default=True, help='STT model: "auto", "groq", "openai/whisper-1", ...')
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write to this file (.json → full JSON, anything else → Markdown). Default: stdout.",
)
@click.option(
    "--allow-provider-fallback",
    is_flag=True,
    default=False,
    help="Let the audio go to another configured STT provider if the first one fails.",
)
@click.option("--language", default=None, help='Language hint, e.g. "en".')
@click.option("--summary", is_flag=True, default=False, help="Also summarize the transcript with key points.")
@click.option(
    "--summary-model",
    default=None,
    help="LLM for --summary (default: openai/gpt-4o-mini).",
)
@click.option("--focus", default=None, help="Topic to focus the --summary on.")
@click.option("--no-timestamps", is_flag=True, default=False, help="Plain text without [HH:MM:SS] prefixes.")
@click.option("--max-chunks", type=int, default=None, help="Cap on 10-minute chunks (default 24 ≈ 4 h).")
def transcribe(
    source: str,
    model: str,
    out_path: str | None,
    allow_provider_fallback: bool,
    language: str | None,
    summary: bool,
    summary_model: str | None,
    focus: str | None,
    no_timestamps: bool,
    max_chunks: int | None,
) -> None:
    """Transcribe SOURCE (URL or local file) with timestamps."""
    from ..media.understand import summarize_transcript
    from ..media.understand import transcribe as run_transcribe
    from ..media.understand.summarize import DEFAULT_SUMMARY_MODEL

    kwargs: dict[str, object] = {}
    if max_chunks is not None:
        kwargs["max_chunks"] = max_chunks
    try:
        transcript = run_transcribe(
            source,
            model=model,
            allow_provider_fallback=allow_provider_fallback,
            language=language,
            **kwargs,  # type: ignore[arg-type]
        )
        result = None
        if summary:
            result = summarize_transcript(transcript, model=summary_model or DEFAULT_SUMMARY_MODEL, focus=focus)
    except Exception as exc:
        hint = getattr(exc, "hint", None)
        msg = f"{type(exc).__name__}: {exc}"
        if hint and str(hint) not in msg:
            msg += f"\nFix: {hint}"
        raise click.ClickException(scrub_secrets(msg)) from exc

    if out_path and out_path.lower().endswith(".json"):
        payload = result.to_dict() if result is not None else transcript.to_dict()
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        parts = []
        if result is not None:
            parts.append(result.to_markdown())
        parts.append(transcript.to_markdown(timestamps=not no_timestamps))
        text = "\n".join(parts)

    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        cost = result.cost if result is not None else transcript.cost
        click.echo(
            f"Wrote {out_path} ({len(transcript.segments)} segments, {transcript.model}, ${cost:.4f})",
            err=True,
        )
    else:
        click.echo(text)


COMMANDS = [transcribe]
