"""``prompture research "question"`` — cited multi-source web research.

Progress goes to stderr (silence it with ``--quiet``); the report goes to
stdout as Markdown, or as JSON with ``--json``. Without ``--model`` the model
comes from ``PROMPTURE_RESEARCH_MODEL`` / ``PROMPTURE_DEFAULT_MODEL`` or the
first configured provider's cheap model.
"""

from __future__ import annotations

import json

import click


def _progress(ev: object) -> None:
    event_type = getattr(ev, "event_type", "")
    if event_type == "done":
        return
    elapsed = getattr(ev, "elapsed_s", 0.0)
    click.echo(f"[{elapsed:6.1f}s] {event_type:<10} {getattr(ev, 'message', '')}", err=True)


@click.command("research")
@click.argument("question", nargs=-1, required=True)
@click.option(
    "--depth",
    type=click.Choice(["quick", "standard", "deep"]),
    default="standard",
    show_default=True,
    help="Sub-questions, pages read and budget preset.",
)
@click.option("--model", default=None, help="provider/model for planning and synthesis (default: auto-detect).")
@click.option("--json", "as_json", is_flag=True, help="Print the report as JSON.")
@click.option("--max-fetches", type=click.IntRange(min=0), default=None, help="Maximum pages to open.")
@click.option("--max-cost", type=click.FloatRange(min=0), default=None, help="Maximum USD spent on LLM calls.")
@click.option("--max-tokens", type=click.IntRange(min=0), default=None, help="Maximum LLM tokens.")
@click.option("--timeout", type=click.FloatRange(min=1), default=None, help="Wall-clock limit in seconds.")
@click.option("--compress-sources", is_flag=True, help="Send sources to synthesis as a compact TOON table.")
@click.option("--no-platforms", is_flag=True, help="Skip GitHub / Hacker News / arXiv / YouTube search.")
@click.option("--no-packs", is_flag=True, help="Skip domain tool packs.")
@click.option("--no-transcribe", is_flag=True, help="Never transcribe video or audio sources.")
@click.option("-q", "--quiet", is_flag=True, help="No progress lines on stderr.")
def research(
    question: tuple[str, ...],
    depth: str,
    model: str | None,
    as_json: bool,
    max_fetches: int | None,
    max_cost: float | None,
    max_tokens: int | None,
    timeout: float | None,
    compress_sources: bool,
    no_platforms: bool,
    no_packs: bool,
    no_transcribe: bool,
    quiet: bool,
) -> None:
    """Research QUESTION on the web and print a cited report."""
    from ..exceptions import ConfigurationError
    from ..research import ResearchAgent, ResearchBudget, ResearchTools

    text = " ".join(question).strip()
    if not text:
        raise click.UsageError("QUESTION must not be empty.")
    budget = ResearchBudget(max_fetches=max_fetches, max_cost=max_cost, max_tokens=max_tokens, timeout_s=timeout)
    tools = ResearchTools(
        enable_platforms=not no_platforms,
        enable_packs=not no_packs,
        enable_transcription=not no_transcribe,
    )
    try:
        agent = ResearchAgent(
            model,
            depth=depth,
            budget=budget,
            tools=tools,
            on_event=None if quiet else _progress,
            compress_sources=compress_sources,
            transcribe_media=not no_transcribe,
        )
    except ConfigurationError as exc:
        raise click.ClickException(f"{exc} (or pass --model provider/model)") from None
    try:
        report = agent.run(text)
    except Exception as exc:
        raise click.ClickException(f"Research failed: {type(exc).__name__}: {exc}") from None
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        click.echo(report.to_markdown())


COMMANDS = [research]

__all__ = ["COMMANDS", "research"]
