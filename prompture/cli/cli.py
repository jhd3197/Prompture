import json

import click

from ..drivers import OllamaDriver, get_driver
from .formatters import format_table
from .runner import run_suite_from_spec


@click.group()
def cli() -> None:
    """Prompture CLI -- structured LLM output toolkit."""
    pass


@cli.command()
@click.argument("specfile", type=click.Path(exists=True))
@click.argument("outfile", type=click.Path())
def run(specfile: str, outfile: str) -> None:
    """Run a spec JSON and save report."""
    with open(specfile, encoding="utf-8") as fh:
        spec = json.load(fh)
    # Use Ollama as default driver since it can run locally
    drivers = {"ollama": OllamaDriver(endpoint="http://localhost:11434", model="gemma:latest")}
    report = run_suite_from_spec(spec, drivers)  # type: ignore[arg-type]
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    click.echo(f"Report saved to {outfile}")


def _build_drivers(providers: list[str]) -> dict[str, object]:
    """Instantiate a driver for each requested provider."""
    drivers: dict[str, object] = {}
    for name in providers:
        try:
            drivers[name] = get_driver(name)
        except Exception as exc:
            click.echo(f"Warning: could not load driver for '{name}': {exc}", err=True)
    return drivers


@cli.command("test-suite")
@click.argument("specfile", type=click.Path(exists=True))
@click.option("--providers", default=None, help="Comma-separated provider names (e.g., openai,ollama,groq).")
@click.option("--models", default=None, help="Comma-separated model strings to override spec models.")
@click.option("--format", "fmt", type=click.Choice(["json", "table"]), default="table", help="Output format.")
@click.option("-o", "--output", "outfile", default=None, type=click.Path(), help="Save JSON report to file.")
def test_suite(specfile: str, providers: str | None, models: str | None, fmt: str, outfile: str | None) -> None:
    """Run a cross-model test suite from a spec file.

    Uses the driver registry to auto-detect configured providers.
    Override spec models with --providers or --models.

    Examples:

      prompture test-suite specs/basic_extraction.json

      prompture test-suite specs/basic_extraction.json --providers openai,groq

      prompture test-suite specs/basic_extraction.json --models openai/gpt-4o-mini,ollama/llama3.1:8b --format json -o report.json
    """
    with open(specfile, encoding="utf-8") as fh:
        spec = json.load(fh)

    # Override spec models if --models is provided
    if models:
        model_list = [m.strip() for m in models.split(",")]
        spec["models"] = []
        for model_str in model_list:
            provider = model_str.split("/")[0] if "/" in model_str else model_str
            spec["models"].append({"id": model_str, "driver": provider, "options": {}})

    # Determine which providers we need drivers for
    needed_providers = {m["driver"] for m in spec.get("models", [])}

    if providers:
        # User explicitly requested specific providers — filter to those
        requested = {p.strip() for p in providers.split(",")}
        needed_providers = needed_providers & requested

    drivers = _build_drivers(list(needed_providers))

    if not drivers:
        click.echo("Error: no drivers could be loaded. Check your provider configuration.", err=True)
        raise SystemExit(1)

    click.echo(f"Running suite with drivers: {', '.join(sorted(drivers.keys()))}")
    report = run_suite_from_spec(spec, drivers)  # type: ignore[arg-type]

    if outfile:
        with open(outfile, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        click.echo(f"JSON report saved to {outfile}")

    if fmt == "table":
        click.echo("")
        click.echo(format_table(report))
    elif fmt == "json" and not outfile:
        click.echo(json.dumps(report, indent=2, ensure_ascii=False))


@cli.command()
@click.option("--model", default="openai/gpt-4o-mini", help="Model string (provider/model).")
@click.option("--system-prompt", default=None, help="System prompt for conversations.")
@click.option("--host", default="0.0.0.0", help="Bind host.")
@click.option("--port", default=9471, type=int, help="Bind port.")
@click.option("--cors-origins", default=None, help="Comma-separated CORS origins (use * for all).")
def serve(model: str, system_prompt: str, host: str, port: int, cors_origins: str) -> None:
    """Start an API server wrapping AsyncConversation.

    Requires the 'serve' extra: pip install prompture[serve]
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn not installed. Run: pip install prompture[serve]", err=True)
        raise SystemExit(1) from None

    from .server import create_app

    origins = [o.strip() for o in cors_origins.split(",")] if cors_origins else None
    app = create_app(
        model_name=model,
        system_prompt=system_prompt,
        cors_origins=origins,
    )

    click.echo(f"Starting Prompture server on {host}:{port} with model {model}")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("output_dir", type=click.Path())
@click.option("--name", default="my_app", help="Project name.")
@click.option("--model", default="openai/gpt-4o-mini", help="Default model string.")
@click.option("--docker/--no-docker", default=True, help="Include Dockerfile.")
def scaffold(output_dir: str, name: str, model: str, docker: bool) -> None:
    """Generate a standalone FastAPI project using Prompture.

    Requires the 'scaffold' extra: pip install prompture[scaffold]
    """
    try:
        from ..scaffold.generator import scaffold_project
    except ImportError:
        click.echo("Error: jinja2 not installed. Run: pip install prompture[scaffold]", err=True)
        raise SystemExit(1) from None

    scaffold_project(
        output_dir=output_dir,
        project_name=name,
        model_name=model,
        include_docker=docker,
    )
    click.echo(f"Project scaffolded at {output_dir}")
