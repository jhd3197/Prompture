import json

import click

from .drivers import OllamaDriver
from .runner import run_suite_from_spec


@click.group()
def cli():
    """Prompture CLI -- structured LLM output toolkit."""
    pass


@cli.command()
@click.argument("specfile", type=click.Path(exists=True))
@click.argument("outfile", type=click.Path())
def run(specfile, outfile):
    """Run a spec JSON and save report."""
    with open(specfile, encoding="utf-8") as fh:
        spec = json.load(fh)
    # Use Ollama as default driver since it can run locally
    drivers = {"ollama": OllamaDriver(endpoint="http://localhost:11434", model="gemma:latest")}
    report = run_suite_from_spec(spec, drivers)
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    click.echo(f"Report saved to {outfile}")


@cli.command()
@click.option("--model", default="openai/gpt-4o-mini", help="Model string (provider/model).")
@click.option("--system-prompt", default=None, help="System prompt for conversations.")
@click.option("--host", default="0.0.0.0", help="Bind host.")
@click.option("--port", default=8000, type=int, help="Bind port.")
@click.option("--cors-origins", default=None, help="Comma-separated CORS origins (use * for all).")
def serve(model, system_prompt, host, port, cors_origins):
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
def scaffold(output_dir, name, model, docker):
    """Generate a standalone FastAPI project using Prompture.

    Requires the 'scaffold' extra: pip install prompture[scaffold]
    """
    try:
        from .scaffold.generator import scaffold_project
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
