import json

import click

from .drivers import OllamaDriver
from .runner import run_suite_from_spec


@click.group()
def cli():
    """Simple CLI to run JSON specs"""
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
