import json
import click
from .runner import run_suite_from_spec

@click.group()
def cli():
    """CLI simple para correr specs JSON"""
    pass

@cli.command()
@click.argument("specfile", type=click.Path(exists=True))
@click.argument("outfile", type=click.Path())
def run(specfile, outfile):
    """Run a spec JSON and save report."""
    with open(specfile, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    # drivers mínimos: mock disponible por defecto
    from .drivers import MockDriver
    drivers = {"mock": MockDriver()}
    report = run_suite_from_spec(spec, drivers)
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    click.echo(f"Report saved to {outfile}")