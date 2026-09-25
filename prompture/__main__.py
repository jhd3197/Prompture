"""``python -m prompture …`` runs the ``prompture`` CLI (useful when its script isn't on PATH)."""

from .cli.cli import cli

if __name__ == "__main__":
    cli()
