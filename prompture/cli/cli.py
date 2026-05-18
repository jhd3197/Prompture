import json
import shlex

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


@cli.command("coding-agents")
@click.option("--json", "json_output", is_flag=True, help="Output machine-readable JSON.")
@click.option("--verify", is_flag=True, help="Run a lightweight health check for each detected CLI.")
def coding_agents(json_output: bool, verify: bool) -> None:
    """List detected local coding-agent CLIs."""
    from dataclasses import asdict

    from ..infra import get_available_coding_agents

    agents = get_available_coding_agents(verify=verify)
    if json_output:
        click.echo(json.dumps([asdict(agent) for agent in agents], indent=2))
        return

    for agent in agents:
        if verify:
            status = "healthy" if agent.available else "failed"
        else:
            status = "available" if agent.available else "missing"
        source = "custom" if agent.custom_path else (agent.source or "missing")
        detail = f"  {agent.error}" if agent.error else ""
        click.echo(f"{agent.id:<7} {status:<9} {source:<12} {agent.binary}{detail}")


@cli.command("code-agent")
@click.argument("agent", type=click.Choice(["claude", "codex", "gemini"]))
@click.argument("task", nargs=-1, required=True)
@click.option("--cwd", default=".", type=click.Path(exists=True, file_okay=False), help="Working directory.")
@click.option("--model", default=None, help="Model passed through to the agent CLI.")
@click.option("--binary", default=None, help="Explicit binary path for this run.")
@click.option("--auto-approve", is_flag=True, help="Avoid repeated approval prompts where the CLI supports it.")
@click.option("--yolo", is_flag=True, help="Use the CLI's dangerous no-approval/no-sandbox mode.")
@click.option("--timeout", default=600, type=int, help="Timeout in seconds.")
@click.option("--max-output", default=50000, type=int, help="Maximum output characters to print.")
@click.option("--verify/--no-verify", default=True, help="Health-check the CLI before running.")
@click.option("--dry-run", is_flag=True, help="Print the resolved command without running it.")
def code_agent(
    agent: str,
    task: tuple[str, ...],
    cwd: str,
    model: str | None,
    binary: str | None,
    auto_approve: bool,
    yolo: bool,
    timeout: int,
    max_output: int,
    verify: bool,
    dry_run: bool,
) -> None:
    """Run Claude Code, Codex CLI, or Gemini CLI from Prompture."""
    from ..infra import build_coding_agent_command, run_coding_agent

    if auto_approve and yolo:
        click.echo("Error: choose either --auto-approve or --yolo, not both.", err=True)
        raise SystemExit(2)

    approval_mode = "yolo" if yolo else ("auto" if auto_approve else "default")
    task_text = " ".join(task).strip()

    if dry_run:
        command = build_coding_agent_command(
            agent,
            task_text,
            cwd=cwd,
            approval_mode=approval_mode,
            binary=binary,
            model=model,
        )
        click.echo(shlex.join(command.argv))
        return

    result = run_coding_agent(
        agent,
        task_text,
        cwd=cwd,
        approval_mode=approval_mode,
        binary=binary,
        model=model,
        timeout=timeout,
        max_output_chars=max_output,
        verify_binary=verify,
    )

    if result.output:
        click.echo(result.output)
    if not result.ok:
        raise SystemExit(result.returncode if result.returncode > 0 else 1)


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
@click.option("--model", default="openai/gpt-4o-mini", help="Default model string (provider/model).")
@click.option("--system-prompt", default=None, help="System prompt injected when the client doesn't supply one.")
@click.option("--host", default="0.0.0.0", help="Bind host.")
@click.option("--port", default=9471, type=int, help="Bind port.")
@click.option("--cors-origins", default=None, help="Comma-separated CORS origins (use * for all).")
@click.option("--api-key", default=None, help="Require Bearer authentication with this key.")
@click.option(
    "--allow-models",
    default=None,
    help="Comma-separated allowlist of model strings. Other models return 403.",
)
@click.option(
    "--rate-limit",
    type=int,
    default=None,
    help="Per-IP request-per-minute cap. Defaults to no rate limit.",
)
@click.option(
    "--sandbox/--no-sandbox",
    default=False,
    help="Register the sandboxed python_execute tool server-side. Needs prompture[sandbox].",
)
@click.option(
    "--web-search/--no-web-search",
    default=False,
    help="Register the web_search tool server-side. Needs a search provider key in env.",
)
@click.option(
    "--max-conversations",
    type=int,
    default=1000,
    help="Maximum in-memory conversations before oldest are evicted.",
)
def serve(
    model: str,
    system_prompt: str,
    host: str,
    port: int,
    cors_origins: str,
    api_key: str,
    allow_models: str,
    rate_limit: int,
    sandbox: bool,
    web_search: bool,
    max_conversations: int,
) -> None:
    """Start the OpenAI-compatible API server.

    Once running, point any OpenAI SDK at ``http://<host>:<port>/v1`` and
    send requests for any Prompture-supported model.

    Pass ``--sandbox`` and/or ``--web-search`` to register those tools
    server-side — the LLM uses them transparently and clients only see
    the final assistant message.

    Requires the 'serve' extra: pip install prompture[serve]
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("Error: uvicorn not installed. Run: pip install prompture[serve]", err=True)
        raise SystemExit(1) from None

    from ..agents.tools_schema import ToolRegistry
    from .server import create_app

    origins = [o.strip() for o in cors_origins.split(",")] if cors_origins else None
    allowed = [m.strip() for m in allow_models.split(",")] if allow_models else None

    tool_registry: object | None = None
    if sandbox or web_search:
        tool_registry = ToolRegistry()
        if sandbox:
            try:
                from ..tools import PythonSandboxTool

                PythonSandboxTool().register_on(tool_registry)
                click.echo("Registered server-side tool: python_execute")
            except ImportError as exc:
                click.echo(
                    f"Error: --sandbox requires the sandbox extra. {exc}",
                    err=True,
                )
                raise SystemExit(1) from None
        if web_search:
            try:
                from ..tools import WebSearchTool

                ws = WebSearchTool()
                ws.register_on(tool_registry)
                click.echo(f"Registered server-side tool: web_search (provider={ws.provider})")
            except Exception as exc:
                click.echo(f"Error: --web-search failed to initialise. {exc}", err=True)
                raise SystemExit(1) from None

    app = create_app(
        model_name=model,
        system_prompt=system_prompt,
        tools=tool_registry,
        cors_origins=origins,
        api_key=api_key,
        allowed_models=allowed,
        rate_limit=rate_limit,
        max_conversations=max_conversations,
    )

    auth = "Bearer auth required" if api_key else "no auth"
    click.echo(f"Starting Prompture server on {host}:{port} model={model} ({auth})")
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
