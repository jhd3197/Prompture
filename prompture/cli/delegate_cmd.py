"""Delegate a task to a coding agent through the companion's automation queue.

The companion (``prompture companion``) runs queued coding-agent steps
unattended: Claude Code or Codex does the task with permission prompts
skipped, model access coming from this machine's Prompture setup — the
delegating caller never needs a provider key. These commands are the
scriptable front end to that queue, so another agent (or a person) can
hand off a task and read the result from stdout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import click

#: Run states that mean the queue is over.
ENDED = ("finished", "failed", "stopped")
#: Exit code when the run paused on a question and stdin can't supply an answer.
EXIT_ASK = 3
#: Exit code when the run carries on without us (stopped, cost cap, timeout, Ctrl+C).
EXIT_DETACHED = 4
#: Seconds between polls while a run is followed.
POLL_SECONDS = 1.5


class _Companion:
    """Tiny stdlib client for the companion's loopback API."""

    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        raw = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            self.url + path,
            data=raw,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                **({"Content-Type": "application/json"} if raw else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read())
            except ValueError:
                return exc.code, {"detail": str(exc.reason)}
        except OSError as exc:
            raise click.ClickException(f"Can't reach the companion at {self.url}: {exc}") from None

    def get(self, path: str) -> tuple[int, Any]:
        return self.call("GET", path)

    def post(self, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        return self.call("POST", path, body if body is not None else {})


def _start_companion(timeout: float = 30.0) -> dict[str, Any] | None:
    """Launch ``prompture companion`` detached and wait for its state file to answer."""
    from ..companion import running_instance

    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen([sys.executable, "-m", "prompture", "companion"], **kwargs)
    except OSError as exc:
        raise click.ClickException(f"Couldn't start the companion: {exc}") from None
    click.echo("Started a companion for this run…", err=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = running_instance(timeout=1.0)
        if state:
            return state
        time.sleep(0.5)
    return None


def _ensure_companion() -> _Companion:
    """A client for the running companion, starting one first if needed."""
    from ..companion import running_instance

    state = running_instance() or _start_companion()
    if state is None:
        raise click.ClickException(
            "No companion is running and starting one failed. Run `prompture companion` yourself to see why."
        )
    return _Companion(str(state["url"]), str(state["token"]))


def _detail(payload: Any) -> str:
    return str(payload.get("detail") or payload) if isinstance(payload, dict) else str(payload)


def _stream_logs(client: _Companion, run: dict[str, Any], seen: dict[str, int], announced: set[str]) -> None:
    """Print each step's new log lines since the last poll."""
    steps = run.get("steps") or []
    for index, step in enumerate(steps):
        if step.get("status") == "waiting":
            continue
        if step["id"] not in announced:
            announced.add(step["id"])
            click.echo(f"\n── step {index + 1}/{len(steps)}: {step.get('text', '')}")
        status, body = client.get(f"/v1/automations/runs/{run['id']}/steps/{step['id']}/log")
        if status != 200:
            continue
        lines = body.get("lines", []) if isinstance(body, dict) else []
        offset = seen.get(step["id"], 0)
        if offset > len(lines):  # the server only keeps the last LOG_MAX lines
            offset = 0
        for line in lines[offset:]:
            click.echo(line.get("text", ""), err=line.get("kind") == "err")
        seen[step["id"]] = len(lines)


def _follow(client: _Companion, run_id: str, *, as_json: bool = False, timeout: float = 0.0) -> int:
    """Stream a run's log until it ends; returns the process exit code."""
    seen: dict[str, int] = {}
    announced: set[str] = set()
    last_note: tuple[Any, Any] | None = None
    deadline = time.monotonic() + timeout if timeout else None
    try:
        while True:
            status, run = client.get(f"/v1/automations/runs/{run_id}")
            if status != 200:
                raise click.ClickException(f"Lost the run: {_detail(run)}")
            if not as_json:
                _stream_logs(client, run, seen, announced)
            state = run.get("status")
            if state in ENDED:
                if as_json:
                    click.echo(json.dumps(run, indent=2))
                else:
                    click.echo(f"\n{state} — ${run.get('cost_usd') or 0:.4f}, {run.get('duration_s') or 0}s")
                return 0 if state == "finished" else (1 if state == "failed" else EXIT_DETACHED)
            if state == "paused":
                reason = run.get("reason")
                if reason == "ask":
                    question = run.get("question") or "The agent is asking a question."
                    click.echo(f"\nThe agent asks: {question}")
                    if not sys.stdin.isatty():
                        click.echo('Answer with: prompture delegate-answer "your answer"')
                        return EXIT_ASK
                    answer = click.prompt("Answer", default="", show_default=False)
                    if not answer.strip():
                        click.echo("No answer — the queue stays paused.")
                        return EXIT_ASK
                    status, payload = client.post("/v1/automations/current/answer", {"text": answer})
                    if status != 200:
                        raise click.ClickException(_detail(payload))
                    continue
                note_key = (state, reason)
                if note_key != last_note and not as_json:
                    last_note = note_key
                    if reason == "limit":
                        when = f" at {time.ctime(run['resumes_at'])}" if run.get("resumes_at") else ""
                        click.echo(f"\nPlan window nearly used up — the queue resumes by itself{when}.")
                    elif reason == "fail":
                        step = (run.get("steps") or [{}])[run.get("current", 0)]
                        click.echo(f"\nStep failed: {step.get('error') or 'unknown error'}")
                        click.echo("Fix and carry on with: prompture delegate-resume")
                        return 1
                    elif reason == "cost":
                        click.echo(f"\nCost cap reached (${run.get('cost_usd') or 0:.4f}).")
                        click.echo("Carry on without a cap: prompture delegate-resume")
                        return EXIT_DETACHED
                    elif reason == "manual":
                        click.echo("\nPaused from a companion app — waiting for it to resume.")
            else:
                last_note = None
            if deadline is not None and time.monotonic() > deadline:
                click.echo("\nStill running — leaving it in the background. Check with: prompture delegate-status")
                return EXIT_DETACHED
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        click.echo("\nDetached — the run continues. Check with: prompture delegate-status")
        return 130


def _wait_options(command: Any) -> Any:
    command = click.option(
        "--wait/--no-wait", default=True, help="Stream until the run ends (default) or return at once."
    )(command)
    command = click.option(
        "--timeout", default=0.0, type=float, help="Stop watching after N seconds (the run continues; 0 = no limit)."
    )(command)
    return click.option(
        "--json", "as_json", is_flag=True, help="Print the final run as JSON instead of streaming the log."
    )(command)


@click.command()
@click.argument("task", nargs=-1, required=True)
@click.option(
    "--cwd", default=".", type=click.Path(exists=True, file_okay=False), help="Project folder (default: this one)."
)
@click.option(
    "--agent",
    type=click.Choice(["auto", "claude", "codex"]),
    default="auto",
    help="Coding agent for the task (default: auto — whichever has the most plan left).",
)
@click.option("--model", default=None, help="Model string (provider/model); default is the agent's own.")
@click.option("--cost-cap", default=None, type=float, help="Pause the run once it passes this many USD.")
@_wait_options
def delegate(
    task: tuple[str, ...],
    cwd: str,
    agent: str,
    model: str | None,
    cost_cap: float | None,
    wait: bool,
    timeout: float,
    as_json: bool,
) -> None:
    """Hand a task to a coding agent and watch it finish.

    The companion runs TASK unattended — permission prompts skipped, model
    access from this machine's Prompture setup, so no provider key is needed
    here. The agent's work streams to stdout. If it ends on a question, answer
    with `prompture delegate-answer "…"`.

    Exit codes: 0 finished, 1 failed, 3 waiting for an answer, 4 stopped or
    still running in the background.
    """
    text = " ".join(task).strip()
    if not text:
        raise click.ClickException("Say what to do.")
    client = _ensure_companion()
    body: dict[str, Any] = {
        "cwd": str(Path(cwd).resolve()),
        "agent": agent,
        "steps": [{"text": text, "session": "new"}],
    }
    if model:
        body["model"] = model
    if cost_cap:
        body["stop"] = {"cost_usd": cost_cap}
    status, run = client.post("/v1/automations", body)
    if status == 404:
        raise click.ClickException(
            "This companion was started with automations off. Restart it with `prompture companion` (they're on by default)."
        )
    if status == 409:
        raise click.ClickException("The companion is already running a queue — see `prompture delegate-status`.")
    if status != 200:
        raise click.ClickException(_detail(run))
    if not wait and not as_json:
        click.echo(
            f"Running as {run.get('agent_name', agent)} (run {run['id']}). Watch with: prompture delegate-status"
        )
        return
    if not as_json:
        click.echo(f"Delegated to {run.get('agent_name', agent)} (run {run['id']}).")
    code = _follow(client, run["id"], as_json=as_json, timeout=timeout)
    if code:
        raise SystemExit(code)


@click.command("delegate-answer")
@click.argument("text", nargs=-1, required=True)
@_wait_options
def delegate_answer(text: tuple[str, ...], wait: bool, timeout: float, as_json: bool) -> None:
    """Answer a delegated run that is waiting on a question, then keep watching."""
    client = _ensure_companion()
    status, run = client.post("/v1/automations/current/answer", {"text": " ".join(text).strip()})
    if status != 200:
        raise click.ClickException(_detail(run))
    if not wait and not as_json:
        click.echo("Answer sent — the step's session continues.")
        return
    code = _follow(client, run["id"], as_json=as_json, timeout=timeout)
    if code:
        raise SystemExit(code)


@click.command("delegate-resume")
@_wait_options
def delegate_resume(wait: bool, timeout: float, as_json: bool) -> None:
    """Carry on a paused run (after a failure, a cost cap, or a plan limit)."""
    client = _ensure_companion()
    status, run = client.post("/v1/automations/current/resume")
    if status != 200:
        raise click.ClickException(_detail(run))
    if not wait and not as_json:
        click.echo("Resumed.")
        return
    code = _follow(client, run["id"], as_json=as_json, timeout=timeout)
    if code:
        raise SystemExit(code)


@click.command("delegate-status")
@click.option("--json", "as_json", is_flag=True, help="Print the queue state as JSON.")
def delegate_status(as_json: bool) -> None:
    """Show the queue the companion is running now."""
    client = _ensure_companion()
    status, state = client.get("/v1/automations")
    if status == 404:
        raise click.ClickException(
            "This companion was started with automations off. Restart it with `prompture companion` (they're on by default)."
        )
    if status != 200:
        raise click.ClickException(_detail(state))
    if as_json:
        click.echo(json.dumps(state, indent=2))
        return
    run = state.get("current")
    if not run:
        click.echo("No queue running.")
        history = state.get("history") or []
        if history:
            last = history[0]
            click.echo(f"Last run {last.get('id')}: {last.get('status')} ({last.get('project', '')})")
        return
    marks = {"done": "✓", "running": "▶", "failed": "✗", "asked": "?", "skipped": "–", "stopped": "■"}
    line = f"{run.get('agent_name', run.get('agent'))} — {run.get('status')}"
    if run.get("reason"):
        line += f" ({run['reason']})"
    click.echo(line)
    if run.get("question"):
        click.echo(f"Question: {run['question']}")
        click.echo('Answer with: prompture delegate-answer "your answer"')
    for index, step in enumerate(run.get("steps") or []):
        mark = marks.get(step.get("status"), " ")
        click.echo(f"  {mark} {index + 1}. {step.get('text', '')}")
    click.echo(f"Cost so far: ${run.get('cost_usd') or 0:.4f}")


COMMANDS = (delegate, delegate_answer, delegate_resume, delegate_status)
