"""Helpers for discovering and running local coding-agent CLIs."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from .coding_agent_events import CodingAgentEvent
from .coding_agent_specs import CODING_AGENT_SPECS, get_spec
from .discovery import (
    CodingAgentExecutable,
    resolve_coding_agent_executable,
)

CodingAgentId = str
ApprovalMode = Literal["default", "auto", "yolo"]
OutputFormat = Literal["text", "json"]

_SUPPORTED_APPROVAL_MODES = {"default", "auto", "yolo"}
_SUPPORTED_OUTPUT_FORMATS = {"text", "json"}


@dataclasses.dataclass(frozen=True)
class CodingAgentCommand:
    """Resolved command for a local coding-agent CLI."""

    agent: str
    argv: list[str]
    binary: str
    cwd: str
    approval_mode: ApprovalMode


@dataclasses.dataclass(frozen=True)
class CodingAgentRunResult:
    """Result from running a local coding-agent CLI."""

    agent: str
    command: list[str]
    cwd: str
    returncode: int
    output: str
    timed_out: bool = False
    duration_seconds: float = 0.0
    events: tuple[CodingAgentEvent, ...] = ()
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def ok(self) -> bool:
        """Whether the command completed successfully."""
        return self.returncode == 0 and not self.timed_out


def _normalize_agent(agent: str) -> str:
    normalized = agent.strip().lower()
    if normalized not in CODING_AGENT_SPECS:
        available = ", ".join(sorted(CODING_AGENT_SPECS))
        raise ValueError(f"Unsupported coding agent '{agent}'. Available: {available}")
    return normalized


def _normalize_approval_mode(approval_mode: str) -> ApprovalMode:
    normalized = approval_mode.strip().lower()
    if normalized not in _SUPPORTED_APPROVAL_MODES:
        available = ", ".join(sorted(_SUPPORTED_APPROVAL_MODES))
        raise ValueError(f"Unsupported approval_mode '{approval_mode}'. Available: {available}")
    return normalized  # type: ignore[return-value]


def _normalize_output_format(agent_id: str, output_format: str) -> OutputFormat:
    normalized = output_format.strip().lower()
    if normalized not in _SUPPORTED_OUTPUT_FORMATS:
        available = ", ".join(sorted(_SUPPORTED_OUTPUT_FORMATS))
        raise ValueError(f"Unsupported output_format '{output_format}'. Available: {available}")
    if normalized == "json":
        spec = get_spec(agent_id)
        if spec is None or not spec.supports_structured_output:
            raise ValueError(
                f"Agent '{agent_id}' does not support output_format='json' yet — "
                f"only agents whose spec sets parse_events do."
            )
    return normalized  # type: ignore[return-value]


def _resolve_cwd(cwd: str | os.PathLike[str] | None) -> str:
    path = Path(cwd or os.getcwd()).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Working directory does not exist: {path}")
    return str(path)


def build_coding_agent_command(
    agent: str,
    task: str,
    *,
    cwd: str | os.PathLike[str] | None = None,
    approval_mode: ApprovalMode = "default",
    binary: str | None = None,
    agent_paths: Mapping[str, str] | None = None,
    model: str | None = None,
    extra_args: Sequence[str] | None = None,
    output_format: OutputFormat = "text",
) -> CodingAgentCommand:
    """Build the subprocess command for a local coding-agent CLI.

    Args:
        agent: ``"claude"``, ``"codex"``, or ``"gemini"``.
        task: Prompt/task to hand to the coding agent.
        cwd: Working directory for the coding agent. Defaults to the current
            process directory.
        approval_mode: ``"default"`` uses the CLI defaults, ``"auto"`` avoids
            repeated approval prompts while preserving a workspace-oriented
            mode where the CLI supports it, and ``"yolo"`` enables each CLI's
            dangerous bypass flag.
        binary: Optional explicit binary path for this call.
        agent_paths: Optional per-agent binary path overrides.
        model: Optional model flag passed to CLIs that support model selection.
        extra_args: Extra CLI flags inserted before the task.
    """
    agent_id = _normalize_agent(agent)
    mode = _normalize_approval_mode(approval_mode)
    fmt = _normalize_output_format(agent_id, output_format)
    task_text = task.strip()
    if not task_text:
        raise ValueError("task must not be empty")

    executable = _resolve_executable(agent_id, binary=binary, agent_paths=agent_paths)
    if not executable:
        path_overrides = agent_paths or {}
        binary_name = binary or path_overrides.get(agent_id) or agent_id
        raise RuntimeError(f"'{binary_name}' is not installed or not on PATH")

    cwd_str = _resolve_cwd(cwd)
    args_extra = list(extra_args or [])

    return _build_command_from_executable(
        agent_id,
        task_text,
        executable=executable,
        cwd_str=cwd_str,
        approval_mode=mode,
        model=model,
        extra_args=args_extra,
        output_format=fmt,
    )


def _resolve_executable(
    agent_id: str,
    *,
    binary: str | None = None,
    agent_paths: Mapping[str, str] | None = None,
) -> CodingAgentExecutable | None:
    path_overrides = agent_paths or {}
    binary_name = binary or path_overrides.get(agent_id) or agent_id
    executable, _healthy, _error = resolve_coding_agent_executable(agent_id, binary_name)
    return executable


def _resolve_healthy_executable(
    agent_id: str,
    *,
    binary: str | None = None,
    agent_paths: Mapping[str, str] | None = None,
    verify_timeout: int = 5,
) -> tuple[CodingAgentExecutable | None, str | None]:
    path_overrides = agent_paths or {}
    binary_name = binary or path_overrides.get(agent_id) or agent_id
    executable, healthy, error = resolve_coding_agent_executable(
        agent_id,
        binary_name,
        verify=True,
        verify_timeout=verify_timeout,
    )
    if healthy:
        return executable, None
    return None, error


def _build_command_from_executable(
    agent_id: str,
    task_text: str,
    *,
    executable: CodingAgentExecutable,
    cwd_str: str,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
    output_format: OutputFormat = "text",
) -> CodingAgentCommand:
    spec = get_spec(agent_id)
    if spec is None:
        raise ValueError(f"Unsupported coding agent '{agent_id}'")
    args = spec.build_args(
        task_text,
        approval_mode=approval_mode,
        model=model,
        extra_args=extra_args,
        output_format=output_format,
    )

    return CodingAgentCommand(
        agent=agent_id,
        argv=[*executable.argv, *args],
        binary=executable.binary,
        cwd=cwd_str,
        approval_mode=approval_mode,
    )


def _merge_output(stdout: str | None, stderr: str | None) -> str:
    chunks = [part for part in (stdout, stderr) if part]
    return "\n".join(chunks).strip()


def run_coding_agent(
    agent: str,
    task: str,
    *,
    cwd: str | os.PathLike[str] | None = None,
    approval_mode: ApprovalMode = "default",
    binary: str | None = None,
    agent_paths: Mapping[str, str] | None = None,
    model: str | None = None,
    extra_args: Sequence[str] | None = None,
    timeout: int = 600,
    max_output_chars: int = 50000,
    env: Mapping[str, str] | None = None,
    verify_binary: bool = True,
    verify_timeout: int = 5,
    output_format: OutputFormat = "text",
) -> CodingAgentRunResult:
    """Run a local coding-agent CLI and return its output.

    This is intentionally a thin subprocess wrapper. It does not sandbox the
    agent itself; use ``approval_mode="yolo"`` only inside an environment you
    already trust to contain the blast radius. By default, the resolved binary
    is health-checked before the agent task starts so broken PATH shims fail
    early with a clear message.

    When ``output_format="json"`` is supported by the agent (Claude Code today)
    the result's ``events`` field is populated with parsed
    :class:`CodingAgentEvent` instances and ``cost_usd`` / ``input_tokens`` /
    ``output_tokens`` are extracted from the final result event.
    """
    agent_id = _normalize_agent(agent)
    mode = _normalize_approval_mode(approval_mode)
    fmt = _normalize_output_format(agent_id, output_format)
    task_text = task.strip()
    if not task_text:
        raise ValueError("task must not be empty")
    cwd_str = _resolve_cwd(cwd)
    args_extra = list(extra_args or [])
    start = time.monotonic()

    if verify_binary:
        executable, error = _resolve_healthy_executable(
            agent_id,
            binary=binary,
            agent_paths=agent_paths,
            verify_timeout=verify_timeout,
        )
        if executable is None:
            return CodingAgentRunResult(
                agent=agent_id,
                command=[],
                cwd=cwd_str,
                returncode=-1,
                output=f"{agent_id} CLI health check failed: {error}",
                duration_seconds=time.monotonic() - start,
            )
        command = _build_command_from_executable(
            agent_id,
            task_text,
            executable=executable,
            cwd_str=cwd_str,
            approval_mode=mode,
            model=model,
            extra_args=args_extra,
            output_format=fmt,
        )
    else:
        command = build_coding_agent_command(
            agent_id,
            task_text,
            cwd=cwd_str,
            approval_mode=mode,
            binary=binary,
            agent_paths=agent_paths,
            model=model,
            extra_args=args_extra,
            output_format=fmt,
        )

    child_env = None if env is None else {**os.environ, **dict(env)}
    try:
        completed = subprocess.run(
            command.argv,
            cwd=command.cwd,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        timed_out = False
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        if not stdout and not stderr:
            stderr = f"{command.binary} timed out after {timeout}s"
        timed_out = True
        returncode = -1

    events: tuple[CodingAgentEvent, ...] = ()
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    if fmt == "json":
        spec = get_spec(agent_id)
        parser = spec.parse_events if spec else None
        if parser is not None:
            events = tuple(parser(stdout.splitlines()))
            for ev in events:
                if ev.type == "done":
                    cost_usd = ev.cost_usd
                    input_tokens = ev.input_tokens
                    output_tokens = ev.output_tokens
                    break

    output = _merge_output(stdout, stderr)
    if len(output) > max_output_chars:
        output = output[:max_output_chars] + f"\n\n... (truncated at {max_output_chars} chars)"

    return CodingAgentRunResult(
        agent=command.agent,
        command=command.argv,
        cwd=command.cwd,
        returncode=returncode,
        output=output,
        timed_out=timed_out,
        duration_seconds=time.monotonic() - start,
        events=events,
        cost_usd=cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


async def arun_coding_agent(*args: object, **kwargs: object) -> CodingAgentRunResult:
    """Async wrapper around :func:`run_coding_agent`."""
    return await asyncio.to_thread(run_coding_agent, *args, **kwargs)
