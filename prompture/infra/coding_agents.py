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

from .discovery import resolve_coding_agent_binary

CodingAgentId = Literal["claude", "codex", "gemini"]
ApprovalMode = Literal["default", "auto", "yolo"]

_SUPPORTED_AGENTS = {"claude", "codex", "gemini"}
_SUPPORTED_APPROVAL_MODES = {"default", "auto", "yolo"}


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

    @property
    def ok(self) -> bool:
        """Whether the command completed successfully."""
        return self.returncode == 0 and not self.timed_out


def _normalize_agent(agent: str) -> str:
    normalized = agent.strip().lower()
    if normalized not in _SUPPORTED_AGENTS:
        available = ", ".join(sorted(_SUPPORTED_AGENTS))
        raise ValueError(f"Unsupported coding agent '{agent}'. Available: {available}")
    return normalized


def _normalize_approval_mode(approval_mode: str) -> ApprovalMode:
    normalized = approval_mode.strip().lower()
    if normalized not in _SUPPORTED_APPROVAL_MODES:
        available = ", ".join(sorted(_SUPPORTED_APPROVAL_MODES))
        raise ValueError(f"Unsupported approval_mode '{approval_mode}'. Available: {available}")
    return normalized  # type: ignore[return-value]


def _resolve_cwd(cwd: str | os.PathLike[str] | None) -> str:
    path = Path(cwd or os.getcwd()).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Working directory does not exist: {path}")
    return str(path)


def _build_claude_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
) -> list[str]:
    args = ["--print", "--output-format", "text"]
    if model:
        args.extend(["--model", model])
    if approval_mode == "auto":
        args.extend(["--permission-mode", "dontAsk"])
    elif approval_mode == "yolo":
        args.append("--dangerously-skip-permissions")
    args.extend(extra_args)
    args.append(task)
    return args


def _build_codex_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
) -> list[str]:
    args = ["exec"]
    if model:
        args.extend(["--model", model])
    if approval_mode == "auto":
        args.extend(["--sandbox", "workspace-write", "--ask-for-approval", "never"])
    elif approval_mode == "yolo":
        args.append("--dangerously-bypass-approvals-and-sandbox")
    args.extend(extra_args)
    args.append(task)
    return args


def _build_gemini_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
) -> list[str]:
    args: list[str] = []
    if model:
        args.extend(["--model", model])
    if approval_mode in {"auto", "yolo"}:
        args.append("-y")
    args.extend(extra_args)
    args.append(task)
    return args


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
    task_text = task.strip()
    if not task_text:
        raise ValueError("task must not be empty")

    path_overrides = agent_paths or {}
    binary_name = binary or path_overrides.get(agent_id) or agent_id
    resolved_binary = resolve_coding_agent_binary(binary_name)
    if not resolved_binary:
        raise RuntimeError(f"'{binary_name}' is not installed or not on PATH")

    cwd_str = _resolve_cwd(cwd)
    args_extra = list(extra_args or [])

    if agent_id == "claude":
        args = _build_claude_args(task_text, approval_mode=mode, model=model, extra_args=args_extra)
    elif agent_id == "codex":
        args = _build_codex_args(task_text, approval_mode=mode, model=model, extra_args=args_extra)
    else:
        args = _build_gemini_args(task_text, approval_mode=mode, model=model, extra_args=args_extra)

    return CodingAgentCommand(
        agent=agent_id,
        argv=[resolved_binary, *args],
        binary=resolved_binary,
        cwd=cwd_str,
        approval_mode=mode,
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
) -> CodingAgentRunResult:
    """Run a local coding-agent CLI and return its output.

    This is intentionally a thin subprocess wrapper. It does not sandbox the
    agent itself; use ``approval_mode="yolo"`` only inside an environment you
    already trust to contain the blast radius.
    """
    command = build_coding_agent_command(
        agent,
        task,
        cwd=cwd,
        approval_mode=approval_mode,
        binary=binary,
        agent_paths=agent_paths,
        model=model,
        extra_args=extra_args,
    )

    child_env = None if env is None else {**os.environ, **dict(env)}
    start = time.monotonic()
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
        output = _merge_output(completed.stdout, completed.stderr)
        timed_out = False
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        output = _merge_output(stdout, stderr) or f"{command.binary} timed out after {timeout}s"
        timed_out = True
        returncode = -1

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
    )


async def arun_coding_agent(*args: object, **kwargs: object) -> CodingAgentRunResult:
    """Async wrapper around :func:`run_coding_agent`."""
    return await asyncio.to_thread(run_coding_agent, *args, **kwargs)
