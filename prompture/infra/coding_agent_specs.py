"""Registry of supported local coding-agent CLIs.

Each :class:`CodingAgentSpec` bundles the metadata needed to discover, verify,
and invoke a coding-agent CLI. Adding a new agent is a single registration in
:data:`CODING_AGENT_SPECS` — discovery (PATH + npm package entrypoints + health
checks) and command construction read from this registry.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Literal

from .coding_agent_events import (
    CodingAgentEvent,
    parse_claude_stream_json_lines,
    parse_codex_json_lines,
)

ApprovalMode = Literal["default", "auto", "yolo"]
OutputFormat = Literal["text", "json"]

BuildArgsFn = Callable[..., list[str]]
ParseEventsFn = Callable[[Iterable[str]], Iterator[CodingAgentEvent]]


@dataclasses.dataclass(frozen=True)
class CodingAgentSpec:
    """Static description of a supported coding-agent CLI."""

    id: str
    display_name: str
    default_binary: str
    npm_packages: tuple[str, ...]
    build_args: BuildArgsFn
    parse_events: ParseEventsFn | None = None

    @property
    def supports_structured_output(self) -> bool:
        return self.parse_events is not None


def _build_claude_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
    output_format: OutputFormat = "text",
    session_id: str | None = None,
) -> list[str]:
    args = ["--print"]
    if output_format == "json":
        # Claude Code requires --verbose alongside stream-json.
        args.extend(["--output-format", "stream-json", "--verbose"])
    else:
        args.extend(["--output-format", "text"])
    if model:
        args.extend(["--model", model])
    if session_id:
        args.extend(["--resume", session_id])
    if approval_mode in {"auto", "yolo"}:
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
    output_format: OutputFormat = "text",
    session_id: str | None = None,
) -> list[str]:
    # Codex uses `exec` for a new task and `exec resume <id>` to continue one.
    args: list[str] = ["exec"]
    if session_id:
        args.extend(["resume", session_id])
    if output_format == "json":
        args.append("--json")
    if model:
        args.extend(["--model", model])
    if approval_mode == "auto":
        args.extend(["--sandbox", "workspace-write", "--ask-for-approval", "never"])
    elif approval_mode == "yolo":
        args.append("--dangerously-bypass-approvals-and-sandbox")
    args.extend(extra_args)
    args.append(task)
    return args


def _build_gemini_style_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
    output_format: OutputFormat = "text",
    session_id: str | None = None,
) -> list[str]:
    """Argument builder for gemini-cli and forks (e.g. Qwen Code)."""
    args: list[str] = []
    if model:
        args.extend(["--model", model])
    if approval_mode in {"auto", "yolo"}:
        args.append("-y")
    args.extend(extra_args)
    args.append(task)
    return args


def _build_aider_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
    output_format: OutputFormat = "text",
    session_id: str | None = None,
) -> list[str]:
    args: list[str] = []
    if model:
        args.extend(["--model", model])
    if approval_mode in {"auto", "yolo"}:
        args.append("--yes-always")
    args.extend(extra_args)
    args.extend(["--message", task])
    return args


def _build_opencode_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
    output_format: OutputFormat = "text",
    session_id: str | None = None,
) -> list[str]:
    args = ["run"]
    if model:
        args.extend(["--model", model])
    args.extend(extra_args)
    args.append(task)
    return args


def _build_cursor_agent_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
    output_format: OutputFormat = "text",
    session_id: str | None = None,
) -> list[str]:
    args = ["-p"]
    if model:
        args.extend(["--model", model])
    if approval_mode in {"auto", "yolo"}:
        args.append("--force")
    args.extend(extra_args)
    args.append(task)
    return args


def _build_crush_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
    output_format: OutputFormat = "text",
    session_id: str | None = None,
) -> list[str]:
    args = ["run"]
    if model:
        args.extend(["--model", model])
    if approval_mode in {"auto", "yolo"}:
        args.append("--yolo")
    args.extend(extra_args)
    args.append(task)
    return args


CODING_AGENT_SPECS: dict[str, CodingAgentSpec] = {
    "claude": CodingAgentSpec(
        id="claude",
        display_name="Claude Code",
        default_binary="claude",
        npm_packages=("@anthropic-ai/claude-code",),
        build_args=_build_claude_args,
        parse_events=parse_claude_stream_json_lines,
    ),
    "codex": CodingAgentSpec(
        id="codex",
        display_name="Codex CLI",
        default_binary="codex",
        npm_packages=("@openai/codex",),
        build_args=_build_codex_args,
        parse_events=parse_codex_json_lines,
    ),
    "gemini": CodingAgentSpec(
        id="gemini",
        display_name="Gemini CLI",
        default_binary="gemini",
        npm_packages=("@google/gemini-cli",),
        build_args=_build_gemini_style_args,
    ),
    "qwen": CodingAgentSpec(
        id="qwen",
        display_name="Qwen Code",
        default_binary="qwen",
        npm_packages=("@qwen-code/qwen-code",),
        build_args=_build_gemini_style_args,
    ),
    "aider": CodingAgentSpec(
        id="aider",
        display_name="Aider",
        default_binary="aider",
        npm_packages=(),
        build_args=_build_aider_args,
    ),
    "opencode": CodingAgentSpec(
        id="opencode",
        display_name="OpenCode",
        default_binary="opencode",
        npm_packages=("opencode-ai",),
        build_args=_build_opencode_args,
    ),
    "cursor-agent": CodingAgentSpec(
        id="cursor-agent",
        display_name="Cursor Agent",
        default_binary="cursor-agent",
        npm_packages=(),
        build_args=_build_cursor_agent_args,
    ),
    "crush": CodingAgentSpec(
        id="crush",
        display_name="Crush",
        default_binary="crush",
        npm_packages=(),
        build_args=_build_crush_args,
    ),
}


def get_spec(agent_id: str) -> CodingAgentSpec | None:
    """Return the spec for ``agent_id`` or ``None`` if unknown."""
    return CODING_AGENT_SPECS.get(agent_id)


def supported_agent_ids() -> tuple[str, ...]:
    """Return the tuple of registered agent ids."""
    return tuple(CODING_AGENT_SPECS.keys())
