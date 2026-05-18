"""Registry of supported local coding-agent CLIs.

Each :class:`CodingAgentSpec` bundles the metadata needed to discover, verify,
and invoke a coding-agent CLI. Adding a new agent is a single registration in
:data:`CODING_AGENT_SPECS` — discovery (PATH + npm package entrypoints + health
checks) and command construction read from this registry.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Literal

ApprovalMode = Literal["default", "auto", "yolo"]

BuildArgsFn = Callable[..., list[str]]


@dataclasses.dataclass(frozen=True)
class CodingAgentSpec:
    """Static description of a supported coding-agent CLI."""

    id: str
    display_name: str
    default_binary: str
    npm_packages: tuple[str, ...]
    build_args: BuildArgsFn


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


def _build_gemini_style_args(
    task: str,
    *,
    approval_mode: ApprovalMode,
    model: str | None,
    extra_args: Sequence[str],
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


CODING_AGENT_SPECS: dict[str, CodingAgentSpec] = {
    "claude": CodingAgentSpec(
        id="claude",
        display_name="Claude Code",
        default_binary="claude",
        npm_packages=("@anthropic-ai/claude-code",),
        build_args=_build_claude_args,
    ),
    "codex": CodingAgentSpec(
        id="codex",
        display_name="Codex CLI",
        default_binary="codex",
        npm_packages=("@openai/codex",),
        build_args=_build_codex_args,
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
}


def get_spec(agent_id: str) -> CodingAgentSpec | None:
    """Return the spec for ``agent_id`` or ``None`` if unknown."""
    return CODING_AGENT_SPECS.get(agent_id)


def supported_agent_ids() -> tuple[str, ...]:
    """Return the tuple of registered agent ids."""
    return tuple(CODING_AGENT_SPECS.keys())
