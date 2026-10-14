"""A coding-agent CLI you are signed in to (Claude Code, Codex) used as a plain model.

``claude-code/<model>`` and ``codex/<model>`` work anywhere a model string does
(``extract_with_model``, ``Conversation``, the game harnesses). Each call runs the CLI once in
print mode, in an empty temporary folder, with its default approvals, so it answers from your
subscription login without an API key and without touching any project. ``<model>`` is passed
to the CLI's ``--model`` (``opus``, ``sonnet``, ``gpt-6-sol`` ...); ``default`` leaves the
CLI's own choice.

Calls are slower than an API (a process start each time) and count against the
subscription's limits, not a bill, so ``meta["cost"]`` is 0 and the CLI's own estimate, when
it gives one, is in ``meta["reported_cost"]``.
"""

from __future__ import annotations

import tempfile
from typing import Any

from ..infra.coding_agents import CodingAgentRunResult, run_coding_agent
from .base import Driver

DEFAULT_MODEL = "default"
DEFAULT_TIMEOUT_S = 600


class CodingAgentError(RuntimeError):
    """The CLI did not answer: not installed, not signed in, timed out, or failed."""


class CodingAgentDriver(Driver):
    """Base for one coding-agent CLI; subclasses set ``agent`` and ``provider``."""

    agent = ""
    provider = ""
    supports_messages = True
    supports_json_mode = False
    MODELS: tuple[str, ...] = (DEFAULT_MODEL,)
    MODEL_PRICING = {"default": {"prompt": 0.0, "completion": 0.0}}

    def __init__(self, model: str = DEFAULT_MODEL, **_: Any) -> None:
        self.model = model or DEFAULT_MODEL
        self.options: dict[str, Any] = {}

    @classmethod
    def list_models(cls, **kwargs: Any) -> list[str] | None:
        """The model names this CLI takes (it has no listing of its own)."""
        return list(cls.MODELS)

    @classmethod
    def available(cls, **_: Any) -> bool:
        """Whether the CLI is installed (on PATH, or set with CODING_AGENT_BIN_<AGENT>)."""
        from ..infra.coding_agent_specs import get_spec
        from ..infra.discovery import resolve_coding_agent_executable

        spec = get_spec(cls.agent)
        if spec is None:
            return False
        executable, _healthy, _error = resolve_coding_agent_executable(cls.agent, spec.default_binary)
        return executable is not None

    # -- the agent's flags -------------------------------------------------

    def _extra_args(self, system: str | None, options: dict[str, Any]) -> list[str]:
        return []

    def _task(self, system: str | None, prompt: str) -> str:
        return f"{system}\n\n{prompt}" if system else prompt

    # -- Driver ------------------------------------------------------------

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        return self._run(None, prompt, options or {})

    def generate_messages(self, messages: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        system = "\n\n".join(str(m.get("content") or "") for m in messages if m.get("role") == "system").strip()
        rest = [m for m in messages if m.get("role") != "system"]
        return self._run(system or None, self._flatten_messages(rest), options or {})

    def _run(self, system: str | None, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        model = options.get("model", self.model)
        with tempfile.TemporaryDirectory(prefix="prompture-agent-") as folder:
            result = run_coding_agent(
                self.agent,
                self._task(system, prompt),
                cwd=folder,
                model=None if model in (None, "", DEFAULT_MODEL) else model,
                extra_args=self._extra_args(system, options),
                timeout=int(options.get("timeout") or DEFAULT_TIMEOUT_S),
                output_format="json",
            )
        text = final_text(result)
        if not result.ok or text is None:
            raise CodingAgentError(_failure(result))
        prompt_tokens, completion_tokens = result.input_tokens or 0, result.output_tokens or 0
        meta = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": 0.0,
            "reported_cost": result.cost_usd,
            "model_name": f"{self.provider}/{model}",
            "raw_response": {"agent": result.agent, "duration_seconds": result.duration_seconds},
        }
        return {"text": text, "meta": meta}


def final_text(result: CodingAgentRunResult) -> str | None:
    """The agent's answer: the final result, else its last message, from the JSON events."""
    done = next((e.text for e in reversed(result.events) if e.type == "done" and e.text), None)
    if done:
        return done
    return next((e.text for e in reversed(result.events) if e.type == "message" and e.text), None)


def _failure(result: CodingAgentRunResult) -> str:
    if result.timed_out:
        return f"{result.agent} did not answer in time"
    error = next(
        (e.error or e.text for e in reversed(result.events) if e.type == "error" and (e.error or e.text)), None
    )
    lines = (result.output or "").strip().splitlines()
    detail = error or (lines[-1] if lines else "no answer")
    return f"{result.agent} failed: {detail}"[:500]


class ClaudeCodeDriver(CodingAgentDriver):
    """``claude-code/<model>``: Claude Code in print mode, with your Claude login."""

    agent = "claude"
    provider = "claude-code"
    MODELS = (DEFAULT_MODEL, "opus", "sonnet", "haiku")

    def _extra_args(self, system: str | None, options: dict[str, Any]) -> list[str]:
        # One turn and no coding persona: a model call, not an agent session.
        args = ["--max-turns", "1"]
        if system:
            args += ["--system-prompt", system]
        return args

    def _task(self, system: str | None, prompt: str) -> str:
        return prompt  # the system prompt goes in --system-prompt


class CodexDriver(CodingAgentDriver):
    """``codex/<model>``: Codex CLI's ``exec``, with your ChatGPT login."""

    agent = "codex"
    provider = "codex"

    def _extra_args(self, system: str | None, options: dict[str, Any]) -> list[str]:
        from ._effort import requested_effort

        # The empty temporary folder is not a git repository, and the answer needs no writes.
        args = ["--skip-git-repo-check", "--sandbox", "read-only"]
        effort = requested_effort(options)
        return args + (["-c", f"model_reasoning_effort={effort}"] if effort else [])
