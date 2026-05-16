"""Sandboxed Python execution tool for Prompture agents.

Wraps :class:`tukuy.sandbox.PythonSandbox` (restricted ``exec`` with
import whitelist, path restrictions, timeout, memory limit) plus
:func:`tukuy.analysis.analyze_python` (AST risk scoring) and exposes
the combination as a :class:`~prompture.ToolDefinition` named
``python_execute``.

Before execution, the code is statically analysed.  When the analysed
risk level meets or exceeds ``risk_threshold``, the tool raises
:class:`prompture.ApprovalRequired` — the agent's
``on_approval_needed`` callback decides whether to retry the call.

Requires the ``sandbox`` extra::

    pip install prompture[sandbox]
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..agents.tools_schema import ToolDefinition
from ..agents.types import ApprovalRequired


class PythonSandboxTool:
    """Build a sandboxed ``python_execute`` tool.

    Args:
        allowed_imports: Modules to allow.  Empty means use Tukuy's
            built-in ``SAFE_IMPORTS`` whitelist.
        blocked_imports: Modules to block on top of the always-blocked
            security list (``os``, ``subprocess``, ``socket`` etc.).
        allowed_read_paths: Directories the sandbox may read from.
            Empty means no filesystem reads.
        allowed_write_paths: Directories the sandbox may write to.
            Empty means no filesystem writes.
        allow_cwd: Allow read/write inside the current working directory.
        working_directory: Override the working directory used to resolve
            relative paths in restrictions.
        timeout_seconds: Maximum wall-clock time per call.  Enforced via
            ``SIGALRM`` on Unix; on Windows the limit is **not enforced**
            and the call runs to completion.
        max_memory_bytes: Per-call memory cap.  Unix only (``RLIMIT_AS``).
        max_output_chars: Truncate the returned stdout/return-value at
            this many characters.
        risk_threshold: AST risk level that triggers
            :class:`ApprovalRequired`.  Defaults to ``RiskLevel.HIGH`` —
            i.e. ``HIGH`` and ``CRITICAL`` code requires approval.
        tool_name: Override the tool's name (default ``python_execute``).
        tool_description: Override the tool's description shown to the LLM.

    Example::

        from prompture import DeepAgent, ToolRegistry
        from prompture.tools import PythonSandboxTool

        sandbox = PythonSandboxTool(
            allowed_imports=["json", "math", "statistics"],
            timeout_seconds=10,
        )
        registry = ToolRegistry()
        registry.add(sandbox.to_tool_definition())

        agent = DeepAgent("openai/gpt-4o", tools=registry)
        result = agent.run("Compute the stdev of [1, 4, 9, 16, 25].")
    """

    DEFAULT_DESCRIPTION = (
        "Execute Python code in a restricted sandbox. Use this for "
        "calculations, data processing, or any Python snippet. The "
        "code runs with a curated safe-imports whitelist, no network, "
        "and no filesystem access unless paths were configured. "
        "Returns stdout plus the final expression's value, or an "
        "error message starting with 'Error:'."
    )

    def __init__(
        self,
        *,
        allowed_imports: list[str] | None = None,
        blocked_imports: list[str] | None = None,
        allowed_read_paths: list[str | Path] | None = None,
        allowed_write_paths: list[str | Path] | None = None,
        allow_cwd: bool = False,
        working_directory: str | Path | None = None,
        timeout_seconds: float = 30.0,
        max_memory_bytes: int | None = None,
        max_output_chars: int = 10_000,
        risk_threshold: Any = None,
        auto_approve: bool = False,
        tool_name: str = "python_execute",
        tool_description: str | None = None,
    ) -> None:
        try:
            from tukuy.analysis import RiskLevel
            from tukuy.sandbox import PythonSandbox
        except ImportError as exc:
            raise ImportError(
                "PythonSandboxTool requires the 'sandbox' extra. Install with: pip install prompture[sandbox]"
            ) from exc

        self._RiskLevel = RiskLevel
        self.tool_name = tool_name
        self.tool_description = tool_description or self.DEFAULT_DESCRIPTION
        self.max_output_chars = max_output_chars
        self.risk_threshold = risk_threshold if risk_threshold is not None else RiskLevel.HIGH
        self.auto_approve = auto_approve
        # One-shot approval tokens — see :meth:`mark_approved`.
        self._approved_hashes: set[str] = set()

        self._sandbox = PythonSandbox(
            allowed_imports=allowed_imports,
            blocked_imports=blocked_imports,
            timeout_seconds=timeout_seconds,
            max_memory_bytes=max_memory_bytes,
            allowed_read_paths=allowed_read_paths,
            allowed_write_paths=allowed_write_paths,
            allow_cwd=allow_cwd,
            working_directory=working_directory,
            use_safe_imports=allowed_imports is None,
            validate_before_exec=True,
        )

    # ------------------------------------------------------------------
    # Tool function
    # ------------------------------------------------------------------

    def execute(self, code: str) -> str:
        """Analyse, gate on risk, then run *code* in the sandbox.

        Returns the formatted output string the LLM should see.
        """
        from tukuy.analysis import analyze_python

        analysis = analyze_python(code)
        if not analysis.syntax_valid:
            return f"Error: Syntax error: {analysis.syntax_error}"

        code_hash = self._hash_code(code)
        if self.auto_approve:
            gate_bypassed = True
        elif code_hash in self._approved_hashes:
            self._approved_hashes.discard(code_hash)
            gate_bypassed = True
        else:
            gate_bypassed = False

        threshold_order = [
            self._RiskLevel.LOW,
            self._RiskLevel.MEDIUM,
            self._RiskLevel.HIGH,
            self._RiskLevel.CRITICAL,
        ]
        if not gate_bypassed and threshold_order.index(analysis.risk_level) >= threshold_order.index(
            self.risk_threshold
        ):
            raise ApprovalRequired(
                tool_name=self.tool_name,
                action=f"Execute {analysis.risk_level.value}-risk Python code",
                details={
                    "code": code,
                    "code_hash": code_hash,
                    "risk_level": analysis.risk_level.value,
                    "reasons": list(analysis.risk.reasons),
                    "imports": list(analysis.features.imports),
                },
            )

        result = self._sandbox.execute(code)

        if not result.success:
            return f"Error: {result.error}"

        output = (result.output or "").rstrip()
        if result.return_value is not None:
            tail = f"Return value: {result.return_value!r}"
            output = f"{output}\n\n{tail}" if output else tail

        if not output:
            return "Code executed successfully (no output)."

        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + f"\n\n... (output truncated at {self.max_output_chars} chars)"
        return output

    # ------------------------------------------------------------------
    # Approval helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_code(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def mark_approved(self, code: str) -> None:
        """Grant one-shot approval for *code*.

        The next call to :meth:`execute` with this exact code will skip
        the AST risk gate.  Intended for use inside an agent's
        ``on_approval_needed`` callback so that after the user consents,
        the agent's automatic retry succeeds::

            sandbox = PythonSandboxTool()

            def on_approval(name, action, details):
                if confirm_with_user(details):
                    sandbox.mark_approved(details["code"])
                    return True
                return False

            agent = Agent("openai/gpt-4o",
                          tools=[sandbox.to_tool_definition()],
                          on_approval_needed=on_approval)

        Tukuy's runtime restrictions (import block, path restrictions,
        timeout, memory limit) still apply — approval bypasses **only**
        the AST risk gate.
        """
        self._approved_hashes.add(self._hash_code(code))

    def revoke_approval(self, code: str) -> None:
        """Remove a pending one-shot approval for *code*."""
        self._approved_hashes.discard(self._hash_code(code))

    # ------------------------------------------------------------------
    # Build the ToolDefinition
    # ------------------------------------------------------------------

    def to_tool_definition(self) -> ToolDefinition:
        """Return a :class:`ToolDefinition` ready to register on a tool registry."""

        def _python_execute(code: str) -> str:
            """Execute Python code safely in a sandbox.

            Args:
                code: Python source code to execute.
            """
            return self.execute(code)

        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute. Use print() for output.",
                }
            },
            "required": ["code"],
        }

        return ToolDefinition(
            name=self.tool_name,
            description=self.tool_description,
            parameters=parameters,
            function=_python_execute,
        )

    # ------------------------------------------------------------------
    # Convenience: register directly onto a registry
    # ------------------------------------------------------------------

    def register_on(self, registry: Any) -> ToolDefinition:
        """Add the tool to *registry* and return its definition."""
        td = self.to_tool_definition()
        registry.add(td)
        return td

    # ------------------------------------------------------------------
    # Make instances usable as a plain callable too
    # ------------------------------------------------------------------

    def __call__(self, code: str) -> str:
        return self.execute(code)


def python_execute_tool(**kwargs: Any) -> ToolDefinition:
    """Shortcut: build and return a sandboxed ``python_execute`` ToolDefinition.

    Accepts the same keyword arguments as :class:`PythonSandboxTool`.
    Use when you don't need to hold a reference to the wrapper.
    """
    return PythonSandboxTool(**kwargs).to_tool_definition()
