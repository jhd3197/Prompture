"""Wrap command-line programs as safe, read-only agent tools.

A :class:`CLITool` describes one program (``gh``, ``yt-dlp``, ``kubectl``)
and an allowlist of :class:`CLICommand` entries — the only subcommands an
agent may run. Each command declares its arguments as :class:`CLIArg`
objects, which double as the JSON-schema parameters the model sees and as
the recipe that turns validated values into argv items.

Safety model:

* **No shell.** The program is executed from an argv list; values are never
  interpolated into a command string.
* **Allowlist.** Only declared commands run. Anything else is rejected before
  a process is started.
* **No flag injection.** A value for a positional argument may not start with
  ``-`` unless the argument opts in with ``allow_dash`` — which is only
  accepted when the command ends options with ``--``. Long-flag values are
  passed as one ``--flag=value`` item, so they can never become a new flag.
* **Bounded.** Every run has a timeout and a stdout byte cap; the child is
  killed when either is exceeded.
* **Quiet about secrets.** Injected environment values are never logged and
  are scrubbed from any output or error text.
* **Windows batch launchers refused.** ``.bat`` / ``.cmd`` files are run by
  ``cmd.exe``, which re-parses arguments; they are not executed.
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ...agents.tools_schema import ToolDefinition
from ...capabilities.health import HealthStatus
from ...capabilities.probe import cached_probe, install_hint, probe_command, probe_env
from ...security.redaction import scrub_secrets

logger = logging.getLogger("prompture.tools.cli")

OutputFormat = Literal["text", "json", "jsonl", "yaml"]
ArgType = Literal["string", "integer", "number", "boolean", "array"]

# Module-level so tests can simulate Windows without patching os.name globally.
_ON_WINDOWS = os.name == "nt"

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_OUTPUT_BYTES = 256_000
DEFAULT_MAX_VALUE_LENGTH = 2_000
_STDERR_CAP = 64_000

#: Programs that interpret their arguments as code. Wrapping one would turn
#: every argument into a command line, so they are refused outright.
SHELL_PROGRAMS = frozenset(
    {"sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh", "cmd", "powershell", "pwsh", "wsl", "busybox"}
)

_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_ENV_REF_RE = re.compile(r"^\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))$")
_STATUS_MAP = {"ok": "ok", "missing": "missing", "broken": "broken", "timeout": "timeout", "error": "error"}


class CLIToolError(ValueError):
    """Invalid definition, rejected command or invalid argument value."""


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------


@dataclass
class CLIArg:
    """One declared argument of a :class:`CLICommand`.

    Attributes:
        name: Parameter name the model fills in.
        type: ``string | integer | number | boolean | array`` (array of strings).
        description: Shown to the model.
        required: Whether the model must supply it.
        default: Value used when the model omits it (``None`` = omit the argument).
        flag: Flag that carries the value (``"--limit"``). ``None`` makes the
            argument positional. Long flags render as one ``--flag=value``
            item; short flags render as two items.
        false_flag: For booleans, the flag emitted when the value is false.
        enum: Allowed values.
        pattern: Regex every value (or array item) must fully match.
        allow_dash: Let a positional value start with ``-``. Requires the
            command to end options with ``--``.
        template: Format string producing the argv item; ``{value}`` is the
            argument and other parameters are available by name
            (``"ytsearch{max_results}:{value}"``).
        emit: ``False`` keeps the value out of argv (used only by templates).
        minimum / maximum: Numeric bounds.
        max_length: Maximum characters per string value.
        max_items: Maximum array length.
    """

    name: str
    type: ArgType = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    flag: str | None = None
    false_flag: str | None = None
    enum: list[Any] | None = None
    pattern: str | None = None
    allow_dash: bool = False
    template: str | None = None
    emit: bool = True
    minimum: float | None = None
    maximum: float | None = None
    max_length: int = DEFAULT_MAX_VALUE_LENGTH
    max_items: int = 20

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", self.name or ""):
            raise CLIToolError(f"Invalid argument name {self.name!r}")
        if self.type not in ("string", "integer", "number", "boolean", "array"):
            raise CLIToolError(f"Argument {self.name!r}: unsupported type {self.type!r}")
        for f in (self.flag, self.false_flag):
            if f is not None and (not f.startswith("-") or "=" in f or any(c.isspace() for c in f) or "\x00" in f):
                raise CLIToolError(f"Argument {self.name!r}: invalid flag {f!r}")
        if self.type == "boolean" and self.flag is None and self.emit:
            raise CLIToolError(f"Argument {self.name!r}: boolean arguments need a flag")
        if self.pattern is not None:
            try:
                self._compiled = re.compile(self.pattern)
            except re.error as exc:
                raise CLIToolError(f"Argument {self.name!r}: bad pattern: {exc}") from exc
        else:
            self._compiled = None

    @property
    def positional(self) -> bool:
        return self.flag is None

    # -- schema ---------------------------------------------------------

    def json_schema(self) -> dict[str, Any]:
        item: dict[str, Any] = {"type": "string"}
        schema: dict[str, Any]
        if self.type == "array":
            if self.enum:
                item["enum"] = list(self.enum)
            if self.pattern:
                item["pattern"] = self.pattern
            schema = {"type": "array", "items": item, "maxItems": self.max_items}
        else:
            schema = {"type": self.type}
            if self.enum:
                schema["enum"] = list(self.enum)
            if self.pattern and self.type == "string":
                schema["pattern"] = self.pattern
            if self.minimum is not None and self.type in ("integer", "number"):
                schema["minimum"] = self.minimum
            if self.maximum is not None and self.type in ("integer", "number"):
                schema["maximum"] = self.maximum
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        return schema

    # -- validation -----------------------------------------------------

    def _check_string(self, value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise CLIToolError(f"{self.name}: expected a string, got {type(value).__name__}")
        text = str(value)
        if "\x00" in text:
            raise CLIToolError(f"{self.name}: NUL bytes are not allowed")
        if len(text) > self.max_length:
            raise CLIToolError(f"{self.name}: value longer than {self.max_length} characters")
        if self.enum is not None and text not in [str(e) for e in self.enum]:
            raise CLIToolError(f"{self.name}: {text!r} is not one of {', '.join(map(str, self.enum))}")
        if self._compiled is not None and not self._compiled.fullmatch(text):
            raise CLIToolError(f"{self.name}: {text!r} does not match the allowed format {self.pattern}")
        return text

    def coerce(self, value: Any) -> Any:
        """Validate *value* and return it in canonical form."""
        if self.type == "string":
            return self._check_string(value)
        if self.type in ("integer", "number"):
            if isinstance(value, bool):
                raise CLIToolError(f"{self.name}: expected a number, got a boolean")
            try:
                num: float | int = int(value) if self.type == "integer" else float(value)
            except (TypeError, ValueError):
                raise CLIToolError(f"{self.name}: expected {self.type}, got {value!r}") from None
            if self.type == "integer" and isinstance(value, float) and not value.is_integer():
                raise CLIToolError(f"{self.name}: expected an integer, got {value!r}")
            if self.minimum is not None and num < self.minimum:
                raise CLIToolError(f"{self.name}: must be >= {self.minimum:g}")
            if self.maximum is not None and num > self.maximum:
                raise CLIToolError(f"{self.name}: must be <= {self.maximum:g}")
            if self.enum is not None and num not in self.enum:
                raise CLIToolError(f"{self.name}: {num!r} is not one of {self.enum}")
            return num
        if self.type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            raise CLIToolError(f"{self.name}: expected a boolean, got {value!r}")
        # array
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise CLIToolError(f"{self.name}: expected an array of strings")
        items = [self._check_string(v) for v in value]
        if len(items) > self.max_items:
            raise CLIToolError(f"{self.name}: at most {self.max_items} items")
        return items

    # -- rendering ------------------------------------------------------

    def _format(self, value: Any, params: Mapping[str, Any]) -> str:
        if self.template is None:
            return str(value)
        try:
            return self.template.format(**{**params, "value": value})
        except (KeyError, IndexError, ValueError) as exc:
            raise CLIToolError(f"{self.name}: template could not be rendered ({exc})") from None

    def _guard_dash(self, text: str) -> None:
        if text.startswith("-") and not self.allow_dash:
            raise CLIToolError(
                f"{self.name}: values may not start with '-' (they would be read as an option): {text[:40]!r}"
            )

    def render(self, value: Any, params: Mapping[str, Any]) -> tuple[list[str], list[str]]:
        """Return ``(option_items, positional_items)`` for a validated *value*."""
        if not self.emit or value is None:
            return [], []
        if self.type == "boolean":
            if value and self.flag:
                return [self.flag], []
            if not value and self.false_flag:
                return [self.false_flag], []
            return [], []
        values = value if self.type == "array" else [value]
        options: list[str] = []
        positionals: list[str] = []
        for v in values:
            text = self._format(v, params)
            if "\x00" in text:
                raise CLIToolError(f"{self.name}: NUL bytes are not allowed")
            if self.flag is None:
                self._guard_dash(text)
                positionals.append(text)
            elif self.flag.startswith("--"):
                options.append(f"{self.flag}={text}")
            else:
                self._guard_dash(text)
                options.extend([self.flag, text])
        return options, positionals

    # -- config ---------------------------------------------------------

    @classmethod
    def from_dict(cls, name: str | None, data: Mapping[str, Any]) -> CLIArg:
        allowed = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
        unknown = set(data) - allowed
        if unknown:
            raise CLIToolError(f"Argument {name or data.get('name')!r}: unknown keys {sorted(unknown)}")
        kwargs = dict(data)
        if name is not None:
            kwargs.setdefault("name", name)
        if "enum" in kwargs and kwargs["enum"] is not None:
            kwargs["enum"] = list(kwargs["enum"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

ValidateFn = Callable[[dict[str, Any]], "str | None"]
PostprocessFn = Callable[[str, dict[str, Any]], str]


@dataclass
class CLICommand:
    """One allowlisted subcommand of a :class:`CLITool`.

    Attributes:
        name: Short name; the tool is exposed as ``<tool>_<name>``.
        subcommand: Leading argv items (``("issue", "list")``); may be empty.
        description: Shown to the model.
        args: Declared arguments.
        fixed_args: Items always passed after the subcommand. With
            ``temp_dir`` the placeholder ``{tmpdir}`` is replaced by the
            run's private temporary directory.
        output: ``text | json | jsonl | yaml``; structured output is parsed
            and re-serialised compactly (raw text on parse failure).
        json_keys: Keep only these top-level keys of parsed objects.
        end_of_options: Insert ``--`` before positional values.
        timeout / max_output_bytes: Per-command overrides.
        temp_dir: Run with a fresh temporary directory (removed afterwards).
        collect: Glob (relative to the temporary directory) of files whose
            contents become the output instead of stdout.
        postprocess: ``fn(text, params) -> text`` applied to the final output.
        validate: ``fn(params) -> error | None`` extra check before running.
    """

    name: str
    subcommand: tuple[str, ...] = ()
    description: str = ""
    args: list[CLIArg] = field(default_factory=list)
    fixed_args: tuple[str, ...] = ()
    output: OutputFormat = "text"
    json_keys: tuple[str, ...] | None = None
    end_of_options: bool = False
    timeout: float | None = None
    max_output_bytes: int | None = None
    temp_dir: bool = False
    collect: str | None = None
    postprocess: PostprocessFn | None = None
    validate: ValidateFn | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_]{1,40}", self.name or ""):
            raise CLIToolError(f"Invalid command name {self.name!r}")
        if isinstance(self.subcommand, str):
            self.subcommand = tuple(self.subcommand.split())
        self.subcommand = tuple(self.subcommand)
        self.fixed_args = tuple(self.fixed_args)
        if self.json_keys is not None:
            self.json_keys = tuple(self.json_keys)
        for item in (*self.subcommand, *self.fixed_args):
            if not isinstance(item, str) or not item or "\x00" in item:
                raise CLIToolError(f"Command {self.name!r}: invalid fixed argv item {item!r}")
        if self.output not in ("text", "json", "jsonl", "yaml"):
            raise CLIToolError(f"Command {self.name!r}: unsupported output {self.output!r}")
        if self.collect and not self.temp_dir:
            raise CLIToolError(f"Command {self.name!r}: 'collect' requires temp_dir")
        if self.collect and (os.path.isabs(self.collect) or ".." in self.collect.replace("\\", "/").split("/")):
            raise CLIToolError(f"Command {self.name!r}: 'collect' must be a relative glob")
        seen: set[str] = set()
        for arg in self.args:
            if arg.name in seen:
                raise CLIToolError(f"Command {self.name!r}: duplicate argument {arg.name!r}")
            seen.add(arg.name)
            if arg.positional and arg.allow_dash and not self.end_of_options:
                raise CLIToolError(
                    f"Command {self.name!r}: argument {arg.name!r} allows a leading '-' "
                    "but the command does not end options with '--'"
                )

    @property
    def path(self) -> str:
        return " ".join(self.subcommand)

    def parameters(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {a.name: a.json_schema() for a in self.args},
        }
        required = [a.name for a in self.args if a.required]
        if required:
            schema["required"] = required
        return schema

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CLICommand:
        allowed = {"name", "subcommand", "description", "args", "fixed_args", "output", "json_keys"}
        allowed |= {"end_of_options", "timeout", "max_output_bytes", "temp_dir", "collect"}
        unknown = set(data) - allowed
        if unknown:
            raise CLIToolError(f"Command {data.get('name')!r}: unknown keys {sorted(unknown)}")
        kwargs = dict(data)
        raw_args = kwargs.pop("args", None) or []
        if isinstance(raw_args, Mapping):
            args = [CLIArg.from_dict(n, spec or {}) for n, spec in raw_args.items()]
        else:
            args = [CLIArg.from_dict(None, spec) for spec in raw_args]
        sub = kwargs.pop("subcommand", ())
        if isinstance(sub, str):
            sub = sub.split()
        name = kwargs.pop("name", None) or "_".join(sub) or "run"
        return cls(
            name=str(name).replace("-", "_"),
            subcommand=tuple(str(s) for s in sub),
            args=args,
            fixed_args=tuple(str(a) for a in kwargs.pop("fixed_args", ()) or ()),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------


@dataclass
class ProcessOutput:
    """Raw result of :func:`run_process`."""

    exit_code: int | None
    stdout: bytes
    stderr: bytes
    truncated: bool = False
    timed_out: bool = False
    elapsed_ms: int = 0


def run_process(
    argv: Sequence[str],
    *,
    env: dict[str, str] | None,
    timeout: float,
    max_bytes: int,
    cwd: str | None = None,
) -> ProcessOutput:
    """Run *argv* without a shell, capping stdout at *max_bytes* and wall time at *timeout*.

    The child is killed as soon as stdout exceeds the cap or the timeout
    passes, so a runaway program can't exhaust memory or hang the agent.
    """
    start = time.monotonic()
    proc = subprocess.Popen(  # nosec B603 - argv list, shell=False, allowlisted program
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        env=env,
        cwd=cwd,
        shell=False,
    )
    out = bytearray()
    err = bytearray()
    state = {"truncated": False}

    def _kill() -> None:
        with contextlib.suppress(OSError):
            proc.kill()

    def _pump(stream: Any, buf: bytearray, cap: int, is_stdout: bool) -> None:
        try:
            while True:
                chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
                if not chunk:
                    break
                room = cap - len(buf)
                if room > 0:
                    buf.extend(chunk[:room])
                if len(chunk) > room and is_stdout:
                    state["truncated"] = True
                    _kill()
                    break
                # stderr past its cap: keep draining so the child never blocks on a full pipe
        except (OSError, ValueError):
            pass

    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, out, max_bytes, True), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, err, _STDERR_CAP, False), daemon=True),
    ]
    for t in threads:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)
    for t in threads:
        t.join(timeout=5)
    for stream in (proc.stdout, proc.stderr):
        try:
            if stream is not None:
                stream.close()
        except OSError:
            pass
    return ProcessOutput(
        exit_code=proc.returncode,
        stdout=bytes(out),
        stderr=bytes(err),
        truncated=state["truncated"],
        timed_out=timed_out,
        elapsed_ms=int((time.monotonic() - start) * 1000),
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CLIResult:
    """Outcome of :meth:`CLITool.run`. ``output`` is what the model sees."""

    tool: str
    command: str
    argv: list[str] = field(default_factory=list)
    output: str = ""
    parsed: Any = None
    exit_code: int | None = None
    stderr: str = ""
    truncated: bool = False
    timed_out: bool = False
    elapsed_ms: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_text(self) -> str:
        return self.output if self.ok else f"Error: {self.error}"


def _select_keys(obj: Any, keys: tuple[str, ...] | None) -> Any:
    if not keys:
        return obj
    if isinstance(obj, dict):
        return {k: obj[k] for k in keys if k in obj}
    if isinstance(obj, list):
        return [_select_keys(item, keys) for item in obj]
    return obj


def _parse_output(text: str, fmt: OutputFormat) -> Any:
    """Parse *text* per *fmt*; raises ``ValueError`` when it does not parse."""
    if fmt == "json":
        return json.loads(text)
    if fmt == "jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if fmt == "yaml":
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ValueError("PyYAML is not installed") from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(str(exc)) from exc
    raise ValueError(f"not a structured format: {fmt}")


def _redact(text: str, secrets: Iterable[str]) -> str:
    for value in secrets:
        if value and len(value) >= 4:
            text = text.replace(value, "***")
    return scrub_secrets(text)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@dataclass
class CLITool:
    """A command-line program exposed to agents through allowlisted commands.

    Attributes:
        name: Tool name (``"gh"``, ``"yt-dlp"``); health row ``cli:<name>``.
        command: Binary name or path.
        commands: The read-only allowlist. Anything not listed is rejected.
        description: One-line summary.
        version_args: Side-effect-free probe arguments.
        live_check_args: Extra probe run only by ``check(live=True)``.
        timeout / max_output_bytes: Defaults for every command.
        env: Variables injected into the child. A value of ``$NAME`` or
            ``${NAME}`` is read from the parent environment at run time, so
            secrets never have to live in a config file. Values are never logged.
        install_hint: Fix hint when the binary is missing.
        prefix: Tool-name prefix (defaults to *name* with ``-`` → ``_``).
    """

    name: str
    command: str
    commands: list[CLICommand] = field(default_factory=list)
    description: str = ""
    version_args: tuple[str, ...] = ("--version",)
    live_check_args: tuple[str, ...] | None = None
    timeout: float = DEFAULT_TIMEOUT
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    env: dict[str, str] = field(default_factory=dict)
    install_hint: str | None = None
    prefix: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,40}", self.name or ""):
            raise CLIToolError(f"Invalid CLI tool name {self.name!r}")
        if not self.command or "\x00" in self.command:
            raise CLIToolError(f"CLI tool {self.name!r}: missing command")
        base = os.path.basename(self.command).lower()
        for suffix in (".exe", ".com"):
            base = base.removesuffix(suffix)
        if base in SHELL_PROGRAMS:
            raise CLIToolError(f"CLI tool {self.name!r}: wrapping a shell ({self.command}) is not allowed")
        self.version_args = tuple(self.version_args)
        if self.live_check_args is not None:
            self.live_check_args = tuple(self.live_check_args)
        if self.prefix is None:
            self.prefix = re.sub(r"[^A-Za-z0-9_]", "_", self.name)
        names: set[str] = set()
        for cmd in self.commands:
            if cmd.name in names:
                raise CLIToolError(f"CLI tool {self.name!r}: duplicate command {cmd.name!r}")
            names.add(cmd.name)
            tool_name = self.tool_name(cmd)
            if not _TOOL_NAME_RE.match(tool_name):
                raise CLIToolError(f"CLI tool {self.name!r}: tool name {tool_name!r} is invalid")
        for key in self.env:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise CLIToolError(f"CLI tool {self.name!r}: invalid env var name {key!r}")

    # -- lookup ---------------------------------------------------------

    def tool_name(self, cmd: CLICommand) -> str:
        return f"{self.prefix}_{cmd.name}"

    def allowed(self) -> list[str]:
        """Human-readable allowlist (``"gh issue list (issue_list)"``)."""
        return [f"{' '.join([self.name, *c.subcommand])} ({c.name})" for c in self.commands]

    def get_command(self, name: str) -> CLICommand:
        """Find an allowlisted command by name, tool name or subcommand path.

        Raises:
            CLIToolError: The command is not on the allowlist.
        """
        wanted = " ".join(str(name).split())
        for cmd in self.commands:
            if wanted in (cmd.name, self.tool_name(cmd)) or (cmd.path and wanted == cmd.path):
                return cmd
        raise CLIToolError(
            f"`{self.name} {wanted}` is not an allowed command. Allowed: {', '.join(self.allowed()) or 'none'}"
        )

    # -- argv -----------------------------------------------------------

    def _validated_params(self, cmd: CLICommand, params: Mapping[str, Any]) -> dict[str, Any]:
        declared = {a.name: a for a in cmd.args}
        unknown = sorted(set(params) - set(declared))
        if unknown:
            raise CLIToolError(f"Unknown argument(s) for {self.tool_name(cmd)}: {', '.join(unknown)}")
        values: dict[str, Any] = {}
        for arg in cmd.args:
            raw = params.get(arg.name)
            if raw is None:
                raw = arg.default
            if raw is None:
                if arg.required:
                    raise CLIToolError(f"{self.tool_name(cmd)}: missing required argument {arg.name!r}")
                values[arg.name] = None
                continue
            values[arg.name] = arg.coerce(raw)
        if cmd.validate is not None:
            problem = cmd.validate(values)
            if problem:
                raise CLIToolError(problem)
        return values

    def build_argv(
        self, command: str, params: Mapping[str, Any] | None = None, *, tmpdir: str | None = None
    ) -> list[str]:
        """Validate *params* and build the argv list (``argv[0]`` is :attr:`command`).

        Raises:
            CLIToolError: Rejected command, unknown/invalid argument or flag injection.
        """
        return self._prepare(self.get_command(command), params or {}, tmpdir)[0]

    def _prepare(
        self, cmd: CLICommand, params: Mapping[str, Any], tmpdir: str | None
    ) -> tuple[list[str], dict[str, Any]]:
        values = self._validated_params(cmd, params)
        fixed = list(cmd.fixed_args)
        if cmd.temp_dir:
            if tmpdir is None:
                tmpdir = os.path.join(tempfile.gettempdir(), "prompture-cli")
            fixed = [a.replace("{tmpdir}", tmpdir) for a in fixed]
        options: list[str] = []
        positionals: list[str] = []
        template_params = {k: ("" if v is None else v) for k, v in values.items()}
        for arg in cmd.args:
            opts, pos = arg.render(values[arg.name], template_params)
            options.extend(opts)
            positionals.extend(pos)
        argv = [self.command, *cmd.subcommand, *fixed, *options]
        if positionals:
            if cmd.end_of_options:
                argv.append("--")
            argv.extend(positionals)
        return argv, values

    # -- environment ----------------------------------------------------

    def _injected_env(self) -> tuple[dict[str, str], list[str]]:
        """Resolve :attr:`env` → ``(values, missing_references)``."""
        values: dict[str, str] = {}
        missing: list[str] = []
        for key, raw in self.env.items():
            match = _ENV_REF_RE.match(str(raw))
            if match:
                ref = match.group("braced") or match.group("bare")
                found = os.environ.get(ref)
                if found:
                    values[key] = found
                else:
                    missing.append(ref)
            else:
                values[key] = str(raw)
        return values, missing

    # -- health ---------------------------------------------------------

    def _resolve(self) -> str | None:
        return shutil.which(self.command)

    @staticmethod
    def _is_batch_launcher(path: str) -> bool:
        return _ON_WINDOWS and path.lower().endswith((".bat", ".cmd"))

    def is_active(self) -> bool:
        """True when the binary exists and its version probe succeeds."""
        path = self._resolve()
        if path is None or self._is_batch_launcher(path):
            return False
        return cached_probe(self.command, self.version_args, hint=self.install_hint).ok

    def check(self, live: bool = False) -> HealthStatus:
        """Health row for doctor (``cli:<name>``, category ``tools``)."""
        row_name = f"cli:{self.name}"
        details: dict[str, Any] = {"command": self.command, "allowed": self.allowed(), "tools": self.tool_names()}
        path = self._resolve()
        if path is not None and self._is_batch_launcher(path):
            return HealthStatus(
                row_name,
                "broken",
                category="tools",
                message=f"{path} is a batch launcher; refusing to pass arguments through cmd.exe",
                fix_hint=f"Point `command` at the real executable for {self.name}.",
                details={**details, "path": path},
            )
        probe = cached_probe(self.command, self.version_args, hint=self.install_hint or install_hint(self.command))
        details.update(path=probe.path, version=probe.version if probe.ok else None, exit_code=probe.exit_code)
        if not probe.ok:
            message = probe.output.splitlines()[0] if probe.output else probe.status
            return HealthStatus(
                row_name,
                _STATUS_MAP[probe.status],  # type: ignore[arg-type]
                category="tools",
                message=message[:200],
                fix_hint=probe.hint,
                details=details,
            )
        _, missing = self._injected_env()
        status = "ok"
        message = f"{probe.version or 'ok'} — {len(self.commands)} read-only command(s)"
        fix: str | None = None
        if missing:
            status = "degraded"
            fix = "Set " + ", ".join(sorted(set(missing))) + f" (referenced by {self.name} env)."
            details["missing_env"] = sorted(set(missing))
        if live and self.live_check_args:
            env_values, _ = self._injected_env()
            live_probe = probe_command(self.command, self.live_check_args, env=env_values or None, timeout=20)
            details["live"] = live_probe.status
            if not live_probe.ok:
                status = "degraded"
                first = live_probe.output.splitlines()[0] if live_probe.output else live_probe.status
                message = f"{message}; live check failed: {_redact(first, env_values.values())[:160]}"
                fix = fix or live_probe.hint
        return HealthStatus(
            row_name,
            status,
            category="tools",
            active_backend=probe.path,
            message=message,
            fix_hint=fix,
            details=details,
        )  # type: ignore[arg-type]

    # -- execution ------------------------------------------------------

    def run(self, command: str, **params: Any) -> CLIResult:
        """Run an allowlisted command. Never raises; failures land in ``result.error``."""
        label = f"{self.name} {command}".strip()
        result = CLIResult(tool=self.name, command=label)
        try:
            cmd = self.get_command(command)
            label = " ".join([self.name, *cmd.subcommand]) or self.name
            result.command = label
        except CLIToolError as exc:
            result.error = str(exc)
            return result

        env_values, missing = self._injected_env()
        secrets = list(env_values.values())
        tmp_ctx = tempfile.TemporaryDirectory(prefix="prompture-cli-") if cmd.temp_dir else None
        try:
            tmpdir = tmp_ctx.name if tmp_ctx else None
            try:
                argv, values = self._prepare(cmd, params, tmpdir)
            except CLIToolError as exc:
                result.error = _redact(str(exc), secrets)
                return result
            result.argv = argv

            path = self._resolve()
            if path is None:
                result.error = f"`{self.command}` is not installed. {self.install_hint or install_hint(self.command)}"
                return result
            if self._is_batch_launcher(path):
                result.error = f"`{path}` is a batch launcher; refusing to run it with model-supplied arguments."
                return result
            if missing:
                logger.debug("cli tool %s: env references not set: %s", self.name, ", ".join(missing))

            timeout = cmd.timeout or self.timeout
            cap = cmd.max_output_bytes or self.max_output_bytes
            logger.debug(
                "cli tool %s: running %s (injected env keys: %s)",
                self.name,
                _redact(" ".join(argv), secrets),
                ", ".join(sorted(env_values)) or "none",
            )
            try:
                proc = run_process(
                    [path, *argv[1:]],
                    env=probe_env(extra=env_values or None),
                    timeout=timeout,
                    max_bytes=cap,
                    cwd=tmpdir,
                )
            except OSError as exc:
                result.error = _redact(f"could not start `{self.command}`: {exc}", secrets)
                return result

            result.exit_code = proc.exit_code
            result.elapsed_ms = proc.elapsed_ms
            result.truncated = proc.truncated
            result.timed_out = proc.timed_out
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = _redact(proc.stderr.decode("utf-8", errors="replace"), secrets)
            result.stderr = stderr[:4000]

            if proc.timed_out:
                result.error = f"`{label}` timed out after {timeout:g}s"
                return result
            if not proc.truncated and proc.exit_code not in (0, None):
                detail = (stderr.strip() or stdout.strip())[:2000]
                result.error = f"`{label}` exited with code {proc.exit_code}" + (f": {detail}" if detail else "")
                return result

            text = stdout
            if cmd.collect and tmpdir:
                text = self._collect(tmpdir, cmd.collect, cap)
                if not text.strip():
                    note = (stderr.strip() or stdout.strip())[:500]
                    result.error = f"`{label}` produced no output files" + (f": {note}" if note else "")
                    return result

            output = text
            if cmd.output != "text" and not proc.truncated:
                try:
                    parsed = _select_keys(_parse_output(text, cmd.output), cmd.json_keys)
                except ValueError:
                    logger.debug("cli tool %s: %s output did not parse; returning text", self.name, cmd.output)
                else:
                    result.parsed = parsed
                    output = json.dumps(parsed, ensure_ascii=False, default=str)
            if cmd.postprocess is not None:
                output = cmd.postprocess(output, values)
            if proc.truncated:
                output = output.rstrip() + f"\n[output truncated at {cap} bytes]"
            result.output = _redact(output, secrets)
            return result
        except Exception as exc:  # a tool must never raise into the agent loop
            logger.debug("cli tool %s failed", self.name, exc_info=True)
            result.error = _redact(f"{type(exc).__name__}: {exc}", secrets)
            return result
        finally:
            if tmp_ctx is not None:
                try:
                    tmp_ctx.cleanup()
                except OSError:
                    logger.debug("could not remove %s", tmp_ctx.name, exc_info=True)

    @staticmethod
    def _collect(tmpdir: str, pattern: str, cap: int) -> str:
        parts: list[str] = []
        used = 0
        root = os.path.realpath(tmpdir)
        for path in sorted(glob.glob(os.path.join(tmpdir, pattern))):
            real = os.path.realpath(path)
            if not real.startswith(root + os.sep) or not os.path.isfile(real):
                continue
            room = cap - used
            if room <= 0:
                break
            with open(real, "rb") as fh:
                data = fh.read(room)
            used += len(data)
            parts.append(f"## {os.path.basename(path)}\n{data.decode('utf-8', errors='replace')}")
        return "\n\n".join(parts)

    def call(self, command: str, **params: Any) -> str:
        """Run *command* and return the model-facing string (``"Error: ..."`` on failure)."""
        return self.run(command, **params).to_text()

    # -- tool definitions -----------------------------------------------

    def tool_names(self) -> list[str]:
        return [self.tool_name(c) for c in self.commands]

    def to_tool_definition(self, command: str) -> ToolDefinition:
        cmd = self.get_command(command)
        tool = self

        def _invoke(**kwargs: Any) -> str:
            return tool.call(cmd.name, **kwargs)

        _invoke.__name__ = self.tool_name(cmd)
        shown = " ".join([self.command, *cmd.subcommand])
        description = cmd.description or f"Run `{shown}` (read-only)."
        return ToolDefinition(
            name=self.tool_name(cmd),
            description=description[:1024],
            parameters=cmd.parameters(),
            function=_invoke,
            metadata={"source": "cli", "cli_tool": self.name, "command": shown, "is_write": False},
        )

    def to_tool_definitions(self) -> list[ToolDefinition]:
        return [self.to_tool_definition(c.name) for c in self.commands]

    # -- config ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, name: str | None = None) -> CLITool:
        """Build a tool from a config mapping (see :mod:`prompture.tools.cli.config`)."""
        allowed = {"name", "command", "commands", "description", "version_args", "live_check_args"}
        allowed |= {"timeout", "max_output_bytes", "env", "install_hint", "prefix"}
        unknown = set(data) - allowed
        tool_name = name or data.get("name")
        if unknown:
            raise CLIToolError(f"CLI tool {tool_name!r}: unknown keys {sorted(unknown)}")
        if not tool_name:
            raise CLIToolError("CLI tool definition needs a name")
        raw_cmds = data.get("commands") or []
        if isinstance(raw_cmds, Mapping):
            raw_cmds = [{"name": k, **(v or {})} for k, v in raw_cmds.items()]
        if not raw_cmds:
            raise CLIToolError(f"CLI tool {tool_name!r}: declare at least one allowed command")
        env = data.get("env") or {}
        if not isinstance(env, Mapping):
            raise CLIToolError(f"CLI tool {tool_name!r}: env must be a mapping")
        kwargs: dict[str, Any] = {
            "name": str(tool_name),
            "command": str(data.get("command") or tool_name),
            "commands": [CLICommand.from_dict(c) for c in raw_cmds],
            "description": str(data.get("description") or ""),
            "env": {str(k): str(v) for k, v in env.items()},
        }
        for key in ("version_args", "live_check_args"):
            if data.get(key) is not None:
                kwargs[key] = tuple(str(a) for a in data[key])
        for key in ("timeout", "max_output_bytes", "install_hint", "prefix"):
            if data.get(key) is not None:
                kwargs[key] = data[key]
        return cls(**kwargs)
