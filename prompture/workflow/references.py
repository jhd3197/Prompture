"""Late-bound reference resolver for workflow templates.

A node's ``config`` may embed references to upstream node outputs as template
strings like ``"{{summary.outputs[0].value}}"``. These are resolved *late* — at
the instant the consuming node runs, against the live result store — never at
build time. This module is pure (no I/O, no driver/jobs imports) and is shared
verbatim by the sync and (future) async schedulers.

Reference grammar::

    reference := '{{' expr '}}'
    expr      := root ( '.' IDENT | '[' INT ']' | '[' STR ']' )*
    root      := IDENT          ; a node id, or reserved 'inputs' / 'constants'

Canonical forms (matches the studio's ``{{nodeId.outputs[0].value}}``)::

    {{n.outputs[0].value}}     first output handle's value
    {{n.outputs.caption}}      output handle named 'caption'  (also {{n.outputs['caption']}})
    {{n.value}}                sugar: the sole output's value (errors if >1 output)
    {{n.status}} / {{n.usage.cost}}   node metadata
    {{inputs.topic}}           a graph-level input (reserved root)

Whole-string vs interpolation: if a value is exactly one ``{{expr}}`` it is
returned RAW (object/list/dict/MediaAsset preserved); embedded refs inside
surrounding text are ``str()``-interpolated. Parsing is regex + a tiny hand
tokenizer — never ``eval``/``format`` — so it is injection-safe.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ReferenceResolutionError

__all__ = [
    "RESERVED_ROOTS",
    "Reference",
    "extract_dependencies",
    "extract_refs",
    "parse_reference",
    "resolve_template",
]

RESERVED_ROOTS = frozenset({"inputs", "constants"})

# Capture the body of each {{ ... }} (non-greedy, dotall for safety).
_REF_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}", re.DOTALL)
_ROOT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
# One step: .attr | [int] | ['key'] | ["key"]
_STEP_RE = re.compile(
    r"""\.([A-Za-z_][A-Za-z0-9_]*)      # .attr      -> g1
      | \[\s*(-?\d+)\s*\]               # [int]      -> g2
      | \[\s*'([^']*)'\s*\]             # ['key']    -> g3
      | \[\s*"([^"]*)"\s*\]             # ["key"]    -> g4
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Reference:
    """A parsed ``{{...}}`` reference: a root plus a chain of attr/index steps."""

    root: str
    steps: tuple[tuple[str, Any], ...]  # ("attr", name) | ("index", int|str)
    raw: str

    def resolve(self, scope: Mapping[str, Any]) -> Any:
        if self.root not in scope:
            raise ReferenceResolutionError(self.raw, f"Unknown reference root {self.root!r}")
        obj = scope[self.root]
        for kind, val in self.steps:
            obj = _fold(obj, kind, val, self.raw)
        return obj


def parse_reference(body: str, raw: str | None = None) -> Reference:
    """Parse a reference body (the text between ``{{`` and ``}}``)."""
    raw = raw if raw is not None else "{{" + body + "}}"
    body = body.strip()
    m = _ROOT_RE.match(body)
    if not m:
        raise ReferenceResolutionError(raw, f"Malformed reference {raw!r}: missing root")
    root = m.group(0)
    rest = body[m.end() :]
    steps: list[tuple[str, Any]] = []
    pos = 0
    while pos < len(rest):
        sm = _STEP_RE.match(rest, pos)
        if not sm:
            raise ReferenceResolutionError(raw, f"Malformed reference {raw!r} near {rest[pos:]!r}")
        if sm.group(1) is not None:
            steps.append(("attr", sm.group(1)))
        elif sm.group(2) is not None:
            steps.append(("index", int(sm.group(2))))
        else:
            steps.append(("index", sm.group(3) if sm.group(3) is not None else sm.group(4)))
        pos = sm.end()
    return Reference(root=root, steps=tuple(steps), raw=raw)


def _model_dump(obj: Any) -> dict[str, Any] | None:
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            result = dump()
            if isinstance(result, dict):
                return result
        except Exception:
            return None
    return None


def _fold(obj: Any, kind: str, val: Any, raw: str) -> Any:
    if kind == "attr":
        if isinstance(obj, Mapping):
            if val in obj:
                return obj[val]
            raise ReferenceResolutionError(raw, f"Key {val!r} not found in {raw!r}")
        # Non-mapping object: attribute (no dunder/private), then getitem, then pydantic dump.
        if not val.startswith("_") and hasattr(obj, val):
            return getattr(obj, val)
        try:
            return obj[val]  # type: ignore[index]
        except (TypeError, KeyError, IndexError):
            pass
        dumped = _model_dump(obj)
        if dumped is not None and val in dumped:
            return dumped[val]
        raise ReferenceResolutionError(raw, f"Cannot resolve .{val} in {raw!r}")
    # index
    try:
        return obj[val]
    except (TypeError, KeyError, IndexError) as exc:
        raise ReferenceResolutionError(raw, f"Cannot index [{val!r}] in {raw!r}") from exc


def extract_refs(template: str) -> list[Reference]:
    """Parse every ``{{...}}`` in a template string into :class:`Reference` objects."""
    return [parse_reference(body, "{{" + body + "}}") for body in _REF_RE.findall(template)]


def extract_dependencies(config: Any) -> set[str]:
    """Collect the node-id roots referenced anywhere in *config*.

    Recurses dicts/lists, scans every string leaf, and excludes the reserved
    roots ``inputs`` / ``constants``.
    """
    deps: set[str] = set()
    _collect_deps(config, deps)
    return deps


def _collect_deps(value: Any, deps: set[str]) -> None:
    if isinstance(value, str):
        for body in _REF_RE.findall(value):
            try:
                ref = parse_reference(body)
            except ReferenceResolutionError:
                continue
            if ref.root not in RESERVED_ROOTS:
                deps.add(ref.root)
    elif isinstance(value, Mapping):
        for v in value.values():
            _collect_deps(v, deps)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_deps(v, deps)


def resolve_template(value: Any, scope: Mapping[str, Any], *, strict: bool = True) -> Any:
    """Resolve every ``{{...}}`` in *value* against *scope*.

    - ``dict`` / ``list``: recurse into elements.
    - whole-string single reference: return the referenced object UNCHANGED.
    - embedded references: ``str()``-interpolate each one.
    - other scalars: returned as-is.

    With ``strict=False`` an unresolvable reference is left in place instead of
    raising :class:`ReferenceResolutionError`.
    """
    if isinstance(value, Mapping):
        return {k: resolve_template(v, scope, strict=strict) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        resolved = [resolve_template(v, scope, strict=strict) for v in value]
        return type(value)(resolved) if isinstance(value, tuple) else resolved
    if not isinstance(value, str):
        return value

    matches = list(_REF_RE.finditer(value))
    if not matches:
        return value

    # Whole-string single reference: exactly one ref spanning the entire value
    # (ignoring surrounding whitespace) -> return the referenced object UNCHANGED.
    if len(matches) == 1:
        m = matches[0]
        if value[: m.start()].strip() == "" and value[m.end() :].strip() == "":
            ref = parse_reference(m.group(1), m.group(0))
            try:
                return ref.resolve(scope)
            except ReferenceResolutionError:
                if strict:
                    raise
                return value

    def _sub(match: re.Match[str]) -> str:
        ref = parse_reference(match.group(1), match.group(0))
        try:
            return str(ref.resolve(scope))
        except ReferenceResolutionError:
            if strict:
                raise
            return match.group(0)

    return _REF_RE.sub(_sub, value)
