"""Shared JSON encoder for Prompture.

Replaces the ``json.dumps(..., default=str)`` pattern that was duplicated
across ~15 call sites. The previous pattern coerced *everything* unknown
to a string, including silently stringifying objects that should have
serialized into structured forms (Pydantic models, sets, bytes…). This
encoder handles the common Python types Prompture actually emits.

Usage::

    from prompture._internal.json_encoder import dumps, PromptureJSONEncoder

    text = dumps(payload)                      # one-shot
    text = json.dumps(payload, cls=PromptureJSONEncoder)  # in existing call

Types handled (in order of precedence):

* ``datetime`` / ``date`` / ``time``  → ISO 8601 string
* ``timedelta``                       → total seconds (float)
* ``Decimal``                         → str (preserves precision)
* ``UUID``                            → str
* ``Enum``                            → ``.value``
* ``pathlib.PurePath``                → ``str(path)``
* ``pydantic.BaseModel``              → ``model.model_dump(mode="json")``
* ``set`` / ``frozenset``             → sorted list (deterministic output)
* ``bytes`` / ``bytearray``           → ``ascii`` if printable, else ``base64``
* anything with ``__dict__``          → its ``__dict__``
* fallback                            → ``str(obj)`` (matches old behaviour)
"""

from __future__ import annotations

import base64
import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any
from uuid import UUID


def _encode_bytes(raw: bytes | bytearray) -> str:
    """Round-trip bytes as ASCII when printable, otherwise base64."""
    blob = bytes(raw)
    try:
        decoded = blob.decode("ascii")
    except UnicodeDecodeError:
        return base64.b64encode(blob).decode("ascii")
    if decoded.isprintable():
        return decoded
    return base64.b64encode(blob).decode("ascii")


class PromptureJSONEncoder(json.JSONEncoder):
    """JSON encoder that knows how to serialise Prompture's common types.

    Falls back to ``str(obj)`` for anything unrecognised so this is a
    drop-in replacement for ``json.dumps(..., default=str)``.
    """

    def default(self, o: Any) -> Any:
        # datetime/date/time first because datetime is a subclass of date
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, time):
            return o.isoformat()
        if isinstance(o, timedelta):
            return o.total_seconds()
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, PurePath):
            return str(o)
        if isinstance(o, (set, frozenset)):
            try:
                return sorted(o)
            except TypeError:
                # Items not comparable (e.g. raw Enum members). Fall back to a
                # str()-keyed sort so output stays deterministic — important for
                # cache-key stability.
                return sorted(o, key=str)
        if isinstance(o, (bytes, bytearray)):
            return _encode_bytes(o)

        # Pydantic BaseModel — imported lazily to avoid hard dep at import time.
        try:
            from pydantic import BaseModel
        except ImportError:
            BaseModel = None  # type: ignore[assignment]
        if BaseModel is not None and isinstance(o, BaseModel):
            return o.model_dump(mode="json")

        if hasattr(o, "__dict__"):
            return o.__dict__

        # Final fallback: stringify. Matches the legacy ``default=str`` behaviour
        # so this encoder cannot regress callers that depended on it.
        return str(o)


def dumps(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` shortcut using :class:`PromptureJSONEncoder`.

    Any keyword args are forwarded to :func:`json.dumps`. Setting ``cls``
    explicitly will override the default encoder.
    """
    kwargs.setdefault("cls", PromptureJSONEncoder)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, **kwargs)


__all__ = ["PromptureJSONEncoder", "dumps"]
