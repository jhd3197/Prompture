# tools.py
"""
Tools for enhanced type validation, parsing, and field extraction.

This module provides utilities for:
1. Type determination and JSON schema creation
2. Value conversion with support for human-readable formats
3. Exclusive field extraction against Pydantic models
4. Safe JSON text extraction from messy LLM output
5. Small parsing helpers (booleans, lists, datetimes)
6. Lightweight, flexible debug logging with levels

Notes:
- Only standard lib + pydantic + python-dateutil are required.
- Functions are defensive and avoid raising unless necessary for correctness.
"""
from __future__ import annotations

import re
import sys
import json
import decimal
from decimal import Decimal, InvalidOperation
from datetime import date, time, datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    get_origin,
    get_args,
    Iterable,
    Mapping,
    Tuple,
)
import uuid

import dateutil.parser
from pydantic import BaseModel
from tukuy import TukuyTransformer

# Initialize Tukuy transformer
TUKUY = TukuyTransformer()

__all__ = [
    "create_field_schema",
    "convert_value",
    "extract_fields",
    "parse_shorthand_number",
    "parse_boolean",
    "parse_datetime",
    "as_list",
    "clean_json_text",
    "log_debug",
    "LogLevel",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class LogLevel(int, Enum):
    OFF = 1000
    ERROR = 40
    WARN = 30
    INFO = 20
    DEBUG = 10
    TRACE = 5  # very verbose


def log_debug(
    level: int | LogLevel,
    current_level: int | LogLevel,
    msg: str | Mapping[str, Any] | Iterable[Tuple[str, Any]],
    *,
    prefix: str = "",
    stream = None,
    ts: bool = False,
    json_mode: bool = False,
) -> None:
    """
    Simple leveled logger.

    Args:
        level: Level of this message.
        current_level: Minimum level that should be emitted.
        msg: Message string OR a mapping/iterable of (key, value) to print.
        prefix: Optional prefix (e.g., "[extractor] ").
        stream: File-like stream; defaults to sys.stderr.
        ts: If True, prepend ISO timestamp.
        json_mode: If True, print as a single JSON object line.

    Examples:
        log_debug(LogLevel.DEBUG, current, "Parsed field X")
        log_debug(LogLevel.INFO, current, {"field": "age", "value": 42})
    """
    if int(current_level) > int(level):
        return

    stream = stream or sys.stderr
    parts: List[str] = []

    if ts:
        parts.append(datetime.now(timezone.utc).isoformat())

    if prefix:
        parts.append(prefix.rstrip())

    if json_mode:
        if isinstance(msg, str):
            payload = {"message": msg}
        elif isinstance(msg, Mapping):
            payload = dict(msg)
        else:
            payload = dict(msg)  # try to coerce iterable of pairs
        out = " ".join(parts + [json.dumps(payload, default=str, ensure_ascii=False)])
        stream.write(out + "\n")
        return

    if isinstance(msg, str):
        parts.append(msg)
    elif isinstance(msg, Mapping):
        kv = " ".join(f"{k}={json.dumps(v, default=str, ensure_ascii=False)}" for k, v in msg.items())
        parts.append(kv)
    else:
        kv = " ".join(f"{k}={json.dumps(v, default=str, ensure_ascii=False)}" for k, v in msg)
        parts.append(kv)

    stream.write(" ".join(parts) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CURRENCY_PREFIX = tuple("$€£¥₿₽₹₩₫₪₴₦₲₵₡₱₺₸")  # basic strip-only handling


def parse_boolean(value: Any) -> bool:
    """Best-effort boolean parser with multilingual variants using Tukuy."""
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Cannot parse None as boolean")
    s = str(value).strip().lower()
    return TUKUY.transform(s, ["bool"])

def as_list(value: Any, *, sep: str | None = None) -> List[Any]:
    """
    Convert a value into a list.
    - Lists/tuples: returned as list.
    - None: returns [].
    - String: split by sep (default: comma/semicolon/pipe).
    - Other scalars: [value].
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        if sep is None:
            # Split on comma, semicolon, or pipe
            parts = re.split(r"[,\|;]", value)
        else:
            parts = value.split(sep)
        return [p.strip() for p in parts if p.strip() != ""]
    return [value]

def parse_datetime(
    value: Any,
    *,
    dayfirst: bool = False,
    yearfirst: bool = False,
    default_tz: timezone | None = None,
) -> datetime:
    """
    Parse many common datetime strings into a timezone-aware datetime when possible.
    If the parsed datetime is naive and default_tz is provided, attach default_tz.
    """
    if isinstance(value, datetime):
        dt = value
    else:
        dt = dateutil.parser.parse(str(value), dayfirst=dayfirst, yearfirst=yearfirst)

    if dt.tzinfo is None and default_tz is not None:
        dt = dt.replace(tzinfo=default_tz)
    return dt


def _strip_currency_prefix(s: str) -> str:
    return s[1:].lstrip() if s and s[0] in _CURRENCY_PREFIX else s

def parse_shorthand_number(
    value: Any,
    *,
    allow_currency: bool = True,
    allow_percent: bool = True,
    percent_base: float = 1.0,
    as_decimal: bool | None = None,
) -> Union[int, float, Decimal]:
    """
    Parse a number possibly containing:
    - currency prefix: $1,200
    - separators: 1_200 or 1,200
    - scientific notation: 1e3
    - suffix multipliers: k, m, b, t, bn, mm, tr
    - percentages: '12%' (multiplies by percent_base)

    Args:
        value: str/number to parse
        allow_currency: if True, strip a single leading currency symbol
        allow_percent: if True, recognize trailing %
        percent_base: base used for % -> fraction (1.0 => 12% == 0.12)
        as_decimal: force Decimal output (True) or float/int (False). If None, infer.

    Returns:
        int, float, or Decimal

    Raises:
        ValueError for invalid format.
    """
    if value is None:
        raise ValueError("Cannot parse None as number")

    if isinstance(value, (int, float, Decimal)):
        return value

    s = str(value).strip()
    if not s:
        raise ValueError("Empty string")

    if allow_currency:
        s = _strip_currency_prefix(s)

    # Handle percent before Tukuy transform
    is_percent = False
    if allow_percent and s.endswith("%"):
        is_percent = True
        s = s[:-1].strip()

    # Use appropriate Tukuy transformer based on as_decimal
    transformer = ["shorthand_decimal"] if as_decimal else ["shorthand_number"]
    num = TUKUY.transform(s, transformer)
    
    # Handle percent if needed
    if is_percent:
        if isinstance(num, (int, float)):
            num = num * percent_base / 100
        else:  # Decimal
            num = num * Decimal(str(percent_base)) / Decimal('100')
    
    return num


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def _base_schema_for_type(field_name: str, field_type: Type[Any]) -> Dict[str, Any]:
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Optional / Union
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        nullable = len(non_none) < len(args)
        # Prefer single non-none; otherwise treat as "anyOf"
        if len(non_none) == 1:
            schema = _base_schema_for_type(field_name, non_none[0])
            if nullable:
                schema["nullable"] = True
            return schema
        return {
            "anyOf": [_base_schema_for_type(field_name, a) for a in non_none],
            "nullable": nullable or None,
        }

    # Containers
    if origin in (list, List):
        item_t = args[0] if args else Any
        return {
            "type": "array",
            "items": _strip_desc(_base_schema_for_type(f"{field_name}_item", item_t)),
        }

    if origin in (tuple, Tuple):
        # Treat as array with items; if variable length, use first type as items
        if args and args[-1] is Ellipsis:
            item_t = args[0]
            return {
                "type": "array",
                "items": _strip_desc(_base_schema_for_type(f"{field_name}_item", item_t)),
            }
        elif args:
            return {
                "type": "array",
                "prefixItems": [
                    _strip_desc(_base_schema_for_type(f"{field_name}_{i}", t))
                    for i, t in enumerate(args)
                ],
                "items": False,
            }
        return {"type": "array"}

    if origin in (dict, Dict):
        key_t = args[0] if args else str
        val_t = args[1] if len(args) > 1 else Any
        # JSON Schema keys must be strings; if not, we'll still describe "object"
        if key_t in (str, Any):
            return {
                "type": "object",
                "additionalProperties": _strip_desc(_base_schema_for_type(f"{field_name}_value", val_t)),
            }
        return {"type": "object"}  # fallback

    # Scalars / knowns
    if field_type in (int,):
        return {"type": "integer"}
    if field_type in (float, Decimal):
        return {"type": "number"}
    if field_type is bool:
        return {"type": "boolean"}
    if field_type is str:
        return {"type": "string"}
    if field_type is datetime:
        return {"type": "string", "format": "date-time"}
    if field_type is date:
        return {"type": "string", "format": "date"}
    if field_type is time:
        return {"type": "string", "format": "time"}
    if field_type is uuid.UUID:
        return {"type": "string", "format": "uuid"}

    # Pydantic models
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        # Reference model by title to avoid full inline (keeps schema lighter)
        return {"$ref": f"#/components/schemas/{field_type.__name__}"}

    # Custom types with __schema__
    if hasattr(field_type, "__schema__"):
        sch = getattr(field_type, "__schema__")
        if isinstance(sch, Mapping):
            return dict(sch)

    # Fallback
    return {"type": "string"}


def _strip_desc(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'description' if present (useful when embedding item schemas)."""
    schema = dict(schema)
    schema.pop("description", None)
    return schema


def create_field_schema(
    field_name: str,
    field_type: Type[Any],
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Creates a JSON(-like) schema for a field based on its type.

    - Handles Optional/Union
    - Infers formats for datetime/date/time/uuid
    - Supports list/tuple/dict containers
    - Supports custom types exposing __schema__
    - For string fields named like dates, add date-time format hint
    """
    schema = _base_schema_for_type(field_name, field_type)
    schema["description"] = description or f"Extract the {field_name} from the text."

    # If string but name suggests datetime
    if (
        schema.get("type") == "string"
        and any(term in field_name.lower() for term in ("date", "time", "when", "timestamp", "datetime"))
        and "format" not in schema
    ):
        schema["format"] = "date-time"

    return {k: v for k, v in schema.items() if v is not None}


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _to_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except InvalidOperation as e:
        raise ValueError(f"Cannot convert '{value}' to Decimal: {e}") from e


def convert_value(
    value: Any,
    target_type: Type[Any],
    allow_shorthand: bool = True,
) -> Any:
    """
    Convert 'value' to 'target_type' with support for:
    - Optional/Union
    - Numeric shorthand (1.2k, $3,400, 12%)
    - Booleans ("yes"/"no"/"on"/"off"/"si"/"sí")
    - Datetime parsing (dateutil)
    - Lists from comma/semicolon/pipe strings

    Notes:
    - For List[T], a scalar becomes [T(scalar)]
    - For Decimal and floats, shorthand and currency are supported
    """
    if value is None:
        return None

    origin = get_origin(target_type)
    args = get_args(target_type)

    # Optional / Union
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        # try each non-none type until one works
        last_error: Optional[Exception] = None
        for t in non_none:
            try:
                return convert_value(value, t, allow_shorthand=allow_shorthand)
            except Exception as e:
                last_error = e
        # if all failed, raise last error
        raise ValueError(f"Cannot convert '{value}' to any of {non_none}: {last_error}")

    # Lists / Tuples
    if origin in (list, List):
        item_t = args[0] if args else Any
        items = as_list(value)
        return [convert_value(v, item_t, allow_shorthand=allow_shorthand) for v in items]

    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            value = [value]
        if args and args[-1] is Ellipsis:
            item_t = args[0]
            return tuple(convert_value(v, item_t, allow_shorthand=allow_shorthand) for v in value)
        elif args:
            if len(value) != len(args):
                raise ValueError(f"Expected tuple of len {len(args)}, got {len(value)}")
            return tuple(
                convert_value(v, t, allow_shorthand=allow_shorthand) for v, t in zip(value, args)
            )
        return tuple(value)

    # Dict
    if origin in (dict, Dict):
        key_t = args[0] if args else str
        val_t = args[1] if len(args) > 1 else Any
        if not isinstance(value, Mapping):
            raise ValueError(f"Cannot convert non-mapping '{value}' to dict")
        return {
            convert_value(k, key_t, allow_shorthand=allow_shorthand):
                convert_value(v, val_t, allow_shorthand=allow_shorthand)
            for k, v in value.items()
        }

    # Scalars
    # Numbers
    if target_type is int:
        if allow_shorthand:
            parsed = parse_shorthand_number(value, as_decimal=False)
            return int(parsed)
        try:
            return int(value)
        except Exception:
            # try float/Decimal path
            return int(parse_shorthand_number(value, as_decimal=False, allow_percent=False))

    if target_type is float:
        if allow_shorthand:
            parsed = parse_shorthand_number(value, as_decimal=False)
            return float(parsed)
        return float(value)

    if target_type is Decimal:
        if allow_shorthand:
            parsed = parse_shorthand_number(value, as_decimal=True)
            return _to_decimal(parsed)
        return _to_decimal(value)

    # Bool
    if target_type is bool:
        return parse_boolean(value)

    # Strings
    if target_type is str:
        return "" if value is None else str(value)

    # Datetime / Date / Time
    if target_type is datetime:
        return parse_datetime(value)
    if target_type is date:
        dt = parse_datetime(value)
        return dt.date()
    if target_type is time:
        dt = parse_datetime(value)
        return dt.time()

    # UUID
    if target_type is uuid.UUID:
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(str(value))

    # Pydantic models
    if isinstance(target_type, type) and issubclass(target_type, BaseModel):
        if isinstance(value, target_type):
            return value
        if isinstance(value, Mapping):
            return target_type(**value)
        raise ValueError(f"Cannot convert non-mapping '{value}' to {target_type.__name__}")

    # Fallback: direct cast if possible
    try:
        return target_type(value)  # type: ignore[call-arg]
    except Exception as e:
        raise ValueError(f"Cannot convert '{value}' to {getattr(target_type, '__name__', target_type)}: {e}") from e


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_fields(
    model_cls: Type[BaseModel],
    data: Dict[str, Any],
    fields: Optional[List[str]] = None,
    *,
    strict: bool = True,
    missing: str = "skip",  # "skip" | "none" | "error"
    level: int | LogLevel = LogLevel.OFF,
) -> Dict[str, Any]:
    """
    Extract and convert only specified fields based on a Pydantic model.

    Args:
        model_cls: Pydantic model class.
        data: Source mapping.
        fields: If None, all model fields are considered.
        strict: If True, unknown keys in 'fields' raise KeyError.
        missing: What to do when a requested field isn't in 'data':
                 - "skip": drop it
                 - "none": include with None
                 - "error": raise KeyError
        level: LogLevel for internal debug logs (uses log_debug).

    Returns:
        Dict of converted values suitable for instantiating the model.
    """
    model_fields = model_cls.model_fields
    valid_fields = set(model_fields.keys())

    if fields is None:
        fields = list(valid_fields)

    # Validate requested fields
    req = set(fields)
    invalid = req - valid_fields
    if strict and invalid:
        raise KeyError(f"Fields not found in model: {', '.join(sorted(invalid))}")

    result: Dict[str, Any] = {}

    for fname in fields:
        if fname not in valid_fields:
            # silently ignore if not strict
            continue

        finfo = model_fields[fname]
        source_key = finfo.alias or fname

        if source_key not in data:
            if missing == "skip":
                log_debug(LogLevel.DEBUG, level, {"skip_missing": fname})
                continue
            if missing == "none":
                result[fname] = None
                log_debug(LogLevel.DEBUG, level, {"missing_none": fname})
                continue
            raise KeyError(f"Missing required field in data: {source_key}")

        raw = data[source_key]
        try:
            converted = convert_value(raw, finfo.annotation, allow_shorthand=True)
            result[fname] = converted
            log_debug(LogLevel.TRACE, level, {"field": fname, "raw": raw, "converted": converted})
        except Exception as e:
            raise ValueError(f"Validation failed for field '{fname}': {e}") from e

    return result


# ---------------------------------------------------------------------------
# JSON text cleaning
# ---------------------------------------------------------------------------

def clean_json_text(text: str) -> str:
    """Attempts to extract a valid JSON object string from text.

    Handles multiple possible formatting issues:
    - Removes <think>...</think> blocks.
    - Strips markdown code fences (```json ... ```).
    - Falls back to first {...} block found.

    Args:
        text: Raw string that may contain JSON plus extra formatting.

    Returns:
        A string that best resembles valid JSON content.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    if text.startswith("```"):
        start_fence = text.find("```")
        if start_fence != -1:
            start_content = text.find("\n", start_fence)
            if start_content != -1:
                end_fence = text.find("```", start_content)
                if end_fence != -1:
                    return text[start_content + 1:end_fence].strip()
                else:
                    return text[start_content + 1 :].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text