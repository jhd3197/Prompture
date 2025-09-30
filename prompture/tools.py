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
    "load_field_definitions",
    "validate_field_definition",
    "get_type_default",
    "get_field_default",
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
    """
    Enhanced boolean parser with multilingual variants and edge case handling.
    
    Supports:
    - Standard: true/false, yes/no, on/off, 1/0
    - Multilingual: si/no (Spanish), oui/non (French), ja/nein (German)
    - Edge cases: empty strings, whitespace, case variations
    
    Uses Tukuy transformer for robust multilingual support.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("Cannot parse None as boolean")
    
    # Handle numeric values
    if isinstance(value, (int, float, Decimal)):
        return bool(value)
    
    s = str(value).strip().lower()
    
    # Handle empty strings and common "falsy" representations
    if not s or s in ("", "null", "none", "n/a", "na", "nil", "undefined"):
        return False
    
    # Quick check for obvious true/false values before using Tukuy
    if s in ("1", "true", "yes", "on", "si", "sí", "oui", "ja", "t", "y"):
        return True
    if s in ("0", "false", "no", "off", "non", "nein", "f", "n"):
        return False
    
    # Use Tukuy for more complex cases
    try:
        return TUKUY.transform(s, ["bool"])
    except Exception:
        # Fallback for unrecognized values - try to be reasonable
        # If it looks like a number, convert to bool
        try:
            num_val = float(s)
            return bool(num_val)
        except (ValueError, TypeError):
            pass
        
        # If it contains "true", "yes", "on", etc., lean towards True
        true_indicators = ["true", "yes", "on", "enable", "active", "si", "oui", "ja"]
        false_indicators = ["false", "no", "off", "disable", "inactive", "non", "nein"]
        
        s_lower = s.lower()
        if any(indicator in s_lower for indicator in true_indicators):
            return True
        if any(indicator in s_lower for indicator in false_indicators):
            return False
        
        # Final fallback - raise error for truly ambiguous cases
        raise ValueError(f"Cannot parse '{value}' as boolean")

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
    field_name: Optional[str] = None,
    field_definitions: Optional[Dict[str, Any]] = None,
    use_defaults_on_failure: bool = True,
) -> Any:
    """
    Enhanced value converter with robust error handling and default value support.
    
    Convert 'value' to 'target_type' with support for:
    - Optional/Union with intelligent type ordering
    - Numeric shorthand (1.2k, $3,400, 12%) with better error handling
    - Enhanced booleans with multilingual support
    - Datetime parsing with graceful failures
    - Lists from comma/semicolon/pipe strings with item conversion fallbacks
    - Default value fallback system when conversion fails

    Args:
        value: Value to convert
        target_type: Target type for conversion
        allow_shorthand: Enable shorthand number parsing (1.2k, $100, etc.)
        field_name: Name of field being converted (for field-specific defaults)
        field_definitions: Field definitions dict for custom defaults/instructions
        use_defaults_on_failure: Whether to use defaults when conversion fails
        
    Returns:
        Converted value or appropriate default
        
    Notes:
    - For List[T], a scalar becomes [T(scalar)]
    - For Decimal and floats, shorthand and currency are supported
    - Conversion failures log warnings but continue with defaults if enabled
    - Union types try conversions in order and use first successful result
    """
    
    def _get_fallback_value(error_msg: str = "") -> Any:
        """Get appropriate fallback value when conversion fails."""
        if not use_defaults_on_failure:
            raise ValueError(error_msg)
        
        try:
            # Try field-specific default first
            if field_name and field_definitions:
                field_def = field_definitions.get(field_name, {})
                if isinstance(field_def, dict) and 'default' in field_def:
                    log_debug(LogLevel.DEBUG, LogLevel.INFO,
                             f"Using field default for '{field_name}': {field_def['default']}")
                    return field_def['default']
            
            # Fall back to type default
            type_default = get_type_default(target_type)
            log_debug(LogLevel.DEBUG, LogLevel.INFO,
                     f"Using type default for {target_type}: {type_default}")
            return type_default
            
        except Exception as fallback_error:
            log_debug(LogLevel.WARN, LogLevel.INFO,
                     f"Failed to get fallback for {target_type}: {fallback_error}")
            return None

    def _safe_convert_recursive(val: Any, typ: Type[Any]) -> Any:
        """Recursively convert with same parameters but no fallback to avoid infinite recursion."""
        return convert_value(
            val, typ,
            allow_shorthand=allow_shorthand,
            field_name=field_name,
            field_definitions=field_definitions,
            use_defaults_on_failure=False  # Avoid recursion in fallbacks
        )

    # Handle None values early
    if value is None:
        origin = get_origin(target_type)
        args = get_args(target_type)
        
        # Check if target type is Optional (Union with None)
        if origin is Union and type(None) in args:
            return None
        
        # For non-optional types, use fallback
        return _get_fallback_value("Cannot convert None to non-optional type")

    origin = get_origin(target_type)
    args = get_args(target_type)

    # Optional / Union - Enhanced with better error recovery
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        is_optional = type(None) in args
        
        if value is None and is_optional:
            return None
        
        # Try each non-none type until one works
        conversion_errors = []
        
        for i, t in enumerate(non_none):
            try:
                result = _safe_convert_recursive(value, t)
                log_debug(LogLevel.TRACE, LogLevel.DEBUG,
                         f"Union conversion succeeded with type {t} for value '{value}'")
                return result
            except Exception as e:
                conversion_errors.append((t, str(e)))
                log_debug(LogLevel.TRACE, LogLevel.DEBUG,
                         f"Union conversion failed for type {t}: {e}")
        
        # All conversions failed
        error_msg = f"Cannot convert '{value}' to any Union type {non_none}. Errors: {conversion_errors}"
        log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
        return _get_fallback_value(error_msg)

    # Lists / Tuples - Enhanced error handling for individual items
    if origin in (list, List):
        item_t = args[0] if args else Any
        try:
            items = as_list(value)
            result_items = []
            
            for i, item in enumerate(items):
                try:
                    converted_item = _safe_convert_recursive(item, item_t)
                    result_items.append(converted_item)
                except Exception as e:
                    log_debug(LogLevel.WARN, LogLevel.INFO,
                             f"Failed to convert list item {i} '{item}' to {item_t}: {e}")
                    # Try to get default for item type
                    try:
                        default_item = get_type_default(item_t)
                        result_items.append(default_item)
                    except Exception:
                        # Skip item if we can't get a default
                        continue
            
            return result_items
            
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to list: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    if origin in (tuple, Tuple):
        try:
            if not isinstance(value, (list, tuple)):
                value = [value]
            if args and args[-1] is Ellipsis:
                item_t = args[0]
                converted_items = []
                for item in value:
                    try:
                        converted_items.append(_safe_convert_recursive(item, item_t))
                    except Exception as e:
                        log_debug(LogLevel.WARN, LogLevel.INFO,
                                 f"Failed to convert tuple item '{item}': {e}")
                        converted_items.append(get_type_default(item_t))
                return tuple(converted_items)
            elif args:
                if len(value) != len(args):
                    raise ValueError(f"Expected tuple of len {len(args)}, got {len(value)}")
                converted_items = []
                for v, t in zip(value, args):
                    try:
                        converted_items.append(_safe_convert_recursive(v, t))
                    except Exception as e:
                        log_debug(LogLevel.WARN, LogLevel.INFO,
                                 f"Failed to convert tuple item '{v}' to {t}: {e}")
                        converted_items.append(get_type_default(t))
                return tuple(converted_items)
            return tuple(value)
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to tuple: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    # Dict - Enhanced error handling
    if origin in (dict, Dict):
        key_t = args[0] if args else str
        val_t = args[1] if len(args) > 1 else Any
        
        try:
            if not isinstance(value, Mapping):
                raise ValueError(f"Cannot convert non-mapping '{value}' to dict")
            
            result_dict = {}
            for k, v in value.items():
                try:
                    converted_key = _safe_convert_recursive(k, key_t)
                    converted_val = _safe_convert_recursive(v, val_t)
                    result_dict[converted_key] = converted_val
                except Exception as e:
                    log_debug(LogLevel.WARN, LogLevel.INFO,
                             f"Failed to convert dict item {k}:{v}: {e}")
                    # Skip problematic items
                    continue
            
            return result_dict
            
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to dict: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    # Scalars with enhanced error handling
    
    # Numbers - More robust handling
    if target_type is int:
        try:
            # Handle common edge cases first
            if isinstance(value, str):
                s = value.strip()
                if not s or s.lower() in ("", "null", "none", "n/a", "na"):
                    return _get_fallback_value(f"Empty/null string cannot be converted to int")
            
            if allow_shorthand:
                parsed = parse_shorthand_number(value, as_decimal=False)
                return int(parsed)
            else:
                return int(value)
                
        except Exception as e:
            # Try alternative parsing
            try:
                return int(parse_shorthand_number(value, as_decimal=False, allow_percent=False))
            except Exception:
                error_msg = f"Cannot convert '{value}' to int: {e}"
                log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
                return _get_fallback_value(error_msg)

    if target_type is float:
        try:
            # Handle edge cases
            if isinstance(value, str):
                s = value.strip()
                if not s or s.lower() in ("", "null", "none", "n/a", "na"):
                    return _get_fallback_value(f"Empty/null string cannot be converted to float")
            
            if allow_shorthand:
                parsed = parse_shorthand_number(value, as_decimal=False)
                return float(parsed)
            else:
                return float(value)
                
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to float: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    if target_type is Decimal:
        try:
            # Handle edge cases
            if isinstance(value, str):
                s = value.strip()
                if not s or s.lower() in ("", "null", "none", "n/a", "na"):
                    return _get_fallback_value(f"Empty/null string cannot be converted to Decimal")
            
            if allow_shorthand:
                parsed = parse_shorthand_number(value, as_decimal=True)
                return _to_decimal(parsed)
            else:
                return _to_decimal(value)
                
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to Decimal: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    # Bool - Enhanced error handling
    if target_type is bool:
        try:
            return parse_boolean(value)
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to bool: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    # Strings - More robust handling
    if target_type is str:
        try:
            if value is None:
                return ""  # Standard behavior
            return str(value)
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to str: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    # Datetime / Date / Time - Enhanced error handling
    if target_type is datetime:
        try:
            return parse_datetime(value)
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to datetime: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)
            
    if target_type is date:
        try:
            dt = parse_datetime(value)
            return dt.date()
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to date: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)
            
    if target_type is time:
        try:
            dt = parse_datetime(value)
            return dt.time()
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to time: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    # UUID - Enhanced error handling
    if target_type is uuid.UUID:
        try:
            if isinstance(value, uuid.UUID):
                return value
            return uuid.UUID(str(value))
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to UUID: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    # Pydantic models - Enhanced error handling
    if isinstance(target_type, type) and issubclass(target_type, BaseModel):
        try:
            if isinstance(value, target_type):
                return value
            if isinstance(value, Mapping):
                return target_type(**value)
            else:
                raise ValueError(f"Cannot convert non-mapping '{value}' to {target_type.__name__}")
        except Exception as e:
            error_msg = f"Cannot convert '{value}' to {target_type.__name__}: {e}"
            log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
            return _get_fallback_value(error_msg)

    # Fallback: direct cast if possible
    try:
        return target_type(value)  # type: ignore[call-arg]
    except Exception as e:
        error_msg = f"Cannot convert '{value}' to {getattr(target_type, '__name__', target_type)}: {e}"
        log_debug(LogLevel.WARN, LogLevel.INFO, error_msg)
        return _get_fallback_value(error_msg)


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
    field_definitions: Optional[Dict[str, Any]] = None,
    use_defaults_on_failure: bool = True,
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
        field_definitions: Optional field definitions for default values and conversion hints.
        use_defaults_on_failure: Whether to use default values when conversion fails.

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
            converted = convert_value(
                raw,
                finfo.annotation,
                allow_shorthand=True,
                field_name=fname,
                field_definitions=field_definitions,
                use_defaults_on_failure=use_defaults_on_failure
            )
            result[fname] = converted
            log_debug(LogLevel.TRACE, level, {"field": fname, "raw": raw, "converted": converted})
        except Exception as e:
            # If we're not using defaults, re-raise the original error
            if not use_defaults_on_failure:
                raise ValueError(f"Validation failed for field '{fname}': {e}") from e
            
            # Try to get a fallback value using the field info
            try:
                fallback_value = get_field_default(fname, finfo, field_definitions)
                result[fname] = fallback_value
                log_debug(LogLevel.WARN, level, {
                    "field": fname,
                    "error": str(e),
                    "fallback": fallback_value
                })
            except Exception as fallback_error:
                # If even fallback fails, re-raise original error
                log_debug(LogLevel.ERROR, level, {
                    "field": fname,
                    "conversion_error": str(e),
                    "fallback_error": str(fallback_error)
                })
                raise ValueError(f"Validation failed for field '{fname}': {e}") from e

    return result


# ---------------------------------------------------------------------------
# Field Definitions
# ---------------------------------------------------------------------------

def load_field_definitions(path: str) -> Dict[str, Any]:
    """
    Load field definitions from a JSON or YAML file.

    Args:
        path: Path to the JSON or YAML file containing field definitions

    Returns:
        Dict[str, Any]: Loaded field definitions

    Raises:
        ValueError: If file format is not supported or content is invalid
        FileNotFoundError: If the file doesn't exist
    """
    import yaml
    from pathlib import Path

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Field definitions file not found: {path}")

    suffix = path_obj.suffix.lower()
    content = path_obj.read_text(encoding='utf-8')

    try:
        if suffix == '.json':
            return json.loads(content)
        elif suffix in ('.yaml', '.yml'):
            return yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    except Exception as e:
        raise ValueError(f"Failed to parse field definitions: {e}") from e

def validate_field_definition(definition: Dict[str, Any]) -> bool:
    """
    Validate a field definition structure.

    Args:
        definition: Field definition dictionary to validate

    Returns:
        bool: True if valid, False otherwise

    Required keys:
    - type: The field's data type
    - description: Human-readable description
    - instructions: Extraction instructions
    - default: Default value
    - nullable: Whether field can be None
    """
    required_keys = {'type', 'description', 'instructions', 'default', 'nullable'}
    
    # Check for required keys
    if not all(key in definition for key in required_keys):
        return False
        
    # Validate type
    if not isinstance(definition['type'], (type, str)):
        return False
        
    # Validate description and instructions are strings
    if not all(isinstance(definition[k], str) for k in ['description', 'instructions']):
        return False
        
    # Validate nullable is boolean
    if not isinstance(definition['nullable'], bool):
        return False
        
    return True

# ---------------------------------------------------------------------------
# Default Value Handling
# ---------------------------------------------------------------------------

def get_type_default(field_type: Type[Any]) -> Any:
    """
    Get a sensible default value for a given type.
    
    Args:
        field_type: The type to get a default for
        
    Returns:
        A default value appropriate for the type
    """
    origin = get_origin(field_type)
    args = get_args(field_type)
    
    # Handle Optional/Union types
    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[T] -> get default for T
            return get_type_default(non_none[0])
        # Multiple non-None types -> return None
        return None
    
    # Container types
    if origin in (list, List) or field_type is list:
        return []
    if origin in (tuple, Tuple) or field_type is tuple:
        return ()
    if origin in (dict, Dict) or field_type is dict:
        return {}
    
    # Scalar types
    if field_type is int:
        return 0
    if field_type in (float, Decimal):
        return 0.0
    if field_type is bool:
        return False
    if field_type is str:
        return ""
    if field_type is datetime:
        return datetime.now()
    if field_type is date:
        return date.today()
    if field_type is time:
        return time(0, 0)
    if field_type is uuid.UUID:
        return uuid.uuid4()
    
    # Pydantic models - try to create empty instance
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        try:
            return field_type()
        except Exception:
            return None
    
    # Fallback
    return None


def get_field_default(field_name: str, field_info: Any, field_definitions: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get the default value for a field using the priority order:
    1. field_definitions default
    2. Pydantic field default
    3. Type-appropriate default
    
    Args:
        field_name: Name of the field
        field_info: Pydantic field info
        field_definitions: Optional field definitions dict
        
    Returns:
        The appropriate default value
    """
    # Priority 1: field_definitions
    if field_definitions and field_name in field_definitions:
        field_def = field_definitions[field_name]
        if isinstance(field_def, dict) and 'default' in field_def:
            return field_def['default']
    
    # Priority 2: Pydantic default - check for PydanticUndefined
    if hasattr(field_info, 'default'):
        default_val = field_info.default
        # Handle PydanticUndefined (newer Pydantic) and Ellipsis (older Pydantic)
        if default_val is not ... and str(default_val) != 'PydanticUndefined':
            return default_val
    
    # Priority 3: Type default
    return get_type_default(field_info.annotation)

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