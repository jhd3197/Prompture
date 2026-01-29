"""prompture - API package to convert LLM outputs into JSON + test harness."""

from dotenv import load_dotenv

from .async_conversation import AsyncConversation
from .async_driver import AsyncDriver
from .cache import (
    CacheBackend,
    MemoryCacheBackend,
    RedisCacheBackend,
    ResponseCache,
    SQLiteCacheBackend,
    configure_cache,
    get_cache,
)
from .callbacks import DriverCallbacks
from .conversation import Conversation
from .core import (
    Driver,
    ask_for_json,
    clean_json_text_with_ai,
    extract_and_jsonify,
    extract_from_data,
    extract_from_pandas,
    extract_with_model,
    manual_extract_and_jsonify,
    render_output,
    stepwise_extract_with_model,
)
from .discovery import get_available_models
from .drivers import (
    AirLLMDriver,
    AzureDriver,
    ClaudeDriver,
    GoogleDriver,
    GrokDriver,
    GroqDriver,
    LMStudioDriver,
    LocalHTTPDriver,
    OllamaDriver,
    OpenAIDriver,
    OpenRouterDriver,
    get_driver,
    get_driver_for_model,
    # Plugin registration API
    is_async_driver_registered,
    is_driver_registered,
    list_registered_async_drivers,
    list_registered_drivers,
    load_entry_point_drivers,
    register_async_driver,
    register_driver,
    unregister_async_driver,
    unregister_driver,
)
from .field_definitions import (
    FIELD_DEFINITIONS,
    add_field_definition,
    add_field_definitions,
    clear_registry,
    field_from_registry,
    get_field_definition,
    get_field_names,
    get_registry_snapshot,
    get_required_fields,
    normalize_enum_value,
    register_field,
    reset_registry,
    validate_enum_value,
)
from .logging import JSONFormatter, configure_logging
from .model_rates import get_model_info, get_model_rates, refresh_rates_cache
from .runner import run_suite_from_spec
from .session import UsageSession
from .settings import settings as _settings
from .tools import clean_json_text, clean_toon_text
from .tools_schema import ToolDefinition, ToolRegistry, tool_from_function
from .validator import validate_against_schema

# Load environment variables from .env file
load_dotenv()

# Auto-configure cache from settings if enabled
if _settings.cache_enabled:
    configure_cache(
        backend=_settings.cache_backend,
        enabled=True,
        ttl=_settings.cache_ttl_seconds,
        maxsize=_settings.cache_memory_maxsize,
        db_path=_settings.cache_sqlite_path,
        redis_url=_settings.cache_redis_url,
    )

# runtime package version (from installed metadata)
try:
    # Python 3.8+
    from importlib.metadata import version as _get_version
except Exception:
    # older python using importlib-metadata backport (if you include it)
    from importlib_metadata import version as _get_version

try:
    __version__ = _get_version("prompture")
except Exception:
    # fallback during local editable development
    __version__ = "0.0.0"

__all__ = [
    "FIELD_DEFINITIONS",
    "AirLLMDriver",
    "AsyncConversation",
    "AsyncDriver",
    "AzureDriver",
    "CacheBackend",
    "ClaudeDriver",
    "Conversation",
    "Driver",
    "DriverCallbacks",
    "GoogleDriver",
    "GrokDriver",
    "GroqDriver",
    "JSONFormatter",
    "LMStudioDriver",
    "LocalHTTPDriver",
    "MemoryCacheBackend",
    "OllamaDriver",
    "OpenAIDriver",
    "OpenRouterDriver",
    "RedisCacheBackend",
    "ResponseCache",
    "SQLiteCacheBackend",
    "ToolDefinition",
    "ToolRegistry",
    "UsageSession",
    "add_field_definition",
    "add_field_definitions",
    "ask_for_json",
    "clean_json_text",
    "clean_json_text_with_ai",
    "clean_toon_text",
    "clear_registry",
    "configure_cache",
    "configure_logging",
    "extract_and_jsonify",
    "extract_from_data",
    "extract_from_pandas",
    "extract_with_model",
    "field_from_registry",
    "get_available_models",
    "get_cache",
    "get_driver",
    "get_driver_for_model",
    "get_field_definition",
    "get_field_names",
    "get_model_info",
    "get_model_rates",
    "get_registry_snapshot",
    "get_required_fields",
    # Plugin registration API
    "is_async_driver_registered",
    "is_driver_registered",
    "list_registered_async_drivers",
    "list_registered_drivers",
    "load_entry_point_drivers",
    # Other exports
    "manual_extract_and_jsonify",
    "normalize_enum_value",
    "refresh_rates_cache",
    "register_async_driver",
    "register_driver",
    "register_field",
    "render_output",
    "reset_registry",
    "run_suite_from_spec",
    "stepwise_extract_with_model",
    "tool_from_function",
    "unregister_async_driver",
    "unregister_driver",
    "validate_against_schema",
    "validate_enum_value",
]
