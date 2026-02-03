"""prompture - API package to convert LLM outputs into JSON + test harness."""

from dotenv import load_dotenv

from .agent import Agent, AgentIterator, StreamedAgentResult
from .agent_types import (
    AgentCallbacks,
    AgentResult,
    AgentState,
    AgentStep,
    ApprovalRequired,
    GuardrailError,
    ModelRetry,
    RunContext,
    StepType,
    StreamEvent,
    StreamEventType,
)
from .analysis import (
    CodeAnalysis,
    CodeFeatures,
    FeatureExtractor,
    RiskAssessment,
    RiskLevel,
    analyze_python,
    calculate_risk,
)
from .async_agent import AsyncAgent, AsyncAgentIterator, AsyncStreamedAgentResult
from .async_conversation import AsyncConversation
from .async_driver import AsyncDriver
from .async_groups import (
    AsyncLoopGroup,
    AsyncRouterAgent,
    AsyncSequentialGroup,
    ParallelGroup,
)
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
    # Azure config API
    clear_azure_configs,
    get_driver,
    get_driver_for_model,
    # Plugin registration API
    is_async_driver_registered,
    is_driver_registered,
    list_registered_async_drivers,
    list_registered_drivers,
    load_entry_point_drivers,
    register_async_driver,
    register_azure_config,
    register_driver,
    set_azure_config_resolver,
    unregister_async_driver,
    unregister_azure_config,
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
from .group_types import (
    AgentError,
    ErrorPolicy,
    GroupCallbacks,
    GroupResult,
    GroupStep,
)
from .groups import (
    GroupAsAgent,
    LoopGroup,
    RouterAgent,
    SequentialGroup,
)
from .history import (
    calculate_cost_breakdown,
    export_result_json,
    filter_steps,
    get_tool_call_summary,
    result_to_dict,
    search_messages,
)
from .image import (
    ImageContent,
    ImageInput,
    image_from_base64,
    image_from_bytes,
    image_from_file,
    image_from_url,
    make_image,
)
from .ledger import ModelUsageLedger, get_recently_used_models
from .logging import JSONFormatter, configure_logging
from .model_rates import (
    ModelCapabilities,
    get_model_capabilities,
    get_model_info,
    get_model_rates,
    refresh_rates_cache,
)
from .persistence import ConversationStore
from .persona import (
    PERSONAS,
    Persona,
    clear_persona_registry,
    get_persona,
    get_persona_names,
    get_persona_registry_snapshot,
    get_trait,
    get_trait_names,
    load_personas_from_directory,
    register_persona,
    register_trait,
    reset_persona_registry,
    reset_trait_registry,
)
from .runner import run_suite_from_spec
from .sandbox import (
    ALWAYS_BLOCKED_IMPORTS,
    ImportRestrictions,
    ImportViolationError,
    PathRestrictions,
    PathViolationError,
    PythonSandbox,
    ResourceContext,
    ResourceLimitError,
    ResourceLimits,
    SandboxError,
    SandboxResult,
    SandboxTimeoutError,
)
from .serialization import (
    EXPORT_VERSION,
    export_conversation,
    export_usage_session,
    import_conversation,
    import_usage_session,
)
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
    # Sandbox module
    "ALWAYS_BLOCKED_IMPORTS",
    # Core exports
    "EXPORT_VERSION",
    "FIELD_DEFINITIONS",
    "PERSONAS",
    "Agent",
    "AgentCallbacks",
    "AgentError",
    "AgentIterator",
    "AgentResult",
    "AgentState",
    "AgentStep",
    "AirLLMDriver",
    "ApprovalRequired",
    "AsyncAgent",
    "AsyncAgentIterator",
    "AsyncConversation",
    "AsyncDriver",
    "AsyncLoopGroup",
    "AsyncRouterAgent",
    "AsyncSequentialGroup",
    "AsyncStreamedAgentResult",
    "AzureDriver",
    "CacheBackend",
    "ClaudeDriver",
    # Analysis module
    "CodeAnalysis",
    "CodeFeatures",
    "Conversation",
    "ConversationStore",
    "Driver",
    "DriverCallbacks",
    "ErrorPolicy",
    "FeatureExtractor",
    "GoogleDriver",
    "GrokDriver",
    "GroqDriver",
    "GroupAsAgent",
    "GroupCallbacks",
    "GroupResult",
    "GroupStep",
    "GuardrailError",
    "ImageContent",
    "ImageInput",
    "ImportRestrictions",
    "ImportViolationError",
    "JSONFormatter",
    "LMStudioDriver",
    "LocalHTTPDriver",
    "LoopGroup",
    "MemoryCacheBackend",
    "ModelCapabilities",
    "ModelRetry",
    "ModelUsageLedger",
    "OllamaDriver",
    "OpenAIDriver",
    "OpenRouterDriver",
    "ParallelGroup",
    "PathRestrictions",
    "PathViolationError",
    "Persona",
    "PythonSandbox",
    "RedisCacheBackend",
    "ResourceContext",
    "ResourceLimitError",
    "ResourceLimits",
    "ResponseCache",
    "RiskAssessment",
    "RiskLevel",
    "RouterAgent",
    "RunContext",
    "SQLiteCacheBackend",
    "SandboxError",
    "SandboxResult",
    "SandboxTimeoutError",
    "SequentialGroup",
    "StepType",
    "StreamEvent",
    "StreamEventType",
    "StreamedAgentResult",
    "ToolDefinition",
    "ToolRegistry",
    "UsageSession",
    "add_field_definition",
    "add_field_definitions",
    "analyze_python",
    "ask_for_json",
    # History module
    "calculate_cost_breakdown",
    "calculate_risk",
    "clean_json_text",
    "clean_json_text_with_ai",
    "clean_toon_text",
    "clear_azure_configs",
    "clear_persona_registry",
    "clear_registry",
    "configure_cache",
    "configure_logging",
    "export_conversation",
    "export_result_json",
    "export_usage_session",
    "extract_and_jsonify",
    "extract_from_data",
    "extract_from_pandas",
    "extract_with_model",
    "field_from_registry",
    "filter_steps",
    "get_available_models",
    "get_cache",
    "get_driver",
    "get_driver_for_model",
    "get_field_definition",
    "get_field_names",
    "get_model_capabilities",
    "get_model_info",
    "get_model_rates",
    "get_persona",
    "get_persona_names",
    "get_persona_registry_snapshot",
    "get_recently_used_models",
    "get_registry_snapshot",
    "get_required_fields",
    "get_tool_call_summary",
    "get_trait",
    "get_trait_names",
    "image_from_base64",
    "image_from_bytes",
    "image_from_file",
    "image_from_url",
    "import_conversation",
    "import_usage_session",
    "is_async_driver_registered",
    "is_driver_registered",
    "list_registered_async_drivers",
    "list_registered_drivers",
    "load_entry_point_drivers",
    "load_personas_from_directory",
    "make_image",
    "manual_extract_and_jsonify",
    "normalize_enum_value",
    "refresh_rates_cache",
    "register_async_driver",
    "register_azure_config",
    "register_driver",
    "register_field",
    "register_persona",
    "register_trait",
    "render_output",
    "reset_persona_registry",
    "reset_registry",
    "reset_trait_registry",
    "result_to_dict",
    "run_suite_from_spec",
    "search_messages",
    "set_azure_config_resolver",
    "stepwise_extract_with_model",
    "tool_from_function",
    "unregister_async_driver",
    "unregister_azure_config",
    "unregister_driver",
    "validate_against_schema",
    "validate_enum_value",
]
