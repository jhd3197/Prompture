"""Infrastructure: settings, logging, callbacks, caching, costs, discovery."""

from .budget import (
    BudgetPolicy,
    BudgetState,
    CostEstimate,
    enforce_budget,
    estimate_call_cost,
    estimate_cost,
    estimate_tokens,
    resolve_budget_policy,
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
from .capabilities import (
    ProviderCapabilities,
    clear_overrides,
    get_capabilities,
    get_compatibility_matrix,
    override_capabilities,
    register_model,
    register_provider,
)
from .coding_agent_events import (
    CodingAgentEvent,
    detect_question,
    parse_claude_stream_json_lines,
    parse_codex_json_lines,
)
from .coding_agent_specs import (
    CODING_AGENT_SPECS,
    CodingAgentSpec,
)
from .coding_agent_specs import (
    get_spec as get_coding_agent_spec,
)
from .coding_agent_specs import (
    supported_agent_ids as supported_coding_agent_ids,
)
from .coding_agents import (
    ApprovalMode,
    CodingAgentCommand,
    CodingAgentRunResult,
    arun_coding_agent,
    astream_coding_agent,
    build_coding_agent_command,
    run_coding_agent,
)
from .cost_mixin import AudioCostMixin, EmbeddingCostMixin, VideoCostMixin
from .discovery import (
    CodingAgentExecutable,
    CodingAgentInfo,
    clear_discovery_cache,
    display_available_models,
    get_available_audio_models,
    get_available_coding_agents,
    get_available_embedding_models,
    get_available_image_gen_models,
    get_available_lipsync_models,
    get_available_models,
    get_available_moderation_models,
    get_available_music_models,
    get_available_rerank_models,
    get_available_video_gen_models,
    get_coding_agent_executable_candidates,
    pick_best_coding_agent,
    resolve_coding_agent_binary,
    resolve_coding_agent_executable,
    verify_coding_agent_binary,
    verify_coding_agent_executable,
)
from .ledger import ModelUsageLedger, get_recently_used_models
from .logging import JSONFormatter, configure_logging
from .model_rates import (
    ModelCapabilities,
    get_model_capabilities,
    get_model_info,
    get_model_lifecycle,
    get_model_rates,
    refresh_rates_cache,
)
from .provider_env import ProviderEnvironment

try:
    from .tukuy_backend import TukuyLLMBackend, create_tukuy_backend
except ImportError:  # tukuy not installed
    TukuyLLMBackend = None  # type: ignore[assignment,misc]
    create_tukuy_backend = None  # type: ignore[assignment]
from .session import UsageSession
from .settings import settings
from .tracker import UsageEvent, UsageSink, UsageTracker, configure_tracker, get_tracker

__all__ = [
    "CODING_AGENT_SPECS",
    "ApprovalMode",
    "AudioCostMixin",
    "BudgetPolicy",
    "BudgetState",
    "CacheBackend",
    "CodingAgentCommand",
    "CodingAgentEvent",
    "CodingAgentExecutable",
    "CodingAgentInfo",
    "CodingAgentRunResult",
    "CodingAgentSpec",
    "CostEstimate",
    "DriverCallbacks",
    "EmbeddingCostMixin",
    "JSONFormatter",
    "MemoryCacheBackend",
    "ModelCapabilities",
    "ModelUsageLedger",
    "ProviderCapabilities",
    "ProviderEnvironment",
    "RedisCacheBackend",
    "ResponseCache",
    "SQLiteCacheBackend",
    "TukuyLLMBackend",
    "UsageEvent",
    "UsageSession",
    "UsageSink",
    "UsageTracker",
    "VideoCostMixin",
    "arun_coding_agent",
    "astream_coding_agent",
    "build_coding_agent_command",
    "clear_discovery_cache",
    "clear_overrides",
    "configure_cache",
    "configure_logging",
    "configure_tracker",
    "create_tukuy_backend",
    "detect_question",
    "display_available_models",
    "enforce_budget",
    "estimate_call_cost",
    "estimate_cost",
    "estimate_tokens",
    "get_available_audio_models",
    "get_available_coding_agents",
    "get_available_embedding_models",
    "get_available_image_gen_models",
    "get_available_lipsync_models",
    "get_available_models",
    "get_available_moderation_models",
    "get_available_music_models",
    "get_available_rerank_models",
    "get_available_video_gen_models",
    "get_cache",
    "get_capabilities",
    "get_coding_agent_executable_candidates",
    "get_coding_agent_spec",
    "get_compatibility_matrix",
    "get_model_capabilities",
    "get_model_info",
    "get_model_lifecycle",
    "get_model_rates",
    "get_recently_used_models",
    "get_tracker",
    "override_capabilities",
    "parse_claude_stream_json_lines",
    "parse_codex_json_lines",
    "pick_best_coding_agent",
    "refresh_rates_cache",
    "register_model",
    "register_provider",
    "resolve_budget_policy",
    "resolve_coding_agent_binary",
    "resolve_coding_agent_executable",
    "run_coding_agent",
    "settings",
    "supported_coding_agent_ids",
    "verify_coding_agent_binary",
    "verify_coding_agent_executable",
]
