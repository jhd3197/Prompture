"""Infrastructure: settings, logging, callbacks, caching, costs, discovery."""

from .budget import (
    BudgetPolicy,
    BudgetState,
    enforce_budget,
    estimate_cost,
    estimate_tokens,
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
from .cost_mixin import AudioCostMixin, EmbeddingCostMixin
from .discovery import (
    clear_discovery_cache,
    display_available_models,
    get_available_audio_models,
    get_available_embedding_models,
    get_available_image_gen_models,
    get_available_models,
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
from .tracker import configure_tracker, get_tracker

__all__ = [
    "AudioCostMixin",
    "BudgetPolicy",
    "BudgetState",
    "CacheBackend",
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
    "UsageSession",
    "clear_discovery_cache",
    "clear_overrides",
    "configure_cache",
    "configure_logging",
    "configure_tracker",
    "create_tukuy_backend",
    "display_available_models",
    "enforce_budget",
    "estimate_cost",
    "estimate_tokens",
    "get_available_audio_models",
    "get_available_embedding_models",
    "get_available_image_gen_models",
    "get_available_models",
    "get_cache",
    "get_capabilities",
    "get_compatibility_matrix",
    "get_model_capabilities",
    "get_model_info",
    "get_model_lifecycle",
    "get_model_rates",
    "get_recently_used_models",
    "get_tracker",
    "override_capabilities",
    "refresh_rates_cache",
    "register_model",
    "register_provider",
    "settings",
]
