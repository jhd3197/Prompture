"""Resilience layer: error classification, retries, circuit breakers, key pools,
failover, routing strategies and virtual models (combos, aliases, ``auto/``).

The entry points are :func:`resilient` / :func:`async_resilient`, which build
a driver that retries transient errors, parks rate-limited or broken
targets, rotates API keys, and fails over across models — while exposing the
ordinary driver interface.
"""

from .async_driver import AsyncResilientDriver, async_resilient
from .backoff import NO_RETRY, RetryPolicy
from .breaker import (
    BreakerConfig,
    BreakerRegistry,
    BreakerState,
    CircuitBreaker,
    get_breaker_registry,
    key_scope,
    model_scope,
    provider_scope,
)
from .driver import ResilientDriver, resilient
from .errors import (
    AllTargetsFailedError,
    ErrorAction,
    ErrorInfo,
    ErrorRule,
    classify_error,
    parse_duration,
    register_error_rule,
    reset_error_rules,
)
from .explain import explain_attempt, explain_route
from .headroom import HeadroomTracker, get_headroom_tracker, partition_by_headroom
from .keys import KeyPool, clear_key_pools, get_key_pool, key_id, register_key_pool
from .router import Target
from .strategies import STRATEGIES, RouteStats, get_route_stats
from .virtual import (
    Combo,
    auto_targets,
    clear_virtual_models,
    get_combo,
    list_combos,
    list_model_aliases,
    list_virtual_models,
    load_combos,
    register_combo,
    register_model_alias,
    resolve_model_alias,
    resolve_virtual_model,
    unregister_combo,
    unregister_model_alias,
)

__all__ = [
    "NO_RETRY",
    "STRATEGIES",
    "AllTargetsFailedError",
    "AsyncResilientDriver",
    "BreakerConfig",
    "BreakerRegistry",
    "BreakerState",
    "CircuitBreaker",
    "Combo",
    "ErrorAction",
    "ErrorInfo",
    "ErrorRule",
    "HeadroomTracker",
    "KeyPool",
    "ResilientDriver",
    "RetryPolicy",
    "RouteStats",
    "Target",
    "async_resilient",
    "auto_targets",
    "classify_error",
    "clear_key_pools",
    "clear_virtual_models",
    "explain_attempt",
    "explain_route",
    "get_breaker_registry",
    "get_combo",
    "get_headroom_tracker",
    "get_key_pool",
    "get_route_stats",
    "key_id",
    "key_scope",
    "list_combos",
    "list_model_aliases",
    "list_virtual_models",
    "load_combos",
    "model_scope",
    "parse_duration",
    "partition_by_headroom",
    "provider_scope",
    "register_combo",
    "register_error_rule",
    "register_key_pool",
    "register_model_alias",
    "reset_error_rules",
    "resilient",
    "resolve_model_alias",
    "resolve_virtual_model",
    "unregister_combo",
    "unregister_model_alias",
]
