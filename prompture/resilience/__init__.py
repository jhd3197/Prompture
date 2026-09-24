"""Resilience layer: error classification, retries, circuit breakers, key pools, failover.

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
from .keys import KeyPool, clear_key_pools, get_key_pool, key_id, register_key_pool
from .router import Target

__all__ = [
    "NO_RETRY",
    "AllTargetsFailedError",
    "AsyncResilientDriver",
    "BreakerConfig",
    "BreakerRegistry",
    "BreakerState",
    "CircuitBreaker",
    "ErrorAction",
    "ErrorInfo",
    "ErrorRule",
    "KeyPool",
    "ResilientDriver",
    "RetryPolicy",
    "Target",
    "async_resilient",
    "classify_error",
    "clear_key_pools",
    "get_breaker_registry",
    "get_key_pool",
    "key_id",
    "key_scope",
    "model_scope",
    "parse_duration",
    "provider_scope",
    "register_error_rule",
    "register_key_pool",
    "reset_error_rules",
    "resilient",
]
