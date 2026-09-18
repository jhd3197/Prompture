"""Name → strategy factory registry.

Keeps the *set* of available execution strategies open (a host can register its
own) while giving the adaptive policy and the workflow node a stable way to
resolve one by name.

The registry stores factories, not instances, because a strategy carries
configuration (model, driver, iteration bounds) that belongs to a call site
rather than to a process-wide singleton.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .base import ExecutionStrategy
from .critique import DraftAndCritiqueStrategy
from .direct import DirectStrategy
from .retrieve import RetrieveAndVerifyStrategy

__all__ = [
    "BUILTIN_STRATEGIES",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "unregister_strategy",
]

StrategyFactory = Callable[..., ExecutionStrategy]

_lock = threading.RLock()
_REGISTRY: dict[str, StrategyFactory] = {}

#: The three fixed strategies this roadmap phase delivers.
BUILTIN_STRATEGIES: dict[str, StrategyFactory] = {
    DirectStrategy.name: DirectStrategy,
    RetrieveAndVerifyStrategy.name: RetrieveAndVerifyStrategy,
    DraftAndCritiqueStrategy.name: DraftAndCritiqueStrategy,
}


def register_strategy(name: str, factory: StrategyFactory, *, overwrite: bool = False) -> None:
    """Register *factory* under *name*.

    Raises:
        ValueError: When *name* is taken and ``overwrite`` is False.
    """
    with _lock:
        if name in _REGISTRY and not overwrite:
            raise ValueError(f"Execution strategy {name!r} is already registered; pass overwrite=True to replace it.")
        _REGISTRY[name] = factory


def unregister_strategy(name: str) -> bool:
    """Remove a registration.  Returns whether one existed."""
    with _lock:
        return _REGISTRY.pop(name, None) is not None


def list_strategies() -> list[str]:
    """Every registered strategy name, sorted."""
    with _lock:
        return sorted(_REGISTRY)


def get_strategy(name: str, **kwargs: Any) -> ExecutionStrategy:
    """Build the strategy registered as *name*, forwarding ``**kwargs``.

    Raises:
        KeyError: When no strategy is registered under that name.
    """
    with _lock:
        factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(f"No execution strategy registered as {name!r}. Known: {list_strategies()}")
    return factory(**kwargs)


for _name, _factory in BUILTIN_STRATEGIES.items():
    register_strategy(_name, _factory, overwrite=True)
del _name, _factory
