"""Virtual model names: combos, aliases and ``auto/`` models.

Anywhere a model string is accepted — ``get_driver_for_model``,
``Conversation(model_name=...)``, ``prompture serve``, a gateway — these
resolve to a :class:`~.driver.ResilientDriver`:

* ``combo/<name>`` — a registered fallback chain with a routing strategy::

      register_combo("chat", ["openai/gpt-4o", "claude/claude-sonnet-4-5"], strategy="latency")
      get_driver_for_model("combo/chat")

* aliases — a short name for a model string (or for a combo)::

      register_model_alias("fast", "groq/llama-3.3-70b-versatile")
      get_driver_for_model("fast")

* ``auto/<mode>`` — built from whatever providers are configured, using the
  pricing-tier classification of :class:`~prompture.pipeline.routing.ModelRouter`:
  ``auto/cheap``, ``auto/fast``, ``auto/best``, ``auto/balanced``, or a tier
  name ``auto/budget`` / ``auto/standard`` / ``auto/premium``.

Combos and aliases can also be loaded from JSON (:func:`load_combos`), and a
file named by ``PROMPTURE_COMBOS_FILE`` is loaded automatically on first use.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backoff import RetryPolicy
from .router import Target
from .strategies import STRATEGIES

COMBO_PREFIX = "combo/"
FUSION_PREFIX = "fusion/"
AUTO_PREFIX = "auto/"

AUTO_MODES: dict[str, tuple[tuple[str, ...], str]] = {
    # mode: (tiers to draw from, strategy)
    "cheap": (("budget", "standard"), "cheapest"),
    "fast": (("budget", "standard"), "latency"),
    "best": (("premium", "standard"), "priority"),
    "balanced": (("standard", "budget", "premium"), "priority"),
    "budget": (("budget",), "cheapest"),
    "standard": (("standard",), "priority"),
    "premium": (("premium",), "priority"),
}

#: How many models an ``auto/`` route falls back across.
AUTO_MAX_TARGETS = 5


@dataclass
class Combo:
    """A named, ordered set of targets plus how to route across them."""

    name: str
    targets: list[str | Target]
    strategy: str = "priority"
    weights: list[float] | None = None
    sticky: bool = False
    policy: RetryPolicy | None = None
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError(f"Combo '{self.name}' needs at least one target")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"Combo '{self.name}': unknown strategy '{self.strategy}'")
        if self.weights is not None and len(self.weights) != len(self.targets):
            raise ValueError(f"Combo '{self.name}': {len(self.targets)} targets but {len(self.weights)} weights")

    @property
    def model_name(self) -> str:
        return f"{COMBO_PREFIX}{self.name}"

    def driver(self, *, async_: bool = False, **kwargs: Any) -> Any:
        from .async_driver import AsyncResilientDriver
        from .driver import ResilientDriver

        cls = AsyncResilientDriver if async_ else ResilientDriver
        drv = cls(
            list(self.targets),
            policy=self.policy,
            strategy=self.strategy,
            weights=self.weights,
            sticky=self.sticky,
            **kwargs,
        )
        drv.model = self.model_name
        return drv

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "targets": [t if isinstance(t, str) else t.model for t in self.targets],
            "strategy": self.strategy,
            "weights": self.weights,
            "sticky": self.sticky,
            "description": self.description,
        }


_lock = threading.RLock()
_combos: dict[str, Combo] = {}
_aliases: dict[str, str] = {}
_env_loaded = False


def register_combo(
    name: str,
    targets: Sequence[str | Target],
    *,
    strategy: str = "priority",
    weights: Sequence[float] | None = None,
    sticky: bool = False,
    policy: RetryPolicy | None = None,
    description: str = "",
) -> Combo:
    """Register (or replace) ``combo/<name>``."""
    name = name.removeprefix(COMBO_PREFIX)
    combo = Combo(
        name,
        list(targets),
        strategy=strategy,
        weights=list(weights) if weights is not None else None,
        sticky=sticky,
        policy=policy,
        description=description,
    )
    with _lock:
        _combos[name] = combo
    return combo


def unregister_combo(name: str) -> None:
    with _lock:
        _combos.pop(name.removeprefix(COMBO_PREFIX), None)


def get_combo(name: str) -> Combo | None:
    _ensure_env_loaded()
    with _lock:
        return _combos.get(name.removeprefix(COMBO_PREFIX))


def list_combos() -> list[Combo]:
    _ensure_env_loaded()
    with _lock:
        return list(_combos.values())


def register_model_alias(alias: str, target: str) -> None:
    """Make *alias* resolve to *target* (a model string, ``combo/…`` or ``auto/…``)."""
    if alias == target:
        raise ValueError("An alias cannot point at itself")
    with _lock:
        _aliases[alias] = target


def unregister_model_alias(alias: str) -> None:
    with _lock:
        _aliases.pop(alias, None)


def list_model_aliases() -> dict[str, str]:
    _ensure_env_loaded()
    with _lock:
        return dict(_aliases)


def resolve_model_alias(model: str, *, max_depth: int = 8) -> str:
    """Follow aliases until a non-alias name is reached."""
    _ensure_env_loaded()
    seen = [model]
    current = model
    with _lock:
        for _ in range(max_depth):
            nxt = _aliases.get(current)
            if nxt is None:
                return current
            if nxt in seen:
                raise ValueError(f"Alias cycle: {' -> '.join([*seen, nxt])}")
            seen.append(nxt)
            current = nxt
    raise ValueError(f"Alias chain too deep starting at '{model}'")


def clear_virtual_models() -> None:
    """Drop every registered combo and alias (tests, reloads)."""
    global _env_loaded
    with _lock:
        _combos.clear()
        _aliases.clear()
        _env_loaded = True  # don't re-read the env file after an explicit clear


def load_combos(source: str | os.PathLike[str] | Mapping[str, Any]) -> list[Combo]:
    """Register combos and aliases from a JSON file path or an already-parsed mapping.

    Shape::

        {
          "combos": {
            "chat": {"targets": ["openai/gpt-4o", "claude/claude-sonnet-4-5"], "strategy": "latency"},
            "cheap": ["groq/llama-3.1-8b-instant", "openai/gpt-4o-mini"]
          },
          "aliases": {"fast": "groq/llama-3.3-70b-versatile", "default": "combo/chat"}
        }
    """
    data: Mapping[str, Any]
    if isinstance(source, Mapping):
        data = source
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    loaded: list[Combo] = []
    for name, spec in (data.get("combos") or {}).items():
        if isinstance(spec, list):
            spec = {"targets": spec}
        policy = spec.get("policy")
        loaded.append(
            register_combo(
                name,
                spec["targets"],
                strategy=spec.get("strategy", "priority"),
                weights=spec.get("weights"),
                sticky=bool(spec.get("sticky", False)),
                policy=RetryPolicy(**policy) if isinstance(policy, Mapping) else None,
                description=spec.get("description", ""),
            )
        )
    for alias, target in (data.get("aliases") or {}).items():
        register_model_alias(alias, target)
    return loaded


def _ensure_env_loaded() -> None:
    global _env_loaded
    if _env_loaded:
        return
    with _lock:
        if _env_loaded:
            return
        _env_loaded = True
        path = os.environ.get("PROMPTURE_COMBOS_FILE")
    if path:
        load_combos(path)


# ---------------------------------------------------------------------------
# auto/<mode>
# ---------------------------------------------------------------------------


_NON_CHAT_RE = re.compile(
    r"embed|rerank|moderation|whisper|transcri|tts|speech|dall-e|imagen|image|video|audio|realtime|"
    r"guard|search|similarity|lipsync|music|lyria|veo-|sora|kling|flux|stable-diffusion|seedance",
    re.IGNORECASE,
)


def _is_chat_model(model: str) -> bool:
    """Best-effort: does *model* produce text in a chat turn?"""
    if _NON_CHAT_RE.search(model):
        return False
    if "/" not in model:
        return True
    provider, model_id = model.split("/", 1)
    try:
        from ..infra.model_rates import get_model_capabilities

        caps = get_model_capabilities(provider, model_id)
    except Exception:
        return True
    return caps is None or not caps.modalities_output or "text" in caps.modalities_output


def auto_targets(mode: str, *, limit: int = AUTO_MAX_TARGETS) -> list[str]:
    """Models an ``auto/<mode>`` route will try, best first."""
    if mode not in AUTO_MODES:
        raise ValueError(f"Unknown auto mode '{mode}'. Choose one of: {', '.join(AUTO_MODES)}")
    from ..pipeline.routing import ModelRouter

    tiers, _strategy = AUTO_MODES[mode]
    router = ModelRouter()
    available = router.available_models()
    by_tier: dict[str, list[str]] = {t: [] for t in tiers}
    for model in available:
        if not _is_chat_model(model):
            continue
        tier = router.model_tier(model)
        if tier in by_tier:
            by_tier[tier].append(model)
    from .strategies import _price_per_mtok

    # "best" wants the priciest (flagship) models first; everything else the cheapest.
    descending = mode in ("best", "premium")

    def price_key(m: str) -> tuple[bool, float]:
        price = _price_per_mtok(m)
        if price is None:
            return (True, 0.0)
        return (False, -price if descending else price)

    ordered: list[str] = []
    for tier in tiers:
        ordered.extend(_interleave_providers(sorted(by_tier[tier], key=price_key)))
    return ordered[:limit]


def _interleave_providers(models: list[str]) -> list[str]:
    """Round-robin across providers so one outage can't take out every fallback."""
    queues: dict[str, list[str]] = {}
    for m in models:
        queues.setdefault(m.split("/", 1)[0], []).append(m)
    out: list[str] = []
    while any(queues.values()):
        for provider in list(queues):
            if queues[provider]:
                out.append(queues[provider].pop(0))
    return out


def _auto_driver(mode: str, *, async_: bool) -> Any:
    from .async_driver import AsyncResilientDriver
    from .driver import ResilientDriver

    targets = auto_targets(mode)
    if not targets:
        raise ValueError(f"auto/{mode}: no configured models fall in its tiers — set a provider API key first")
    _tiers, strategy = AUTO_MODES[mode]
    cls = AsyncResilientDriver if async_ else ResilientDriver
    drv = cls(targets, strategy=strategy)
    drv.model = f"{AUTO_PREFIX}{mode}"
    return drv


# ---------------------------------------------------------------------------
# Resolution hook used by the driver factories
# ---------------------------------------------------------------------------


def resolve_virtual_model(model: str, *, async_: bool = False) -> tuple[str, Any | None]:
    """Resolve aliases, then build a driver for ``combo/`` / ``auto/`` names.

    Returns ``(resolved_model_string, driver_or_None)``. ``None`` means the
    name is an ordinary ``provider/model`` string the caller should build
    itself (possibly after alias substitution).
    """
    resolved = resolve_model_alias(model)
    if resolved.startswith(COMBO_PREFIX):
        combo = get_combo(resolved)
        if combo is None:
            known = ", ".join(c.model_name for c in list_combos()) or "none registered"
            raise ValueError(f"Unknown combo '{resolved}' ({known})")
        return resolved, combo.driver(async_=async_)
    if resolved.startswith(AUTO_PREFIX):
        return resolved, _auto_driver(resolved[len(AUTO_PREFIX) :], async_=async_)
    if resolved.startswith(FUSION_PREFIX):
        from ..groups.fusion import AsyncFusionDriver, get_fusion, list_fusions

        spec = get_fusion(resolved)
        if spec is None:
            known = ", ".join(f"fusion/{f.name}" for f in list_fusions()) or "none registered"
            raise ValueError(f"Unknown fusion '{resolved}' ({known})")
        drv = spec.driver()
        return resolved, AsyncFusionDriver(drv) if async_ else drv
    return resolved, None


def list_virtual_models() -> list[str]:
    """Model names that only exist as combos, aliases or auto modes (for ``/v1/models``)."""
    from ..groups.fusion import list_fusions

    names = [c.model_name for c in list_combos()]
    names += [f"{FUSION_PREFIX}{f.name}" for f in list_fusions()]
    names += sorted(list_model_aliases())
    names += [f"{AUTO_PREFIX}{m}" for m in AUTO_MODES]
    return names
