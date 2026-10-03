"""The ``providers`` section of ``prompture doctor``.

Offline, each provider row checks what can be known without a network call:
the key (or local endpoint) is configured, the configured base URL is a
well-formed http(s) URL, the provider's SDK package is importable and its
driver class imports. Providers with no key at all collapse into one summary
row unless ``verbose`` is set.

With ``live=True`` configured providers get the cheapest real call there is,
their model listing endpoint, with a short timeout and the latency reported.
Drivers that only return a static catalog (no listing endpoint) are not
called. Nothing here ever runs a paid generation.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from ..capabilities.health import HealthStatus, register_capability
from ..security.redaction import scrub_secrets, scrub_url_credentials

logger = logging.getLogger("prompture.doctor")

CAPABILITY_NAME = "providers"
LIVE_TIMEOUT_SECONDS = 6.0
#: Extra wait past the driver's own timeout before a live call counts as hung.
LIVE_GRACE_SECONDS = 2.0

#: Modalities checked, in the order a driver spec is picked for the SDK/live check.
MODALITIES: tuple[str, ...] = (
    "llm",
    "embedding",
    "stt",
    "tts",
    "img_gen",
    "video_gen",
    "rerank",
    "moderation",
    "decision",
    "lipsync",
    "music",
)

#: provider → (importable SDK modules, ``prompture[extra]`` that installs them or ``None``).
#: Providers not listed here talk plain HTTP through core dependencies.
PROVIDER_SDKS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "openai": (("openai",), "openai"),
    "azure": (("openai",), "openai"),
    "claude": (("anthropic",), "anthropic"),
    "google": (("google.genai",), "google"),
    "google_vertexai": (("google.genai",), "google"),
    "groq": (("groq",), "groq"),
    "bedrock": (("boto3",), "bedrock"),
    "airllm": (("airllm",), "airllm"),
    "laya": (("laya",), "laya"),
}

#: pip distribution name when it differs from the import name.
_PIP_NAMES = {"google.genai": "google-genai"}


def _bedrock_offline_configured() -> bool:
    """Bedrock without boto3's credential chain (which may call the EC2 metadata service)."""
    from ..infra.settings import settings

    key = getattr(settings, "aws_access_key_id", None) or os.getenv("AWS_ACCESS_KEY_ID")
    secret = getattr(settings, "aws_secret_access_key", None) or os.getenv("AWS_SECRET_ACCESS_KEY")
    return bool((key and secret) or os.getenv("AWS_PROFILE"))


#: Offline replacements for descriptor checks that could touch the network.
OFFLINE_CONFIGURED: dict[str, Callable[[], bool]] = {"bedrock": _bedrock_offline_configured}

_POPULAR_KEYS = ("OPENAI_API_KEY", "CLAUDE_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")

_KEY_HINTS = {"bedrock": "AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY (or AWS_PROFILE)"}


@dataclass
class ProviderCheck:
    """Offline facts about one provider, before they become a row."""

    name: str
    display_name: str
    modalities: list[str]
    local: bool
    configured: bool
    explicit: bool
    key_env: str | None
    endpoints: dict[str, str] = field(default_factory=dict)  # env var -> value
    spec_cls_path: str | None = None


# ── Descriptor helpers ────────────────────────────────────────────────────


def _descriptors() -> list[Any]:
    from ..drivers.provider_descriptors import PROVIDER_DESCRIPTORS

    return [d for d in (PROVIDER_DESCRIPTORS or []) if d.alias_for is None]


def _modalities(desc: Any) -> list[str]:
    return [m for m in MODALITIES if getattr(desc, f"{m}_sync", None) is not None]


def _specs(desc: Any) -> list[Any]:
    return [getattr(desc, f"{m}_sync") for m in _modalities(desc)]


def _settings_value(attr: str) -> Any:
    from ..infra.settings import settings

    return getattr(settings, attr, None) or os.getenv(attr.upper())


def _explicitly_set(attr: str) -> bool:
    from ..infra.settings import settings

    if os.getenv(attr.upper()):
        return True
    fields_set = getattr(settings, "model_fields_set", set()) or set()
    return attr in fields_set and bool(getattr(settings, attr, None))


def _endpoint_attrs(desc: Any) -> list[str]:
    attrs: list[str] = []
    for spec in _specs(desc):
        for attr in spec.kwarg_map.values():
            if ("endpoint" in attr or "base_url" in attr) and attr not in attrs:
                attrs.append(attr)
    return attrs


def _key_env(desc: Any) -> str | None:
    if desc.name in _KEY_HINTS:
        return _KEY_HINTS[desc.name]
    if desc.is_configured_check:
        return desc.is_configured_check.upper()
    for spec in _specs(desc):
        for attr in spec.kwarg_map.values():
            if any(tok in attr for tok in ("api_key", "token", "access_key")):
                return attr.upper()
    if desc.is_configured_fn is not None:
        return f"{desc.name.upper()}_API_KEY"
    return None


def _is_configured(desc: Any) -> bool:
    """Mirror of discovery's configured check, evaluated on *desc* itself and kept offline."""
    if desc.name in OFFLINE_CONFIGURED:
        try:
            return OFFLINE_CONFIGURED[desc.name]()
        except Exception:
            return False
    if desc.always_available:
        return not (desc.name == "local_http" and not _settings_value("local_http_endpoint"))
    try:
        if desc.is_configured_fn is not None:
            return bool(desc.is_configured_fn(env=None))
        if desc.is_configured_check:
            return bool(_settings_value(desc.is_configured_check))
    except Exception:
        logger.debug("configured check failed for %s", desc.name, exc_info=True)
    return False


def inspect_provider(desc: Any) -> ProviderCheck:
    """Collect the offline facts for one descriptor."""
    endpoint_attrs = _endpoint_attrs(desc)
    endpoints = {attr.upper(): str(v) for attr in endpoint_attrs if (v := _settings_value(attr))}
    explicit = any(_explicitly_set(a) for a in endpoint_attrs)
    specs = _specs(desc)
    return ProviderCheck(
        name=desc.name,
        display_name=desc.display_name or desc.name,
        modalities=_modalities(desc),
        local=bool(desc.always_available),
        configured=_is_configured(desc),
        explicit=explicit,
        key_env=None if desc.always_available else _key_env(desc),
        endpoints=endpoints,
        spec_cls_path=specs[0].cls_path if specs else None,
    )


# ── Offline checks ────────────────────────────────────────────────────────


def sdk_status(provider: str) -> tuple[bool, str | None, str | None]:
    """``(importable, missing_module, fix)`` for the provider's SDK packages."""
    modules, extra = PROVIDER_SDKS.get(provider, ((), None))
    for module in modules:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            fix = f"pip install prompture[{extra}]" if extra else f"pip install {_PIP_NAMES.get(module, module)}"
            return False, module, fix
    return True, None, None


def _driver_import_error(cls_path: str | None) -> str | None:
    if not cls_path:
        return None
    try:
        from ..drivers.provider_descriptors import _resolve_cls

        _resolve_cls(cls_path)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def url_problem(value: str) -> str | None:
    """Why *value* isn't a usable base URL, or ``None`` when it is."""
    try:
        parsed = urlparse(value.strip())
    except ValueError as exc:
        return f"unparseable URL ({exc})"
    if parsed.scheme not in ("http", "https"):
        return "must start with http:// or https://"
    if not parsed.hostname:
        return "has no host"
    try:
        _ = parsed.port
    except ValueError:
        return "has an invalid port"
    return None


def _example_url(env_var: str) -> str:
    return "http://localhost:11434" if "OLLAMA" in env_var else "https://host.example/v1"


def offline_row(check: ProviderCheck) -> HealthStatus:
    """The offline row for a configured (or local) provider."""
    details: dict[str, Any] = {
        "display_name": check.display_name,
        "modalities": check.modalities,
        "local": check.local,
        "key_env": check.key_env,
        "endpoints": {k: scrub_url_credentials(v) for k, v in check.endpoints.items()},
    }
    active = next(iter(details["endpoints"].values()), None) if check.endpoints else None

    def row(status: str, message: str, fix: str | None = None) -> HealthStatus:
        return HealthStatus(
            check.name,
            status,  # type: ignore[arg-type]
            category="providers",
            active_backend=active if status == "ok" else None,
            message=message,
            fix_hint=fix,
            details=details,
        )

    for env_var, value in check.endpoints.items():
        problem = url_problem(value)
        if problem:
            return row(
                "error",
                f"{env_var} {problem}",
                f"set {env_var} to a full URL, e.g. {env_var}={_example_url(env_var)}",
            )

    importable, module, fix = sdk_status(check.name)
    details["sdk"] = PROVIDER_SDKS.get(check.name, ((), None))[0] or None
    if not importable:
        return row("missing", f"SDK package {module!r} is not installed", fix)

    err = _driver_import_error(check.spec_cls_path)
    if err:
        return row("broken", f"driver failed to import: {err}", "pip install --force-reinstall prompture")

    if check.local:
        where = active or "default endpoint"
        return row("ok", f"local provider at {where} (not contacted; use --live)")
    via = check.key_env or "configuration"
    return row("ok", f"configured via {via}")


def unconfigured_row(check: ProviderCheck) -> HealthStatus:
    if check.local:
        importable, module, fix = sdk_status(check.name)
        message = f"local provider; SDK {module!r} not installed" if not importable else "local provider not set up"
        endpoint_env = next(iter(check.endpoints), None)
        fix = fix or (f"set {endpoint_env}" if endpoint_env else None)
    else:
        message = "no key configured"
        fix = f"set {check.key_env}" if check.key_env else "configure it in code (see the provider docs)"
    return HealthStatus(
        check.name,
        "unconfigured",
        category="providers",
        message=message,
        fix_hint=fix,
        details={"display_name": check.display_name, "modalities": check.modalities, "key_env": check.key_env},
    )


def _include_local(check: ProviderCheck) -> bool:
    """Local providers are shown when explicitly configured or their SDK is installed."""
    if check.explicit:
        return True
    if check.name == "local_http":
        return check.configured
    if check.name in PROVIDER_SDKS:
        return sdk_status(check.name)[0]
    # Plain-HTTP local servers: only when there is an endpoint to talk to.
    return bool(check.endpoints)


def _summary_row(unconfigured: list[ProviderCheck], configured_count: int) -> HealthStatus:
    names = [c.name for c in unconfigured]
    envs = [c.key_env for c in unconfigured if c.key_env]
    hint_envs = sorted(envs, key=lambda e: (e not in _POPULAR_KEYS, not e.endswith("_API_KEY")))[:3]
    shown = ", ".join(names[:12]) + (f", +{len(names) - 12} more" if len(names) > 12 else "")
    lead = "no provider configured" if configured_count == 0 else f"{len(names)} providers not configured"
    fix = "set a provider key"
    if hint_envs:
        fix += f" (e.g. {', '.join(hint_envs)})"
    fix += "; `prompture doctor --verbose` lists each"
    return HealthStatus(
        "unconfigured providers",
        "unconfigured",
        category="providers",
        message=f"{lead}: {shown}",
        fix_hint=fix,
        details={"providers": {c.name: c.key_env for c in unconfigured}},
    )


# ── Live checks ───────────────────────────────────────────────────────────


class _LiveLogCapture(logging.Handler):
    """Keeps the last driver warning per live-check thread, to explain a failed listing."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self._last: dict[str, str] = {}
        self._mutex = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        if not record.threadName or not record.threadName.startswith("doctor-live-"):
            return
        try:
            message = record.getMessage()
        except Exception:
            return
        with self._mutex:
            self._last[record.threadName] = message

    def last(self, thread_name: str) -> str | None:
        with self._mutex:
            return self._last.pop(thread_name, None)


_CAPTURE = _LiveLogCapture()
#: Loggers driver listing code writes to (``base.py`` uses ``prompture.driver``).
DRIVER_LOGGERS = ("prompture.driver", "prompture.drivers")


def _live_driver_cls(desc: Any) -> Any:
    """The first driver class whose ``list_models`` does a real (timeout-bounded) listing call."""
    from ..drivers.provider_descriptors import _resolve_cls

    for spec in _specs(desc):
        try:
            cls = _resolve_cls(spec.cls_path)
            params = inspect.signature(cls.list_models).parameters
        except Exception:
            continue
        if "timeout" in params:
            return cls
    return None


def _list_models_kwargs(desc: Any) -> dict[str, Any]:
    from ..infra.settings import settings

    kwargs: dict[str, Any] = {}
    for ctor_kwarg, attr, env_var in desc.list_models_kwargs:
        value = getattr(settings, attr, None) or (os.getenv(env_var) if env_var else None)
        if value:
            kwargs[ctor_kwarg] = value
    return kwargs


def live_row(desc: Any, check: ProviderCheck, base: HealthStatus, timeout: float) -> HealthStatus:
    """Upgrade an ok offline row with a models-list call."""
    cls = _live_driver_cls(desc)
    details = dict(base.details)
    if cls is None:
        details["live"] = "not available"
        return HealthStatus(
            base.name,
            base.status,
            category="providers",
            active_backend=base.active_backend,
            message=f"{base.message}; no free live check (static model catalog)",
            fix_hint=base.fix_hint,
            details=details,
        )
    kwargs = _list_models_kwargs(desc)
    started = time.perf_counter()
    outcome: dict[str, Any] = {}

    def _call() -> None:
        try:
            outcome["models"] = cls.list_models(timeout=max(1, int(timeout)), **kwargs)
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    # A daemon thread, so an SDK that ignores its timeout can't hold the process open.
    worker = threading.Thread(target=_call, name=f"doctor-live-{check.name}", daemon=True)
    worker.start()
    worker.join(timeout + LIVE_GRACE_SECONDS)
    if worker.is_alive():
        details["live"] = "timeout"
        return HealthStatus(
            check.name,
            "timeout",
            category="providers",
            message=f"models list did not answer within {timeout:.0f}s",
            fix_hint=_live_fix(check),
            details=details,
        )
    models, error = outcome.get("models"), outcome.get("error") or _CAPTURE.last(worker.name)
    latency_ms = round((time.perf_counter() - started) * 1000)
    details["latency_ms"] = latency_ms

    if models is not None:
        details["live"] = "ok"
        details["model_count"] = len(models)
        return HealthStatus(
            check.name,
            "ok",
            category="providers",
            active_backend=base.active_backend,
            message=f"models list OK ({len(models)} models, {latency_ms} ms)",
            details=details,
        )
    details["live"] = "failed"
    if check.local and not check.explicit:
        where = base.active_backend or "the default endpoint"
        return HealthStatus(
            check.name,
            "unconfigured",
            category="providers",
            message=f"no local server answering at {where}",
            fix_hint=_live_fix(check),
            details=details,
        )
    if error:
        reason = _clean_reason(error)
    elif check.local:
        reason = "server not reachable"
    else:
        reason = "rejected key, endpoint down, or no network"
    return HealthStatus(
        check.name,
        "error",
        category="providers",
        message=f"models list failed: {reason} ({latency_ms} ms)",
        fix_hint=_live_fix(check),
        details=details,
    )


def _clean_reason(text: str) -> str:
    """Shorten a driver warning to its cause: no logger prefix, no URLs, no secrets."""
    line = scrub_secrets(text.strip().splitlines()[0] if text.strip() else text)
    line = re.sub(r"^model discovery:\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"https?://\S+\s*", "", line)
    line = re.sub(r"^returned\s+", "", line)
    line = line.replace("—", "-")
    return " ".join(line.split())[:160] or "unknown error"


def _live_fix(check: ProviderCheck) -> str:
    endpoint_env = next(iter(check.endpoints), None)
    if check.local:
        start = f"`{check.name} serve`" if check.name == "ollama" else "the local server"
        return f"start {start}" + (f" or point {endpoint_env} at it" if endpoint_env else "")
    parts = [f"check the value of {check.key_env}"] if check.key_env else []
    if endpoint_env:
        parts.append(f"check {endpoint_env}")
    parts.append("check network/proxy (PROMPTURE_PROXY)")
    return "; ".join(parts)


# ── Entry point ───────────────────────────────────────────────────────────


def provider_rows(
    live: bool = False,
    *,
    verbose: bool = False,
    timeout: float = LIVE_TIMEOUT_SECONDS,
    descriptors: Iterable[Any] | None = None,
) -> list[HealthStatus]:
    """Health rows for every provider (compact unless *verbose*)."""
    descs = list(descriptors) if descriptors is not None else _descriptors()
    rows: list[HealthStatus] = []
    unconfigured: list[ProviderCheck] = []
    live_jobs: list[tuple[int, Any, ProviderCheck]] = []

    for desc in descs:
        try:
            check = inspect_provider(desc)
        except Exception as exc:
            rows.append(HealthStatus(desc.name, "error", category="providers", message=f"inspection failed: {exc}"))
            continue
        active = _include_local(check) if check.local else check.configured
        if not active:
            if verbose:
                rows.append(unconfigured_row(check))
            else:
                unconfigured.append(check)
            continue
        row = offline_row(check)
        rows.append(row)
        if live and row.status == "ok":
            live_jobs.append((len(rows) - 1, desc, check))

    if live_jobs:
        for name in DRIVER_LOGGERS:
            logging.getLogger(name).addHandler(_CAPTURE)
        try:
            _run_live_jobs(rows, live_jobs, timeout)
        finally:
            for name in DRIVER_LOGGERS:
                logging.getLogger(name).removeHandler(_CAPTURE)

    if unconfigured:
        configured_count = sum(1 for r in rows if r.status != "unconfigured")
        rows.append(_summary_row(unconfigured, configured_count))
    return rows


def _run_live_jobs(rows: list[HealthStatus], live_jobs: list[tuple[int, Any, ProviderCheck]], timeout: float) -> None:
    if live_jobs:
        with ThreadPoolExecutor(max_workers=min(8, len(live_jobs)), thread_name_prefix="doctor") as pool:
            futures = {idx: pool.submit(live_row, desc, check, rows[idx], timeout) for idx, desc, check in live_jobs}
            for idx, fut in futures.items():
                try:
                    rows[idx] = fut.result()
                except Exception as exc:  # live_row guards itself; belt and braces
                    rows[idx] = HealthStatus(rows[idx].name, "error", category="providers", message=str(exc))


def _check(live: bool) -> list[HealthStatus]:
    return provider_rows(live)


register_capability(
    CAPABILITY_NAME,
    "providers",
    _check,
    description="LLM / embedding / image / audio providers: key, base URL, SDK; --live lists models",
)
