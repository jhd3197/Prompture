"""``prompture setup`` / ``configure`` / ``reset`` — configure without hand-editing ``.env``.

* ``setup`` — interactive wizard: providers and keys (validated live before
  saving), default model, optional extras, proxy settings tailored to the
  detected environment.
* ``configure KEY VALUE`` — non-interactive store edits; ``--list`` shows key
  names with masked values.
* ``reset`` — removes Prompture's local state under ``~/.prompture``.

Every mutating command accepts ``--dry-run`` and then only prints its plan.
Secrets are never echoed back in full.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click

from ..infra.credentials import (
    CredentialStore,
    CredentialStoreError,
    active_profile,
    is_secret_name,
    mask_value,
    prompture_state_dir,
)

__all__ = [
    "COMMANDS",
    "KeyCheck",
    "ProviderKey",
    "configure",
    "provider_key_catalogue",
    "reset",
    "setup",
    "validate_key",
]

# ---------------------------------------------------------------------------
# Provider catalogue and live key validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderKey:
    """A provider whose access is a single API key."""

    provider: str
    display: str
    env_var: str
    model_setting: str | None = None
    llm: bool = False


def provider_key_catalogue() -> list[ProviderKey]:
    """Key-based providers from the provider registry, LLM providers first."""
    from ..drivers.provider_descriptors import PROVIDER_DESCRIPTORS
    from ..infra.settings import Settings

    fields = Settings.model_fields
    entries: list[ProviderKey] = []
    for desc in PROVIDER_DESCRIPTORS:
        check = desc.is_configured_check
        if desc.alias_for or not check or check not in fields:
            continue
        if not check.endswith(("_key", "_token")):
            continue
        model_setting = None
        if desc.llm_sync and desc.llm_sync.default_model in fields:
            model_setting = desc.llm_sync.default_model
        entries.append(
            ProviderKey(
                provider=desc.name,
                display=desc.display_name or desc.name,
                env_var=check.upper(),
                model_setting=model_setting,
                llm=desc.llm_sync is not None,
            )
        )
    entries.sort(key=lambda e: not e.llm)
    return entries


@dataclass(frozen=True)
class KeyCheck:
    """Outcome of a live key check.

    ``status``: ``valid`` (provider accepted it), ``invalid`` (provider rejected
    it), ``unverified`` (the check could not complete — network, rate limit,
    server error) or ``unchecked`` (no cheap check exists for this provider).
    """

    status: Literal["valid", "invalid", "unverified", "unchecked"]
    detail: str = ""


# provider -> (url, auth style). Each is the cheapest authenticated read the
# provider offers (a models / key-info listing).
_KEY_CHECKS: dict[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1/models", "bearer"),
    "claude": ("https://api.anthropic.com/v1/models?limit=1", "anthropic"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/models?pageSize=1", "goog"),
    "groq": ("https://api.groq.com/openai/v1/models", "bearer"),
    "grok": ("https://api.x.ai/v1/models", "bearer"),
    "openrouter": ("https://openrouter.ai/api/v1/key", "bearer"),
    "moonshot": ("https://api.moonshot.ai/v1/models", "bearer"),
    "mistral": ("https://api.mistral.ai/v1/models", "bearer"),
    "deepseek": ("https://api.deepseek.com/models", "bearer"),
    "cohere": ("https://api.cohere.com/v1/models?page_size=1", "bearer"),
    "elevenlabs": ("https://api.elevenlabs.io/v1/models", "xi"),
    "deepgram": ("https://api.deepgram.com/v1/projects", "token"),
}


def _auth_headers(style: str, key: str) -> dict[str, str]:
    if style == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if style == "goog":
        return {"x-goog-api-key": key}
    if style == "xi":
        return {"xi-api-key": key}
    if style == "token":
        return {"Authorization": f"Token {key}"}
    return {"Authorization": f"Bearer {key}"}


def validate_key(provider: str, key: str, *, timeout: float = 10.0) -> KeyCheck:
    """Check *key* against *provider* with one cheap authenticated GET.

    Honors ``PROMPTURE_<PROVIDER>_PROXY`` / ``PROMPTURE_PROXY``. Never includes
    the key in the returned detail.
    """
    spec = _KEY_CHECKS.get(provider)
    if spec is None:
        return KeyCheck("unchecked", "no live check for this provider; saved as entered")
    url, style = spec
    import requests

    from ..capabilities.http import proxies_for

    try:
        resp = requests.get(
            url,
            headers=_auth_headers(style, key),
            timeout=timeout,
            allow_redirects=False,
            proxies=proxies_for(provider),
        )
    except requests.Timeout:
        return KeyCheck("unverified", "the provider did not answer in time")
    except requests.RequestException as exc:
        return KeyCheck("unverified", f"network error ({type(exc).__name__})")
    code = resp.status_code
    if 200 <= code < 300:
        return KeyCheck("valid", "accepted by the provider")
    if code in (401, 403):
        return KeyCheck("invalid", f"rejected by the provider (HTTP {code})")
    if code == 400 and provider == "google":
        return KeyCheck("invalid", "rejected by the provider (HTTP 400)")
    if code == 429:
        return KeyCheck("unverified", "rate limited (HTTP 429); try again later")
    return KeyCheck("unverified", f"unexpected response (HTTP {code})")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PROXY_SCHEMES = ("http://", "https://", "socks4://", "socks5://", "socks5h://")


def _display(name: str, value: str) -> str:
    return mask_value(value, secret=is_secret_name(name))


def _known_names() -> set[str]:
    from ..infra.settings import Settings

    return {name.upper() for name in Settings.model_fields}


def _store(profile: str | None) -> CredentialStore:
    try:
        return CredentialStore(profile=profile)
    except CredentialStoreError as exc:
        raise click.UsageError(str(exc)) from exc


def _env_source(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and bool(value.strip())


def _print_plan(changes: dict[str, str], removals: list[str], profile: str, store: CredentialStore) -> None:
    click.echo(f"\nPlan for profile '{profile}' in {store.path}:")
    for name, value in changes.items():
        click.echo(f"  set    {name} = {_display(name, value)}")
    for name in removals:
        click.echo(f"  remove {name}")
    for name in changes:
        if _env_source(name):
            click.echo(f"  note   {name} is also set in the environment, which takes precedence")


# ---------------------------------------------------------------------------
# prompture setup
# ---------------------------------------------------------------------------


def _choose_providers(catalogue: list[ProviderKey], store: CredentialStore) -> list[ProviderKey]:
    click.echo("\nProviders:")
    for i, entry in enumerate(catalogue, start=1):
        status = ""
        if _env_source(entry.env_var):
            status = "  [set in environment]"
        else:
            try:
                if store.get(entry.env_var) is not None:
                    status = "  [stored]"
            except CredentialStoreError:
                pass
        kind = "" if entry.llm else " (media/other)"
        click.echo(f"  {i:>2}. {entry.display}{kind}{status}")
    raw = click.prompt(
        "Providers to configure (numbers or names, comma-separated; blank to skip)",
        default="",
        show_default=False,
    )
    by_name = {e.provider.lower(): e for e in catalogue}
    by_name.update({e.display.lower(): e for e in catalogue})
    chosen: list[ProviderKey] = []
    for token in (t.strip() for t in raw.replace(";", ",").split(",")):
        if not token:
            continue
        entry = None
        if token.isdigit() and 1 <= int(token) <= len(catalogue):
            entry = catalogue[int(token) - 1]
        else:
            entry = by_name.get(token.lower())
        if entry is None:
            click.echo(f"  Unknown provider '{token}', skipped.")
        elif entry not in chosen:
            chosen.append(entry)
    return chosen


def _ask_key(entry: ProviderKey) -> str | None:
    """Prompt for one provider key, validating it live; ``None`` means skip."""
    while True:
        key = click.prompt(
            f"\n{entry.display} key ({entry.env_var}, blank to skip)",
            default="",
            show_default=False,
            hide_input=True,
        ).strip()
        if not key:
            click.echo("  Skipped.")
            return None
        check = validate_key(entry.provider, key)
        if check.status == "valid":
            click.echo(f"  OK: {check.detail} ({mask_value(key)})")
            return key
        if check.status == "unchecked":
            click.echo(f"  Note: {check.detail}.")
            return key
        label = "Rejected" if check.status == "invalid" else "Could not verify"
        click.echo(f"  {label}: {check.detail}.")
        action = click.prompt(
            "  [k]eep anyway, [r]e-enter, [s]kip",
            type=click.Choice(["k", "r", "s"]),
            default="r" if check.status == "invalid" else "k",
        )
        if action == "k":
            return key
        if action == "s":
            click.echo("  Skipped.")
            return None


def _ask_default_model(chosen: list[ProviderKey]) -> dict[str, str]:
    from ..infra.settings import settings

    candidates = [
        f"{e.provider}/{getattr(settings, e.model_setting)}"
        for e in chosen
        if e.llm and e.model_setting and getattr(settings, e.model_setting, None)
    ]
    if candidates:
        click.echo("\nSuggested default models: " + ", ".join(candidates))
    while True:
        model = click.prompt(
            "Default model as provider/model (blank to skip)",
            default=candidates[0] if candidates else "",
            show_default=bool(candidates),
        ).strip()
        if not model:
            return {}
        if "/" in model and all(part.strip() for part in model.split("/", 1)):
            break
        click.echo("  Use the provider/model form, e.g. openai/gpt-4o-mini.")
    provider, model_name = model.split("/", 1)
    values = {"PROMPTURE_DEFAULT_MODEL": model, "AI_PROVIDER": provider}
    catalogue = {e.provider: e for e in provider_key_catalogue()}
    entry = catalogue.get(provider)
    if entry and entry.model_setting:
        values[entry.model_setting.upper()] = model_name
    return values


_EXTRAS = (
    ("web", "web search, page fetching and readers"),
    ("media", "video, podcast and audio transcripts"),
    ("mcp", "MCP servers as agent tools"),
)
_WEB_KEYS = (
    ("TAVILY_API_KEY", "Tavily"),
    ("BRAVE_SEARCH_API_KEY", "Brave Search"),
    ("SERPER_API_KEY", "Serper"),
    ("EXA_API_KEY", "Exa"),
)


def _ask_extras() -> dict[str, str]:
    click.echo("\nOptional extras (setup never installs anything):")
    for name, what in _EXTRAS:
        click.echo(f'  pip install "prompture[{name}]"   # {what}')
    values: dict[str, str] = {}
    if not click.confirm("Add optional web search keys? (web tools work without them)", default=False):
        return values
    for env_var, label in _WEB_KEYS:
        key = click.prompt(
            f"  {label} key ({env_var}, blank to skip)", default="", show_default=False, hide_input=True
        ).strip()
        if key:
            values[env_var] = key
    return values


def _valid_proxy(url: str) -> bool:
    return url.lower().startswith(_PROXY_SCHEMES) and len(url) > len("http://")


def _ask_proxy() -> dict[str, str]:
    from ..infra.environment import detect_environment

    info = detect_environment()
    click.echo(f"\nEnvironment: {info.kind}" + (f" ({', '.join(info.markers)})" if info.markers else ""))
    for tip in info.suggestions:
        click.echo(f"  - {tip}")
    values: dict[str, str] = {}
    if not click.confirm("Configure a proxy?", default=info.is_server):
        return values
    while True:
        url = click.prompt(
            "  Global proxy URL for PROMPTURE_PROXY (blank for none)", default="", show_default=False
        ).strip()
        if not url or _valid_proxy(url):
            break
        click.echo("  Proxy URLs start with http://, https://, socks4://, socks5:// or socks5h://.")
    if url:
        values["PROMPTURE_PROXY"] = url
    while True:
        line = click.prompt(
            "  Per-backend proxy as backend=url (blank to finish)", default="", show_default=False
        ).strip()
        if not line:
            break
        backend, _, purl = line.partition("=")
        backend = backend.strip().upper().replace("-", "_")
        purl = purl.strip()
        if not backend or not backend.replace("_", "").isalnum() or not _valid_proxy(purl):
            click.echo("  Expected e.g. youtube=http://proxy.example:3128")
            continue
        values[f"PROMPTURE_{backend}_PROXY"] = purl
    return values


@click.command()
@click.option("--profile", default=None, help="Credential profile to write (default: PROMPTURE_PROFILE or 'default').")
@click.option("--dry-run", is_flag=True, help="Print the plan without writing anything.")
def setup(profile: str | None, dry_run: bool) -> None:
    """Interactive wizard: provider keys, default model, extras and proxy."""
    store = _store(profile)
    prof = store.profile
    click.echo(f"Prompture setup — profile '{prof}', store {store.path}")
    click.echo("Environment variables and .env always take precedence over stored values.")

    changes: dict[str, str] = {}
    chosen = _choose_providers(provider_key_catalogue(), store)
    configured: list[ProviderKey] = []
    for entry in chosen:
        key = _ask_key(entry)
        if key is not None:
            changes[entry.env_var] = key
            configured.append(entry)
    changes.update(_ask_default_model(configured))
    changes.update(_ask_extras())
    changes.update(_ask_proxy())

    if not changes:
        click.echo("\nNothing to save.")
        return
    _print_plan(changes, [], prof, store)
    if dry_run:
        click.echo("\nDry run: nothing was written.")
        return
    if not click.confirm("\nSave these settings?", default=True):
        click.echo("Nothing was written.")
        return
    try:
        store.set_many(changes, profile=prof)
    except CredentialStoreError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved {len(changes)} value(s) to profile '{prof}'.")


# ---------------------------------------------------------------------------
# prompture configure
# ---------------------------------------------------------------------------


def _list_store(store: CredentialStore, profile: str | None) -> None:
    try:
        data = store.load()
    except CredentialStoreError as exc:
        raise click.ClickException(str(exc)) from exc
    if not data:
        click.echo(f"No stored settings ({store.path} does not exist yet).")
        return
    current = active_profile()
    names = [profile] if profile else list(data)
    click.echo(f"Credential store: {store.path}")
    for name in names:
        section = data.get(name)
        marker = " (active)" if name == current else ""
        click.echo(f"[{name}]{marker}")
        if not section:
            click.echo("  (empty)")
            continue
        for key in sorted(section):
            note = "  (overridden by environment)" if _env_source(key) else ""
            click.echo(f"  {key} = {_display(key, section[key])}{note}")


@click.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option("--profile", default=None, help="Credential profile (default: PROMPTURE_PROFILE or 'default').")
@click.option("--unset", is_flag=True, help="Remove KEY from the profile.")
@click.option("--list", "list_", is_flag=True, help="List stored key names with masked values.")
@click.option("--dry-run", is_flag=True, help="Print the plan without writing anything.")
@click.option("--no-validate", is_flag=True, help="Store provider keys without a live check.")
def configure(
    key: str | None,
    value: str | None,
    profile: str | None,
    unset: bool,
    list_: bool,
    dry_run: bool,
    no_validate: bool,
) -> None:
    """Set, unset or list stored settings.

    \b
    prompture configure OPENAI_API_KEY sk-...      # store a key (validated live)
    prompture configure OPENAI_API_KEY             # prompt for it (hidden input)
    printf %s "$KEY" | prompture configure OPENAI_API_KEY -
    prompture configure PROMPTURE_PROXY http://proxy:3128 --profile work
    prompture configure OPENAI_API_KEY --unset
    prompture configure --list
    """
    store = _store(profile)
    if list_:
        _list_store(store, profile)
        return
    if not key:
        raise click.UsageError("KEY is required (or use --list).")
    name = key.strip().upper()
    prof = store.profile

    if unset:
        if value is not None:
            raise click.UsageError("--unset takes no VALUE.")
        try:
            present = name in store.list_keys(prof)
        except CredentialStoreError as exc:
            raise click.ClickException(str(exc)) from exc
        if not present:
            click.echo(f"{name} is not stored in profile '{prof}'.")
            return
        if dry_run:
            _print_plan({}, [name], prof, store)
            click.echo("\nDry run: nothing was written.")
            return
        store.unset(name, profile=prof)
        click.echo(f"Removed {name} from profile '{prof}'.")
        return

    if value is None:
        value = click.prompt(f"Value for {name}", hide_input=is_secret_name(name))
    elif value == "-":
        value = sys.stdin.readline().rstrip("\r\n")
    value = value.strip()
    if not value:
        raise click.UsageError("VALUE is empty; use --unset to remove a key.")

    if name not in _known_names() and not name.startswith("PROMPTURE_"):
        click.echo(f"Note: {name} is not a known Prompture setting; storing it anyway.")
    if name.endswith("_PROXY") and not _valid_proxy(value):
        raise click.UsageError("Proxy URLs start with http://, https://, socks4://, socks5:// or socks5h://.")

    if not no_validate:
        entry = next((e for e in provider_key_catalogue() if e.env_var == name), None)
        if entry is not None:
            check = validate_key(entry.provider, value)
            if check.status == "invalid":
                raise click.ClickException(
                    f"{entry.display} {check.detail}; nothing saved. Use --no-validate to store it anyway."
                )
            if check.status == "unverified":
                click.echo(f"Warning: could not verify the key: {check.detail}.")
            elif check.status == "valid":
                click.echo(f"{entry.display} key {check.detail}.")

    if dry_run:
        _print_plan({name: value}, [], prof, store)
        click.echo("\nDry run: nothing was written.")
        return
    try:
        store.set(name, value, profile=prof)
    except CredentialStoreError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Saved {name} = {_display(name, value)} to profile '{prof}' in {store.path}.")
    if _env_source(name):
        click.echo(f"Note: {name} is also set in the environment, which takes precedence.")


# ---------------------------------------------------------------------------
# prompture reset
# ---------------------------------------------------------------------------

_STATE_DESCRIPTIONS = {
    "credentials.yaml": "stored credentials and settings",
    "mcp.json": "named MCP servers",
    "usage": "usage ledger databases",
    "cache": "model-rate, response and plan-usage caches",
    "update_check.json": "update-check state",
    "companion.json": "local companion address and token",
    "companion-prefs.json": "companion preferences",
    "routes.json": "router routes",
    "router": "router call records",
    "memory": "agent and project memory",
    "automations": "companion automations",
    "checkpoints": "agent checkpoints",
    "kg": "knowledge-graph store",
    "conversations": "saved conversations",
    "session_memory": "session memory store",
    "workflows": "workflow run records",
}


def _link_target_inside(link: Path, root: Path) -> bool:
    try:
        target = Path(os.path.realpath(link))
    except OSError:
        return False
    return target == root or root in target.parents


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _outside_links(directory: Path, root: Path) -> list[Path]:
    found: list[Path] = []
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        for name in [*dirnames, *filenames]:
            candidate = Path(current) / name
            if _is_link(candidate) and not _link_target_inside(candidate, root):
                found.append(candidate)
    return found


@dataclass
class _ResetItem:
    path: Path
    kind: str  # "file" | "dir" | "link"
    description: str
    refused: str | None = None


def _plan_reset(state: Path, keep_credentials: bool) -> list[_ResetItem]:
    root = Path(os.path.realpath(state))
    items: list[_ResetItem] = []
    for child in sorted(state.iterdir(), key=lambda p: p.name.lower()):
        name = child.name
        if keep_credentials and name == "credentials.yaml":
            continue
        desc = _STATE_DESCRIPTIONS.get(name)
        if desc is None and name.startswith(".credentials.") and name.endswith(".tmp"):
            desc = "leftover credential temp file"
        desc = desc or "other Prompture state"
        if _is_link(child):
            item = _ResetItem(child, "link", desc)
            if not _link_target_inside(child, root):
                item.refused = "symlink points outside ~/.prompture"
        elif child.is_dir():
            item = _ResetItem(child, "dir", desc)
            outside = _outside_links(child, root)
            if outside:
                item.refused = f"contains a symlink pointing outside ~/.prompture ({outside[0].name})"
        else:
            item = _ResetItem(child, "file", desc)
        items.append(item)
    return items


def _force_remove(func, path, _exc) -> None:  # type: ignore[no-untyped-def]
    os.chmod(path, 0o700)
    func(path)


def _remove(item: _ResetItem) -> None:
    if item.kind == "link":
        if item.path.is_dir() and not item.path.is_symlink():
            os.rmdir(item.path)  # Windows junction: removes the link, not the target
        else:
            item.path.unlink()
    elif item.kind == "dir":
        if sys.version_info >= (3, 12):
            shutil.rmtree(item.path, onexc=_force_remove)
        else:  # pragma: no cover - older Pythons
            shutil.rmtree(item.path, onerror=_force_remove)
    else:
        try:
            item.path.unlink()
        except PermissionError:
            os.chmod(item.path, 0o600)
            item.path.unlink()


@click.command()
@click.option("--keep-credentials", is_flag=True, help="Keep ~/.prompture/credentials.yaml.")
@click.option("--dry-run", is_flag=True, help="Print what would be removed without removing anything.")
@click.option("--yes", "-y", is_flag=True, help="Do not ask for confirmation.")
def reset(keep_credentials: bool, dry_run: bool, yes: bool) -> None:
    """Remove Prompture's local state under ~/.prompture."""
    state = prompture_state_dir()
    if _is_link(state):
        raise click.ClickException(f"Refusing to reset: {state} is a symlink.")
    if not state.is_dir():
        click.echo(f"Nothing to reset: {state} does not exist.")
        return
    items = _plan_reset(state, keep_credentials)
    removable = [i for i in items if not i.refused]
    refused = [i for i in items if i.refused]

    if not items:
        click.echo(f"Nothing to reset in {state}.")
        return
    click.echo(f"Prompture state in {state}:")
    for item in items:
        suffix = "/" if item.kind == "dir" else (" (link)" if item.kind == "link" else "")
        action = f"skip: {item.refused}" if item.refused else "remove"
        click.echo(f"  {action:<8} {item.path.name}{suffix}  — {item.description}")
    if keep_credentials and (state / "credentials.yaml").exists():
        click.echo("  keep     credentials.yaml  — --keep-credentials")

    if dry_run:
        click.echo("\nDry run: nothing was removed.")
        return
    if not removable:
        click.echo("\nNothing removable.")
        raise SystemExit(1 if refused else 0)
    if not yes and not click.confirm(f"\nRemove {len(removable)} item(s)?", default=False):
        click.echo("Nothing was removed.")
        return

    failures = 0
    for item in removable:
        try:
            _remove(item)
        except OSError as exc:
            failures += 1
            click.echo(f"  failed: {item.path.name} ({type(exc).__name__}: {exc.strerror or exc})", err=True)
    if not keep_credentials and not refused and not failures:
        with contextlib.suppress(OSError):
            state.rmdir()
    click.echo(f"Removed {len(removable) - failures} item(s).")
    if refused or failures:
        raise SystemExit(1)


COMMANDS = [setup, configure, reset]
