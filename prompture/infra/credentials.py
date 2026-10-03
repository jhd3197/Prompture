"""Private, profile-aware credential store at ``~/.prompture/credentials.yaml``.

The store keeps provider keys and Prompture settings out of hand-edited
``.env`` files. It is the lowest-priority configuration source: real
environment variables and ``.env`` always win (see
:class:`prompture.infra.settings.Settings` and :func:`get_config_value`).

On disk it is a two-level YAML mapping — profile → ``KEY: value``::

    default:
      OPENAI_API_KEY: "sk-..."
      PROMPTURE_PROXY: "http://proxy.internal:3128"
    work:
      OPENAI_API_KEY: "sk-..."

Safety properties:

* the file and ``~/.prompture`` are created only on the first write;
* the file is owner-only — mode ``0600`` on POSIX, an ACL granting only the
  current user on Windows (applied with ``icacls`` argv, never a shell);
* writes are atomic (temp file beside the target, then ``os.replace``);
* symlinked (or junctioned) targets and parent directories are refused;
* reads are capped at 1 MB.

Reads use PyYAML when it is importable (so hand edits in any YAML style work)
and a minimal parser for exactly this flat two-level shape otherwise. Writes
always emit one ``"KEY": "value"`` line per entry with JSON-style double-quoted
scalars, which is valid YAML and readable by both paths.

The active profile is ``PROMPTURE_PROFILE`` (default ``"default"``). Lookups
in a non-default profile fall back to the ``default`` profile for keys the
profile does not define, so shared values (a proxy, say) live in one place.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised implicitly when PyYAML is installed
    import yaml as _yaml
except Exception:  # pragma: no cover - fallback path is tested by patching _yaml
    _yaml = None

logger = logging.getLogger("prompture.credentials")

__all__ = [
    "DEFAULT_PROFILE",
    "MAX_STORE_BYTES",
    "PROFILE_ENV_VAR",
    "CredentialStore",
    "CredentialStoreError",
    "active_profile",
    "credentials_path",
    "get_config_value",
    "is_secret_name",
    "mask_value",
    "prompture_state_dir",
]

DEFAULT_PROFILE = "default"
PROFILE_ENV_VAR = "PROMPTURE_PROFILE"
MAX_STORE_BYTES = 1024 * 1024
CREDENTIALS_FILENAME = "credentials.yaml"

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "PROXY", "AUTH", "COOKIE", "SESSION")


class CredentialStoreError(Exception):
    """The store is unreadable, unsafe or malformed, or an argument is invalid."""


# ---------------------------------------------------------------------------
# Paths and profile
# ---------------------------------------------------------------------------


def prompture_state_dir() -> Path:
    """``~/.prompture`` — resolved on every call so tests can move ``HOME``."""
    return Path.home() / ".prompture"


def credentials_path() -> Path:
    """Default location of the credential store."""
    return prompture_state_dir() / CREDENTIALS_FILENAME


def active_profile(profile: str | None = None) -> str:
    """Explicit *profile*, else ``PROMPTURE_PROFILE``, else ``"default"``."""
    chosen = (profile or os.environ.get(PROFILE_ENV_VAR) or "").strip() or DEFAULT_PROFILE
    return _check_profile(chosen)


def _check_profile(name: str) -> str:
    if not isinstance(name, str) or not _PROFILE_RE.match(name):
        raise CredentialStoreError(
            f"Invalid profile name {name!r}: use letters, digits, '_', '-' or '.' (max 64 chars)."
        )
    return name


def _check_key(name: str) -> str:
    if not isinstance(name, str) or not _KEY_RE.match(name):
        raise CredentialStoreError(
            f"Invalid key name {name!r}: use an environment-variable style name such as OPENAI_API_KEY."
        )
    return name.upper()


def is_secret_name(name: str) -> bool:
    """Whether a key name looks like it holds a secret (or a URL that may embed one)."""
    upper = name.upper()
    return any(hint in upper for hint in _SECRET_HINTS)


def mask_value(value: str | None, *, secret: bool = True) -> str:
    """Render *value* for display without revealing a secret.

    Long secrets keep a short prefix and the last four characters; short ones
    are fully masked. Non-secret values are returned unchanged.
    """
    if value is None:
        return ""
    if not secret:
        return value
    if len(value) >= 16:
        return f"{value[:3]}…{value[-4:]}"
    if len(value) >= 8:
        return f"…{value[-2:]}"
    return "****"


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


# ---------------------------------------------------------------------------
# Minimal YAML subset (fallback when PyYAML is missing)
# ---------------------------------------------------------------------------


def _parse_scalar(text: str, lineno: int) -> tuple[str | None, str]:
    """Parse one scalar at the start of *text*; return ``(value, rest)``.

    Supports double-quoted (JSON-compatible escapes), single-quoted (``''``
    escape) and plain scalars. A plain scalar ends at ``": "`` / trailing
    ``":"`` or a `` #`` comment. Empty plain scalars are ``None`` (YAML null).
    """
    if text.startswith('"'):
        i = 1
        while i < len(text):
            ch = text[i]
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                break
            i += 1
        else:
            raise CredentialStoreError(f"credentials.yaml line {lineno}: unterminated double-quoted string")
        try:
            value = json.loads(text[: i + 1])
        except ValueError as exc:
            raise CredentialStoreError(f"credentials.yaml line {lineno}: bad escape in quoted string") from exc
        return value, text[i + 1 :]
    if text.startswith("'"):
        out: list[str] = []
        i = 1
        while i < len(text):
            ch = text[i]
            if ch == "'":
                if text[i + 1 : i + 2] == "'":
                    out.append("'")
                    i += 2
                    continue
                return "".join(out), text[i + 1 :]
            out.append(ch)
            i += 1
        raise CredentialStoreError(f"credentials.yaml line {lineno}: unterminated single-quoted string")
    # Plain scalar: stop at a mapping indicator or a comment.
    end = len(text)
    for marker in (": ", " #"):
        pos = text.find(marker)
        if pos != -1:
            end = min(end, pos)
    if text.endswith(":") and end == len(text):
        end = len(text) - 1
    value = text[:end].strip()
    if value in ("", "~", "null", "Null", "NULL"):
        return None, text[end:]
    return value, text[end:]


def _strip_comment(rest: str, lineno: int) -> None:
    rest = rest.strip()
    if rest and not rest.startswith("#"):
        raise CredentialStoreError(f"credentials.yaml line {lineno}: unexpected text after value")


def _parse_minimal(text: str) -> dict[str, dict[str, str]]:
    """Parse the flat ``profile: {KEY: value}`` YAML subset this module writes."""
    data: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped in ("---", "..."):
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise CredentialStoreError(f"credentials.yaml line {lineno}: tabs are not allowed for indentation")
        indented = line[0] in " "
        key, rest = _parse_scalar(stripped, lineno)
        if key is None or not rest.startswith(":"):
            raise CredentialStoreError(f"credentials.yaml line {lineno}: expected 'name: value'")
        rest = rest[1:]
        if not indented:
            tail = rest.strip()
            if tail.startswith("{}"):
                _strip_comment(tail[2:], lineno)
            else:
                _strip_comment(tail, lineno)
            current = data.setdefault(str(key), {})
            continue
        if current is None:
            raise CredentialStoreError(f"credentials.yaml line {lineno}: key outside of a profile section")
        rest = rest.lstrip()
        value, tail = _parse_scalar(rest, lineno) if rest else (None, "")
        _strip_comment(tail, lineno)
        if value is not None:
            current[str(key)] = value
    return data


def _dump_minimal(data: Mapping[str, Mapping[str, str]]) -> str:
    """Write *data* as YAML with every scalar double-quoted (JSON escapes)."""
    lines = ["# Prompture credential store. Owner-only; managed by `prompture configure`."]
    for profile, values in data.items():
        if values:
            lines.append(f"{json.dumps(profile, ensure_ascii=False)}:")
            for key, value in values.items():
                lines.append(f"  {json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}")
        else:
            lines.append(f"{json.dumps(profile, ensure_ascii=False)}: {{}}")
    return "\n".join(lines) + "\n"


def _normalize_loaded(raw: Any) -> dict[str, dict[str, str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise CredentialStoreError("credentials.yaml must be a mapping of profile names to key/value maps")
    data: dict[str, dict[str, str]] = {}
    for profile, values in raw.items():
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise CredentialStoreError(f"credentials.yaml profile {profile!r} must be a mapping")
        section: dict[str, str] = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, (dict, list)):
                raise CredentialStoreError(f"credentials.yaml key {key!r} must be a scalar value")
            section[str(key).upper()] = str(value)
        data[str(profile)] = section
    return data


# ---------------------------------------------------------------------------
# Owner-only permissions
# ---------------------------------------------------------------------------


def _windows_user_sid() -> str | None:
    """String SID of the current process user via the Win32 API (``None`` on failure)."""
    try:
        import ctypes
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        token = wintypes.HANDLE()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):  # TOKEN_QUERY
            return None
        try:
            size = wintypes.DWORD(0)
            advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))  # TokenUser
            buf = ctypes.create_string_buffer(size.value)
            if not advapi32.GetTokenInformation(token, 1, buf, size, ctypes.byref(size)):
                return None
            psid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]
            sid_str = wintypes.LPWSTR()
            advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
            if not advapi32.ConvertSidToStringSidW(psid, ctypes.byref(sid_str)):
                return None
            try:
                return sid_str.value
            finally:
                kernel32.LocalFree(sid_str)
        finally:
            kernel32.CloseHandle(token)
    except Exception:
        return None


def _icacls_executable() -> str:
    system_root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or r"C:\Windows"
    candidate = Path(system_root) / "System32" / "icacls.exe"
    return str(candidate) if candidate.exists() else "icacls"


def _restrict_windows(path: Path) -> None:
    """Replace the ACL of *path* with a single full-control ACE for the current user."""
    sid = _windows_user_sid()
    if sid:
        principal = f"*{sid}"
    else:
        user = os.environ.get("USERNAME")
        if not user:
            raise CredentialStoreError("Cannot determine the current Windows user to restrict credentials.yaml.")
        domain = os.environ.get("USERDOMAIN")
        principal = f"{domain}\\{user}" if domain else user
    argv = [_icacls_executable(), str(path), "/inheritance:r", "/grant:r", f"{principal}:(F)"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CredentialStoreError(f"Could not restrict credentials file permissions: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        raise CredentialStoreError(
            f"Could not restrict credentials file permissions (icacls exit {result.returncode}): "
            f"{detail[0] if detail else 'no output'}"
        )


def _restrict_permissions(path: Path) -> None:
    if sys.platform == "win32":
        _restrict_windows(path)
    else:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_read_cache: dict[str, tuple[tuple[int, int], dict[str, dict[str, str]]]] = {}


class CredentialStore:
    """Read and write the profile-sectioned credential file.

    Args:
        path: Store location (default ``~/.prompture/credentials.yaml``,
            resolved when the store is created).
        profile: Profile used when a method gets ``profile=None`` (default:
            :func:`active_profile`, i.e. ``PROMPTURE_PROFILE`` or ``default``).
    """

    def __init__(self, path: str | os.PathLike[str] | None = None, *, profile: str | None = None) -> None:
        self.path = Path(path) if path is not None else credentials_path()
        self._profile = _check_profile(profile) if profile else None

    # -- helpers ---------------------------------------------------------

    @property
    def profile(self) -> str:
        """Profile used when none is passed explicitly."""
        return self._profile or active_profile()

    def _resolve_profile(self, profile: str | None) -> str:
        return _check_profile(profile) if profile else self.profile

    def exists(self) -> bool:
        return self.path.is_file()

    def _check_paths(self) -> None:
        if _is_link(self.path):
            raise CredentialStoreError(f"Refusing to use symlinked credentials file: {self.path}")
        if _is_link(self.path.parent):
            raise CredentialStoreError(f"Refusing to use symlinked credentials directory: {self.path.parent}")

    # -- reading ---------------------------------------------------------

    def load(self) -> dict[str, dict[str, str]]:
        """Whole store as ``{profile: {KEY: value}}`` (empty when the file is missing)."""
        self._check_paths()
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return {}
        if not stat.S_ISREG(st.st_mode):
            raise CredentialStoreError(f"Credentials path is not a regular file: {self.path}")
        if st.st_size > MAX_STORE_BYTES:
            raise CredentialStoreError(f"credentials.yaml is larger than {MAX_STORE_BYTES} bytes; refusing to read it")
        cache_key = str(self.path)
        signature = (st.st_mtime_ns, st.st_size)
        with _cache_lock:
            cached = _read_cache.get(cache_key)
            if cached and cached[0] == signature:
                return {p: dict(v) for p, v in cached[1].items()}
        with open(self.path, "rb") as fh:
            raw = fh.read(MAX_STORE_BYTES + 1)
        if len(raw) > MAX_STORE_BYTES:
            raise CredentialStoreError(f"credentials.yaml is larger than {MAX_STORE_BYTES} bytes; refusing to read it")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise CredentialStoreError("credentials.yaml is not valid UTF-8") from exc
        data = self._parse(text)
        with _cache_lock:
            _read_cache[cache_key] = (signature, {p: dict(v) for p, v in data.items()})
        return data

    @staticmethod
    def _parse(text: str) -> dict[str, dict[str, str]]:
        if _yaml is not None:
            try:
                raw = _yaml.safe_load(text)
            except Exception as exc:
                raise CredentialStoreError(f"credentials.yaml is not valid YAML: {type(exc).__name__}") from exc
            return _normalize_loaded(raw)
        return _normalize_loaded(_parse_minimal(text))

    def get(self, name: str, profile: str | None = None, *, inherit: bool = True) -> str | None:
        """Value of *name* in *profile* (falling back to ``default`` when *inherit*)."""
        key = _check_key(name)
        prof = self._resolve_profile(profile)
        data = self.load()
        value = data.get(prof, {}).get(key)
        if value is None and inherit and prof != DEFAULT_PROFILE:
            value = data.get(DEFAULT_PROFILE, {}).get(key)
        return value

    def effective(self, profile: str | None = None) -> dict[str, str]:
        """Merged ``default`` + *profile* values (the profile wins)."""
        prof = self._resolve_profile(profile)
        data = self.load()
        merged = dict(data.get(DEFAULT_PROFILE, {}))
        if prof != DEFAULT_PROFILE:
            merged.update(data.get(prof, {}))
        return merged

    def list_keys(self, profile: str | None = None, *, with_values: bool = False) -> list[str] | dict[str, str]:
        """Key names stored in *profile* (no inheritance). Values only when *with_values*."""
        prof = self._resolve_profile(profile)
        section = self.load().get(prof, {})
        if with_values:
            return dict(section)
        return sorted(section)

    def profiles(self) -> list[str]:
        """Profile names present in the store."""
        return list(self.load())

    # -- writing ---------------------------------------------------------

    def set(self, name: str, value: str, profile: str | None = None) -> None:
        """Store *value* under *name* in *profile*; creates the file on first write."""
        key = _check_key(name)
        if not isinstance(value, str):
            raise CredentialStoreError("Credential values must be strings")
        if "\x00" in value:
            raise CredentialStoreError("Credential values must not contain NUL characters")
        prof = self._resolve_profile(profile)
        data = self.load()
        data.setdefault(prof, {})[key] = value
        self._write(data)

    def set_many(self, values: Mapping[str, str], profile: str | None = None) -> None:
        """Store several values in *profile* with a single atomic write."""
        prof = self._resolve_profile(profile)
        checked: dict[str, str] = {}
        for name, value in values.items():
            key = _check_key(name)
            if not isinstance(value, str) or "\x00" in value:
                raise CredentialStoreError("Credential values must be strings without NUL characters")
            checked[key] = value
        if not checked:
            return
        data = self.load()
        data.setdefault(prof, {}).update(checked)
        self._write(data)

    def unset(self, name: str, profile: str | None = None) -> bool:
        """Remove *name* from *profile*; returns whether something was removed."""
        key = _check_key(name)
        prof = self._resolve_profile(profile)
        data = self.load()
        section = data.get(prof)
        if not section or key not in section:
            return False
        del section[key]
        if not section and prof != DEFAULT_PROFILE:
            del data[prof]
        self._write(data)
        return True

    def delete_profile(self, profile: str) -> bool:
        """Drop a whole profile section; returns whether it existed."""
        prof = _check_profile(profile)
        data = self.load()
        if prof not in data:
            return False
        del data[prof]
        self._write(data)
        return True

    @staticmethod
    def _serialize(data: Mapping[str, Mapping[str, str]]) -> str:
        # Always the single-line, double-quoted form: valid YAML for PyYAML and
        # readable by the built-in fallback parser alike.
        return _dump_minimal(data)

    def _write(self, data: Mapping[str, Mapping[str, str]]) -> None:
        self._check_paths()
        payload = self._serialize(data).encode("utf-8")
        if len(payload) > MAX_STORE_BYTES:
            raise CredentialStoreError(f"credentials.yaml would exceed {MAX_STORE_BYTES} bytes")
        parent = self.path.parent
        if not parent.exists():
            parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._check_paths()
        fd, tmp_name = tempfile.mkstemp(prefix=".credentials.", suffix=".tmp", dir=str(parent))
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            _restrict_permissions(tmp)
            self._check_paths()
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        with _cache_lock:
            _read_cache.pop(str(self.path), None)


# ---------------------------------------------------------------------------
# Lookup helper
# ---------------------------------------------------------------------------


def get_config_value(name: str, default: str | None = None, *, profile: str | None = None) -> str | None:
    """Environment variable *name* if set and non-blank, else the credential store, else *default*.

    The store lookup uses *profile* or the active profile (``PROMPTURE_PROFILE``)
    with fallback to ``default``. A missing, unreadable or unsafe store never
    raises here; it logs a warning and yields *default*.
    """
    env_value = os.environ.get(name)
    if env_value is not None and env_value.strip():
        return env_value
    if not _KEY_RE.match(name or ""):
        return default
    try:
        value = CredentialStore(profile=profile).get(name)
    except CredentialStoreError as exc:
        logger.warning("Ignoring credential store: %s", exc)
        return default
    except OSError as exc:
        logger.warning("Could not read credential store: %s", type(exc).__name__)
        return default
    return value if value is not None else default
