"""Public-URL guard: keep tools from reaching private networks.

:func:`normalize_public_http_url` accepts only ``http``/``https`` URLs whose
host resolves exclusively to public addresses. It blocks private, loopback,
link-local (incl. cloud metadata), multicast, reserved and unspecified
IPv4/IPv6 ranges, IPv4-mapped / 6to4 / Teredo wrappers around them, and the
legacy numeric host spellings (``2130706433``, ``0177.0.0.1``, ``0x7f.1``)
that ``inet_aton`` still accepts.

Set ``PROMPTURE_WEB_ALLOW_PRIVATE=1`` (or pass ``allow_private=True``) to
reach local services on purpose.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from urllib.parse import urlsplit, urlunsplit

from .errors import UnsafeURLError

ALLOW_PRIVATE_ENV = "PROMPTURE_WEB_ALLOW_PRIVATE"

_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "instance-data.ec2.internal",
    }
)
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa", ".lan", ".intranet", ".corp")

_EXTRA_BLOCKED_NETS = (
    ipaddress.ip_network("100.64.0.0/10"),  # carrier-grade NAT
    ipaddress.ip_network("192.0.0.0/24"),
    ipaddress.ip_network("198.18.0.0/15"),  # benchmarking
    ipaddress.ip_network("fd00:ec2::/32"),  # AWS IPv6 metadata
)

_NUMERIC_PART_RE = re.compile(r"^(?:0x[0-9a-f]*|[0-9]+)$", re.IGNORECASE)


def allow_private_default() -> bool:
    """Return ``True`` when ``PROMPTURE_WEB_ALLOW_PRIVATE`` is set to a truthy value."""
    return os.environ.get(ALLOW_PRIVATE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def parse_legacy_ipv4(host: str) -> ipaddress.IPv4Address | None:
    """Parse ``inet_aton``-style IPv4 spellings (decimal, octal, hex, 1-4 parts).

    Returns ``None`` when *host* is not a numeric IPv4 spelling at all.
    """
    parts = host.split(".")
    if parts and parts[-1] == "":
        parts = parts[:-1]  # trailing dot
    if not 1 <= len(parts) <= 4 or not all(_NUMERIC_PART_RE.match(p) for p in parts):
        return None
    values: list[int] = []
    for p in parts:
        low = p.lower()
        if low.startswith("0x"):
            values.append(int(low[2:] or "0", 16))
        elif len(p) > 1 and p.startswith("0"):
            if not all(c in "01234567" for c in p):
                return None
            values.append(int(p, 8))
        else:
            values.append(int(p))
    # The last part fills the remaining bytes.
    head, last = values[:-1], values[-1]
    if any(v > 255 for v in head) or last >= 256 ** (4 - len(head)):
        return None
    number = 0
    for v in head:
        number = (number << 8) | v
    number = (number << (8 * (4 - len(head)))) | last
    return ipaddress.IPv4Address(number)


def _unwrap(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    out: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [ip]
    if isinstance(ip, ipaddress.IPv6Address):
        for embedded in (ip.ipv4_mapped, ip.sixtofour, ip.teredo[1] if ip.teredo else None):
            if embedded is not None:
                out.append(embedded)
        # NAT64 well-known prefix 64:ff9b::/96 embeds an IPv4 address.
        if ip in ipaddress.ip_network("64:ff9b::/96"):
            out.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return out


def is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address | str) -> bool:
    """Return ``True`` when *ip* (and anything it wraps) is a globally routable address."""
    if isinstance(ip, str):
        ip = ipaddress.ip_address(ip.split("%", 1)[0])
    for candidate in _unwrap(ip):
        if (
            not candidate.is_global
            or candidate.is_multicast
            or candidate.is_private
            or candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_reserved
            or candidate.is_unspecified
        ):
            return False
        if any(candidate.version == net.version and candidate in net for net in _EXTRA_BLOCKED_NETS):
            return False
    return True


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    bare = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        return ipaddress.ip_address(bare.split("%", 1)[0])
    except ValueError:
        return parse_legacy_ipv4(bare)


def resolve_host(host: str, port: int | None = None) -> list[str]:
    """Resolve *host* to all of its addresses (IPv4 and IPv6)."""
    infos = socket.getaddrinfo(host, port or 443, proto=socket.IPPROTO_TCP)
    return sorted({info[4][0] for info in infos})


def normalize_public_http_url(url: str, *, allow_private: bool | None = None, resolve: bool = True) -> str:
    """Validate *url* and return it normalized for fetching.

    Normalization lowercases the scheme and host, drops userinfo and the
    fragment, and IDNA-encodes the host. Raises :class:`UnsafeURLError`
    when the URL is not public ``http(s)``.

    Args:
        url: The candidate URL.
        allow_private: Permit private/loopback targets. ``None`` reads
            ``PROMPTURE_WEB_ALLOW_PRIVATE``.
        resolve: Resolve hostnames and check every address. Disable only
            when the caller resolves separately.
    """
    if allow_private is None:
        allow_private = allow_private_default()
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("empty URL")
    raw = url.strip()
    if "://" not in raw and not raw.lower().startswith(("http:", "https:")):
        raw = "https://" + raw
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError(f"malformed URL: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise UnsafeURLError(f"scheme {scheme!r} is not allowed (http/https only)")
    host = (parts.hostname or "").rstrip(".")
    if not host:
        raise UnsafeURLError("URL has no host")
    if any(c in host for c in " \t\r\n\\"):
        raise UnsafeURLError("host contains invalid characters")
    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise UnsafeURLError(f"host is not a valid domain name: {exc}") from exc

    literal = _literal_ip(ascii_host)
    if literal is not None:
        ascii_host = literal.compressed

    if not allow_private:
        if literal is not None:
            if not is_public_ip(literal):
                raise UnsafeURLError(f"{literal} is not a public address")
        else:
            if ascii_host in _BLOCKED_HOSTNAMES or ascii_host.endswith(_BLOCKED_SUFFIXES) or "." not in ascii_host:
                raise UnsafeURLError(f"host {ascii_host!r} is a local or internal name")
            if resolve:
                try:
                    addresses = resolve_host(ascii_host, port)
                except (OSError, UnicodeError) as exc:
                    raise UnsafeURLError(f"could not resolve host {ascii_host!r}: {exc}") from exc
                if not addresses:
                    raise UnsafeURLError(f"host {ascii_host!r} has no addresses")
                blocked = [a for a in addresses if not is_public_ip(a)]
                if blocked:
                    raise UnsafeURLError(f"host {ascii_host!r} resolves to non-public address {blocked[0]}")

    netloc = f"[{ascii_host}]" if ":" in ascii_host else ascii_host
    default_port = 443 if scheme == "https" else 80
    if port is not None and port != default_port:
        netloc = f"{netloc}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def is_public_http_url(url: str, *, allow_private: bool | None = None, resolve: bool = True) -> bool:
    """Boolean form of :func:`normalize_public_http_url`."""
    try:
        normalize_public_http_url(url, allow_private=allow_private, resolve=resolve)
    except UnsafeURLError:
        return False
    return True
