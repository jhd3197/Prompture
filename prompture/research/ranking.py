"""URL normalization, de-duplication and ranking of search hits.

Hits from every phrasing and platform for a sub-question are merged by
normalized URL, scored with reciprocal-rank fusion (a page that several
phrasings agree on rises), and selected with a per-domain cap so one site
can't fill every slot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

#: Query parameters that only track the click and never change the page.
TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "gbraid",
        "wbraid",
        "msclkid",
        "yclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "_hsenc",
        "_hsmi",
        "mkt_tok",
        "ref_src",
        "ref_url",
        "referrer",
        "spm",
        "si",
        "cmpid",
        "ito",
        "ncid",
        "sr_share",
        "trk",
        "trkcampaign",
        "s_cid",
        "oly_enc_id",
        "oly_anon_id",
        "vero_id",
        "_ga",
        "_gl",
    }
)
_TRACKING_PREFIXES = ("utm_", "pk_", "hsa_", "at_")

_DEFAULT_PORTS = {"http": "80", "https": "443"}
_YT_ID = re.compile(r"^[A-Za-z0-9_-]{6,}$")


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k in TRACKING_PARAMS or k.startswith(_TRACKING_PREFIXES)


def normalize_url(url: str) -> str:
    """Return a canonical form of *url* for de-duplication.

    Lower-cases scheme and host, drops ``www.``/``m.``, default ports,
    fragments, tracking parameters and trailing slashes, sorts the query,
    upgrades ``http`` to ``https`` for comparison, and maps ``youtu.be/ID``
    and ``/shorts/ID`` to ``youtube.com/watch?v=ID``. Non-HTTP strings are
    returned stripped but otherwise untouched.
    """
    raw = (url or "").strip()
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not parts.netloc:
        return raw
    host = (parts.hostname or "").lower().rstrip(".")
    for prefix in ("www.", "m.", "mobile."):
        if host.startswith(prefix) and host.count(".") >= 2:
            host = host[len(prefix) :]
    port = parts.port if parts.port is not None else None
    netloc = host if port is None or str(port) == _DEFAULT_PORTS.get(scheme) else f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", parts.path or "/")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)]

    if host == "youtu.be":
        vid = path.strip("/").split("/")[0]
        if _YT_ID.match(vid):
            netloc, path, query = "youtube.com", "/watch", [("v", vid)]
    elif host.endswith("youtube.com"):
        netloc = "youtube.com"
        m = re.match(r"^/(?:shorts|embed|live)/([A-Za-z0-9_-]{6,})", path)
        if m:
            path, query = "/watch", [("v", m.group(1))]
        elif path == "/watch":
            query = [(k, v) for k, v in query if k == "v"]

    if len(path) > 1:
        path = path.rstrip("/")
    query.sort()
    return urlunsplit(("https", netloc, path, urlencode(query, doseq=True), ""))


def strip_tracking(url: str) -> str:
    """Remove tracking parameters and the fragment from *url*, leaving everything else as-is."""
    raw = (url or "").strip()
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return raw
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), ""))


def domain_of(url: str) -> str:
    """Registrable-ish domain used for diversity (``docs.python.org`` → ``python.org``)."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    # Keep three labels for two-letter country second levels (bbc.co.uk).
    if len(labels[-1]) == 2 and labels[-2] in {"co", "com", "org", "net", "ac", "gov", "edu"}:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


@dataclass
class Candidate:
    """A unique URL gathered for one or more sub-questions.

    Attributes:
        url: First URL seen (as returned by the backend).
        key: Normalized URL used for de-duplication.
        title: Best title seen.
        snippet: Longest snippet seen.
        score: Fused ranking score (higher is better).
        hits: How many searches returned this URL.
        origins: Where it came from (``web``, ``github``, ``hackernews``, ...).
        queries: Queries that surfaced it.
        sub_questions: Indexes of the sub-questions it was found for.
        extra: Provider extras from the first hit.
    """

    url: str
    key: str
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    hits: int = 0
    origins: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    sub_questions: set[int] = field(default_factory=set)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return domain_of(self.key)


class CandidatePool:
    """Merges hits from many searches into de-duplicated, scored candidates."""

    def __init__(self, *, rrf_k: int = 60) -> None:
        self._by_key: dict[str, Candidate] = {}
        self._rrf_k = rrf_k

    def __len__(self) -> int:
        return len(self._by_key)

    def add(
        self,
        result: Any,
        *,
        rank: int,
        sub_question: int,
        query: str,
        origin: str = "web",
    ) -> Candidate | None:
        """Merge one search hit (``.url``, ``.title``, ``.snippet``, ``.score``). Ranks start at 0."""
        url = str(_get(result, "url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return None
        key = normalize_url(url)
        cand = self._by_key.get(key)
        if cand is None:
            cand = Candidate(url=strip_tracking(url), key=key, extra=dict(_get(result, "extra") or {}))
            self._by_key[key] = cand
        title = str(_get(result, "title") or "")
        snippet = str(_get(result, "snippet") or "")
        if title and not cand.title:
            cand.title = title
        if len(snippet) > len(cand.snippet):
            cand.snippet = snippet
        cand.hits += 1
        cand.score += 1.0 / (self._rrf_k + rank + 1)
        provider_score = _get(result, "score")
        if isinstance(provider_score, (int, float)) and 0 < provider_score <= 1:
            cand.score += 0.002 * float(provider_score)
        if origin not in cand.origins:
            cand.origins.append(origin)
        if query and query not in cand.queries:
            cand.queries.append(query)
        cand.sub_questions.add(sub_question)
        return cand

    def all(self) -> list[Candidate]:
        return sorted(self._by_key.values(), key=lambda c: (-c.score, c.key))

    def for_sub_question(self, index: int) -> list[Candidate]:
        return [c for c in self.all() if index in c.sub_questions]


def select_diverse(
    candidates: list[Candidate],
    limit: int,
    *,
    max_per_domain: int = 2,
    exclude: set[str] | None = None,
    domain_counts: dict[str, int] | None = None,
) -> list[Candidate]:
    """Pick up to *limit* candidates in score order, at most *max_per_domain* per domain.

    *exclude* holds normalized keys already chosen elsewhere; *domain_counts*
    carries per-domain usage across calls. When the cap leaves slots empty,
    capped candidates fill them in score order rather than returning short.
    """
    exclude = exclude if exclude is not None else set()
    counts = domain_counts if domain_counts is not None else {}
    ranked = sorted(candidates, key=lambda c: (-c.score, c.key))
    chosen: list[Candidate] = []
    overflow: list[Candidate] = []
    for cand in ranked:
        if len(chosen) >= limit:
            break
        if cand.key in exclude:
            continue
        if counts.get(cand.domain, 0) >= max_per_domain:
            overflow.append(cand)
            continue
        chosen.append(cand)
        counts[cand.domain] = counts.get(cand.domain, 0) + 1
    for cand in overflow:
        if len(chosen) >= limit:
            break
        chosen.append(cand)
        counts[cand.domain] = counts.get(cand.domain, 0) + 1
    return chosen


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


__all__ = [
    "TRACKING_PARAMS",
    "Candidate",
    "CandidatePool",
    "domain_of",
    "normalize_url",
    "select_diverse",
    "strip_tracking",
]
