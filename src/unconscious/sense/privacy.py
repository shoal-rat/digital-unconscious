"""What never reaches disk, and what gets blurred before it does.

Filtering happens at capture time, before a sample becomes a trace, so a
quiet app or a private-browsing window leaves no record at all.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

PRIVATE_WINDOW_MARKERS = (
    "private browsing", "inprivate", "incognito", "无痕", "隐私浏览", "隐身",
)

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_LONG_NUMBER = re.compile(r"\b\d{6,}\b")
_TOKEN = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{24,}\b")
_ID_SEGMENT = re.compile(r"^(?=.*\d)[A-Za-z0-9_-]{20,}$|^[0-9a-f]{8}-[0-9a-f-]{27,}$", re.I)


def _matches(value: str, patterns: list[str]) -> bool:
    lowered = value.casefold()
    return any(p and p.casefold() in lowered for p in patterns)


def is_quiet(app: str, domain: str, title: str, quiet_apps: list[str], quiet_domains: list[str]) -> bool:
    """True when the sample must be dropped entirely."""
    if app and _matches(app, quiet_apps):
        return True
    if domain and _matches(domain, quiet_domains):
        return True
    lowered = title.casefold()
    return any(marker in lowered for marker in PRIVATE_WINDOW_MARKERS)


def is_private_app(app: str, private_apps: list[str]) -> bool:
    return bool(app) and _matches(app, private_apps)


def redact(text: str) -> str:
    """Blur identifiers that have no value for finding ideas."""
    if not text:
        return ""
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub("[number]", text)
    text = _LONG_NUMBER.sub("[number]", text)
    return _TOKEN.sub("[id]", text)


def clean_url(url: str) -> str:
    """Keep scheme, host and a sanitised path; drop credentials, query and fragment."""
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return ""
    if parts.scheme not in {"http", "https"}:
        return ""
    host = (parts.hostname or "").lower()
    if parts.port:
        host = f"{host}:{parts.port}"
    segments = [":id" if _ID_SEGMENT.match(seg) else seg for seg in parts.path.split("/")]
    path = "/".join(segments)[:200]
    return urlunsplit((parts.scheme, host, path, "", ""))


def domain_of(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host
