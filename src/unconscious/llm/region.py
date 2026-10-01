"""Where the crew would sail from.

Anthropic and OpenAI do not serve mainland China. While the connection appears to
be there, Claude Code, Codex, the Anthropic API and OpenAI stay ashore: nothing at
all is sent to them until the connection is somewhere else. DeepSeek, GLM, Kimi and
local models are not affected.

The check asks small public endpoints which country the connection comes from. It
goes through the proxy settings in the environment, the same ones the crew's
command-line tools inherit, so it sees the exit an errand would take: a VPN or a
proxy that carries the crew abroad is respected. It runs only in background threads,
right before an errand or when a dive is due, and the answer is kept for a few
minutes. If no endpoint answers, the crew waits: being unsure is not being abroad.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

GUARDED = frozenset({"claude", "codex", "anthropic", "openai"})
UNKNOWN = "?"
TIMEOUT = 4.0


def _trace(text: str) -> str | None:
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "loc" and value.strip():
            return value.strip()
    return None


def _country_is(text: str) -> str | None:
    return (json.loads(text) or {}).get("country") or None


def _plain(text: str) -> str | None:
    value = text.strip()
    return value if len(value) == 2 and value.isalpha() else None


ENDPOINTS: tuple[tuple[str, Callable[[str], str | None]], ...] = (
    ("https://www.cloudflare.com/cdn-cgi/trace", _trace),
    ("https://api.country.is/", _country_is),
    ("https://ipinfo.io/country", _plain),
)


def lookup() -> str | None:
    """The country code the connection appears from, or None if nobody answered."""
    # Environment proxies only: the CLIs inherit these and not macOS's system proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(urllib.request.getproxies_environment()))
    for url, parse in ENDPOINTS:
        try:
            with opener.open(urllib.request.Request(url, headers={"User-Agent": "digital-unconscious"}), timeout=TIMEOUT) as reply:
                country = parse(reply.read(4096).decode("utf-8", "replace"))
            if country:
                return country.upper()
        except Exception as exc:  # try the next one
            log.debug("region lookup via %s failed: %s", url, exc)
    return None


@dataclass
class Fix:
    country: str  # ISO code, or UNKNOWN
    at: float  # time.time()


class RegionCheck:
    """A cached answer to "where is the connection?". ``current`` may touch the network
    and belongs in background threads; ``last`` never does and is safe anywhere."""

    def __init__(self, fetch: Callable[[], str | None] | None = None):
        self.fetch = fetch or lookup
        self.held: frozenset[str] = frozenset({"CN"})  # kept in step with the settings by the router
        self._fix: Fix | None = None
        self._lock = threading.Lock()

    def current(self, max_age: float | None = None) -> str:
        """With no max_age, an answer is kept 5 minutes, a held one 10 (the crew is waiting
        anyway) and "could not tell" 2 (so the crew is back soon after the network is).
        Errands pass max_age=0: a switched connection is always seen before anything is sent."""
        with self._lock:
            if self._fix is None or time.time() - self._fix.at > (self._keep(self._fix) if max_age is None else max_age):
                country = self.fetch()
                self._fix = Fix(country.upper() if country else UNKNOWN, time.time())
                log.info("connection appears to be in %s", self._fix.country)
            return self._fix.country

    def last(self) -> Fix | None:
        return self._fix

    def _keep(self, fix: Fix) -> float:
        if fix.country == UNKNOWN:
            return 120
        return 600 if fix.country in self.held else 300


region_check = RegionCheck()
