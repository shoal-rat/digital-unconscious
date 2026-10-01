"""Who can sail right now, and why not.

A crew member can be aboard (installed, signed in once) and still unable to sail for a
while. Each reason has its own way back, and none of them counts as a failed dive: the
night's attempts are kept for real failures, such as an answer that cannot be used.

    region   the connection is in a held region (region.py), or  looked up again before every errand
             the service itself refused the region
    sign_in  the CLI says it must be signed in again            `claude auth status` / `codex login
                                                                status` tell when it is back
    limit    a usage or rate limit                              tried again after a pause
    offline  no network, a timeout, the service unreachable     tried again after a pause

Pauses grow with each strike (5, 10, 20 minutes… at most two hours), so a long outage
costs a few quick checks rather than a stream of calls, and fallback hands the errand
to the next crew member in the meantime. Troubles are written to the store, so the
shore and the menu-bar mark can tell the person what is wrong without asking anyone.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unconscious.store import Store

log = logging.getLogger(__name__)

REGION, SIGN_IN, LIMIT, OFFLINE = "region", "sign_in", "limit", "offline"
MARK = "crew:"  # errors meaning "no one could sail": crew:<kind>:<who>:<detail>
PAUSE = {REGION: 600, SIGN_IN: 300, LIMIT: 1800, OFFLINE: 300}
LONGEST = 7200

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # the service itself refusing the region: the guard's lookup and the errand's exit disagreed
    (REGION, re.compile(
        r"unsupported (country|region|location)|not (available|supported) in your (country|region)"
        r"|country, region,? or territory|request not allowed", re.I)),
    (SIGN_IN, re.compile(
        r"/login|not logged in|log ?in again|sign ?in again|please (run|sign|log)|unauthori[sz]ed|\b401\b"
        r"|invalid (api |x-api-)?key|authenticat|oauth|token (has )?expired|expired token|credentials", re.I)),
    (LIMIT, re.compile(
        r"rate.?limit|usage limit|limit (reached|exceeded)|\b429\b|too many requests|quota|overloaded|\b529\b"
        r"|credit balance|resets? at", re.I)),
    (OFFLINE, re.compile(
        r"getaddrinfo|enotfound|econnrefused|econnreset|eai_again|etimedout|enetunreach|network|timed? ?out"
        r"|connection (error|refused|reset|closed|failed)|unable to connect|could not (resolve|connect)|offline"
        r"|no route to host|temporarily unavailable|\b50[234]\b|ssl|certificate", re.I)),
)


def classify(error: str) -> str | None:
    """Which kind of trouble an errand's error means, or None for a real failure."""
    for kind, pattern in PATTERNS:
        if pattern.search(error or ""):
            return kind
    return None


def mark(kind: str, who: str, detail: str = "") -> str:
    return f"{MARK}{kind}:{who}:{detail}"


def parse(error: str) -> tuple[str, str, str] | None:
    """(kind, who, detail) from a crew mark found anywhere in an error message."""
    at = (error or "").find(MARK)
    if at < 0:
        return None
    kind, who, detail = (error[at + len(MARK):].split(":", 2) + ["", ""])[:3]
    return (kind, who, detail) if kind else None


@dataclass
class Trouble:
    kind: str
    detail: str
    since: float
    until: float
    strikes: int = 1


class Crew:
    """Troubles per crew member, kept in memory and mirrored to the store's ``crew`` key."""

    def __init__(self, store: Store | None = None, clock=time.time):
        self.store = store
        self.clock = clock
        self.troubles: dict[str, Trouble] = {}
        if store is not None:  # pauses outlive a restart, and every process sees the same ones
            try:
                for name, saved in (store.get("crew") or {}).items():
                    self.troubles[name] = Trouble(**saved)
            except Exception:
                log.debug("could not read crew state", exc_info=True)

    def blocked(self, name: str, provider: Any = None) -> Trouble | None:
        """The trouble keeping this crew member in port right now, if any. Once a sign-in
        pause is over, the CLI is asked whether it is signed in again (a process: background only)."""
        trouble = self.troubles.get(name)
        if trouble is None:
            return None
        if self.clock() < trouble.until:
            return trouble
        if trouble.kind == SIGN_IN and provider is not None and hasattr(provider, "signed_in"):
            if provider.signed_in() is False:
                self._strike(name, SIGN_IN, trouble.detail)
                return self.troubles[name]
        return None  # the pause is over: let the next errand find out

    def failed(self, name: str, error: str) -> str | None:
        kind = classify(error)
        if kind:
            self._strike(name, kind, error[-300:])
        return kind

    def sailed(self, name: str) -> None:
        if self.troubles.pop(name, None) is not None:
            log.info("%s is back at sea", name)
            self._save()

    def _strike(self, name: str, kind: str, detail: str) -> None:
        now = self.clock()
        previous = self.troubles.get(name)
        strikes = previous.strikes + 1 if previous and previous.kind == kind else 1
        pause = min(PAUSE[kind] * 2 ** (strikes - 1), LONGEST)
        self.troubles[name] = Trouble(kind, detail, previous.since if previous and previous.kind == kind else now, now + pause, strikes)
        log.info("%s cannot sail (%s); trying again in %d min", name, kind, pause // 60)
        self._save()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: asdict(trouble) for name, trouble in self.troubles.items()}

    def _save(self) -> None:
        if self.store is not None:
            try:
                self.store.put("crew", self.snapshot())
            except Exception:
                log.debug("could not save crew state", exc_info=True)
