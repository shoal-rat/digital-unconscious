"""Keep the harbour small without forgetting the person.

The main line is what a dive remembers someone by, and housekeeping never touches it:

    the currents and each current's day-by-day history   threads, thread_days
    dreams, fish and their seabed searches                dreams, sparks, dives
    the net: what was kept, followed or thrown back, why  events (and each fish's status)

Only the shark removes those, and only when the person asks.

What fades, once a day, done by whoever is awake (the watcher or the night watch):

    after 14 days         visits to one subject on one day merge into one row. Same subjects,
                          same totals: nothing a dive reads changes.
    after retention_days  raw driftlines expire, with the digests that cached their sorting.
                          By then every day has been sorted into its currents (the night watch
                          sorts any day it missed: ``jobs.due_sorting``).
    after a month / year  finished jobs / the crew's call log.
    above 1 MB            the login item's log keeps its last 256 KB.

Freed pages then go back to the disk.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unconscious.app import App

log = logging.getLogger(__name__)

MAIN_LINE = ("threads", "thread_days", "dreams", "sparks", "dives", "events")  # never tidied away
SMOOTH_AFTER_DAYS = 14  # visits to one subject merge into one row after two weeks
LOG_LIMIT = 1 << 20  # a login-item log above 1 MB keeps only its last 256 KB
LOG_KEEP = 256 * 1024


def tidy(app: App, today: date | None = None, *, force: bool = False) -> dict[str, int] | None:
    """Run once per day across processes; returns what was done, or None if it already ran."""
    today = today or datetime.now().astimezone().date()
    stamp = today.isoformat()
    if not force and app.store.get("tidied_on") == stamp:
        return None
    app.store.put("tidied_on", stamp)
    keep = max(int(app.settings.sense.retention_days), SMOOTH_AFTER_DAYS + 1)
    done = {
        "expired": app.store.forget_before((today - timedelta(days=keep)).isoformat()),
        "merged": app.store.compact_before((today - timedelta(days=SMOOTH_AFTER_DAYS)).isoformat()),
        "pruned": app.store.prune_logs(stamp),
    }
    done["vacuumed"] = int(app.store.vacuum(threshold=0.2))
    for path in log_files(app):
        done["logs"] = done.get("logs", 0) + int(trim_log(path))
    if any(done.values()):
        log.info("tidied the harbour: %s", done)
    return done


def log_files(app: App) -> list[Path]:
    folder = app.home / "logs"
    return sorted(folder.glob("*.log")) if folder.is_dir() else []


def trim_log(path: Path, limit: int = LOG_LIMIT, keep: int = LOG_KEEP) -> bool:
    try:
        if path.stat().st_size <= limit:
            return False
        with path.open("rb") as handle:
            handle.seek(-keep, 2)
            tail = handle.read()
        cut = tail.find(b"\n")
        path.write_bytes(tail[cut + 1:] if cut >= 0 else tail)
        return True
    except OSError:
        return False
