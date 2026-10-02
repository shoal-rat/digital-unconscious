"""Background work for the dashboard: one worker thread, persisted job records.

Model calls through the subscription CLIs take tens of seconds and should not
run concurrently, so jobs are strictly sequential. Progress steps are written
to the store, which is what the dashboard polls.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unconscious.app import App

log = logging.getLogger(__name__)

Work = Callable[[Callable[[str], None]], dict[str, Any]]


class Jobs:
    def __init__(self, app: App):
        self.app = app
        self._queue: queue.Queue[tuple[str, Work]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="dun-jobs", daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.app.store.fail_stale_jobs()
            self._thread.start()
            self._started = True

    def submit(self, kind: str, ref: str, work: Work) -> str:
        for job in self.app.store.active_jobs():
            if job["kind"] == kind and job["ref"] == ref:
                return job["id"]
        job_id = self.app.store.create_job(kind, ref)
        self._queue.put((job_id, work))
        return job_id

    def _run(self) -> None:
        while True:
            job_id, work = self._queue.get()
            store = self.app.store
            store.update_job(job_id, state="running", step="starting")
            try:
                result = work(lambda step, job_id=job_id, store=store: store.update_job(job_id, step=step))
                store.update_job(job_id, state="done", step="done", result=result)
            except Exception as exc:
                log.warning("job %s failed: %s", job_id, exc, exc_info=not isinstance(exc, RuntimeError))
                store.update_job(job_id, state="failed", error=str(exc)[:1500])


def counted(app: App, day: str, key: str, work: Work, keep_days: int = 14) -> Work:
    """Count an automatic attempt only when it really failed. A crew that could not sail
    (signed out, limited, offline, held by region) leaves the attempts untouched: the
    scheduler waits for the crew and tries again."""
    from unconscious.llm.crew import parse

    def run(progress: Callable[[str], None]) -> dict[str, Any]:
        try:
            return work(progress)
        except Exception as exc:
            if parse(str(exc)) is None:
                record_attempt(app, day, key, keep_days)
            raise

    return run


def dream_job(app: App, day: str, force_digest: bool = False) -> Work:
    from unconscious.mind.dream import run_dream

    return lambda progress: run_dream(app, day, progress, force_digest=force_digest)


def dive_job(app: App, spark_id: int) -> Work:
    from unconscious.mind.dive import run_dive

    return lambda progress: run_dive(app, spark_id, progress)


GROWN_SECONDS = 30 * 60  # this much more attention since a day's last dive is worth another dive


def dream_moment(app: App, day: str) -> datetime:
    hour, minute = (int(x) for x in app.settings.dream.time.split(":"))
    return datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:00").astimezone()


def grown_since_dive(app: App, day: str, dive: dict[str, Any]) -> bool:
    """Has the day grown enough since its last dive to dive again: half an hour more on the
    driftline, or a bottle, a note or a fed document that came in after the dive read it.
    Both sides are the whole driftline (dives before 3.2 only kept what their sorting counted)."""
    stats = (dive.get("payload") or {}).get("stats") or {}
    dived = stats.get("day_seconds", stats.get("seconds")) or 0
    if app.store.day_seconds(day) - dived >= GROWN_SECONDS:
        return True
    return app.store.notes_since(day, stats.get("read_at") or dive["created_at"]) > 0


def due_dreams(app: App, now: datetime | None = None) -> list[str]:
    """Days that should be dreamt automatically right now.

    Today, once the configured dream time has passed; and yesterday, if the
    machine was asleep or off at dream time. A day already dived by hand before
    its dream time is dived again only if it has grown a good deal since; a dive
    at or after dream time is the night's. Each day is attempted at most twice.
    """
    settings = app.settings
    if not settings.dream.auto:
        return []
    now = now or datetime.now().astimezone()
    hour, minute = (int(x) for x in settings.dream.time.split(":"))
    attempts: dict[str, int] = app.store.get("auto_dream_attempts", {}) or {}
    today = now.date()
    candidates = []
    if (now.hour, now.minute) >= (hour, minute):
        candidates.append(today.isoformat())
    candidates.append((today - timedelta(days=1)).isoformat())
    due = []
    for day in candidates:
        if attempts.get(day, 0) >= 2:
            continue
        dive = app.store.dream(day)
        if dive is not None:
            if datetime.fromisoformat(dive["created_at"]).astimezone() >= dream_moment(app, day):
                continue  # the night already dived
            if grown_since_dive(app, day, dive):
                due.append(day)
            continue
        if app.store.day_seconds(day) < 15 * 60 and app.store.trace_count(day) < 3:
            continue  # not enough to dream about
        due.append(day)
    return due


def record_attempt(app: App, day: str, key: str = "auto_dream_attempts", keep_days: int = 14) -> None:
    attempts: dict[str, int] = app.store.get(key, {}) or {}
    attempts[day] = attempts.get(day, 0) + 1
    cutoff = (date.today() - timedelta(days=keep_days)).isoformat()
    app.store.put(key, {d: n for d, n in attempts.items() if d >= cutoff})


def sort_job(app: App, day: str) -> Work:
    from unconscious.mind.digest import digest_day

    def work(progress: Callable[[str], None]) -> dict[str, Any]:
        progress("digest")
        topics = len(digest_day(app, day).topics)
        if day == date.today().isoformat():  # tonight's sorting is done; the rest waits for tomorrow's catch-up
            done = {d: v for d, v in (app.store.get("night_sorted") or {}).items() if d >= day}
            app.store.put("night_sorted", {**done, day: datetime.now().astimezone().isoformat(timespec="seconds")})
        return {"topics": topics}

    return work


def due_sorting(app: App, now: datetime | None = None) -> list[str]:
    """Days whose driftline is not (or no longer fully) sorted into currents. Sorting is what
    writes a day into each current's history, and the undercurrents are measured from that
    history, so a day must be sorted before its raw driftlines wash away. Oldest first, one at
    a time, twice at most:

    - past days the watcher saw but no dive sorted (the machine slept through the night);
    - days sorted before the rest of them came in: the evening after a day's dive;
    - today, once after dream time, when it was dived by hand earlier and grew too little to
      dive again (the rest of the evening is caught up tomorrow).

    Yesterday with no dive at all belongs to the dream catch-up, and so does any day that is
    due a dive."""
    settings = app.settings
    if not settings.dream.auto:
        return []
    now = now or datetime.now().astimezone()
    today = now.date()
    since = (today - timedelta(days=max(int(settings.sense.retention_days), 2))).isoformat()
    yesterday = (today - timedelta(days=1)).isoformat()
    attempts: dict[str, int] = app.store.get("auto_sort_attempts", {}) or {}
    diving = set(due_dreams(app, now))
    # an upgraded memory: days dived by 3.1 all look stale (their evenings were never sorted); leave them be
    stale_from = max(since, app.store.get("stale_since") or since)
    days = set(app.store.undigested_days(since, yesterday)) | set(app.store.stale_digests(stale_from, today.isoformat()))
    tonight, latest = today.isoformat(), app.store.dream(today.isoformat())
    if tonight not in diving and latest and now >= dream_moment(app, tonight):
        by_hand = datetime.fromisoformat(latest["created_at"]).astimezone() < dream_moment(app, tonight)
        if by_hand and app.store.stale_digests(tonight, (today + timedelta(days=1)).isoformat()):
            if (app.store.get("night_sorted") or {}).get(tonight) is None:
                days.add(tonight)
    return [day for day in sorted(days) if day not in diving and attempts.get(day, 0) < 2][:1]


def crew_ready(app: App, role: str = "digest") -> bool:
    """Someone aboard who may sail for this role now, the region guard included (background threads)."""
    try:
        return app.router.ready(role)
    except Exception:
        return False


def scheduler_tick(app: App, jobs: Jobs) -> None:
    """One minute of the night watch: tidy the harbour, dive when a dive is due and the crew
    can sail, and otherwise sort any day that was missed. Nothing here waits on the crew."""
    from unconscious.housekeeping import tidy

    try:
        tidy(app)  # at most once a day; covers machines where the watcher is off
    except Exception:
        log.exception("housekeeping failed")
    try:
        due = due_dreams(app)
        # A crew that cannot sail (nobody signed in, signed out, limited, offline, or the connection
        # in a held region) does not spend the day's attempts: the dive waits for the crew.
        if due and crew_ready(app, "digest") and crew_ready(app, "dream"):
            for day in sorted(due):  # oldest first, so dive numbers follow the days
                jobs.submit("dream", day, counted(app, day, "auto_dream_attempts", dream_job(app, day)))
                log.info("auto-dream queued for %s", day)
        if not app.store.active_jobs():
            for day in due_sorting(app):
                if crew_ready(app):
                    keep = app.settings.sense.retention_days
                    jobs.submit("sort", day, counted(app, day, "auto_sort_attempts", sort_job(app, day), keep))
                    log.info("catch-up sorting queued for %s", day)
    except Exception:
        log.exception("scheduler tick failed")


def run_scheduler(app: App, jobs: Jobs, stop: threading.Event, every: float = 60.0) -> None:
    while not stop.wait(every):
        scheduler_tick(app, jobs)
