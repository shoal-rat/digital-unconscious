"""The sampling loop that turns a stream of samples into traces with dwell time.

Attribution rule: the time between two ticks belongs to whatever was in front
at the earlier tick. Reading without touching the keyboard still counts until
``idle_seconds`` passes; after that, the idle stretch is not credited.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from unconscious.sense.sensors import Sample, SensorError, make_sensor
from unconscious.sense.subjects import Subject, describe
from unconscious.store import now_iso

if TYPE_CHECKING:
    from unconscious.app import App

log = logging.getLogger(__name__)


@dataclass
class OpenTrace:
    trace_id: int
    key: str
    day: str
    seconds: float


def pause_state(app: App, now: datetime | None = None) -> dict:
    now = now or datetime.now().astimezone()
    until = app.store.get("sensor_paused_until")
    if not until:
        return {"paused": False, "until": None}
    if until == "forever":
        return {"paused": True, "until": "forever"}
    try:
        if datetime.fromisoformat(until) > now:
            return {"paused": True, "until": until}
    except ValueError:
        pass
    return {"paused": False, "until": None}


class Watcher:
    def __init__(self, app: App, sensor=None):
        self.app = app
        settings = app.settings
        self.sensor = sensor or make_sensor(settings.sense.capture_urls)
        self.current: OpenTrace | None = None
        self.last_tick: datetime | None = None
        self.last_label = ""
        self.last_app = ""
        self.last_state = "starting"
        self.error = ""
        self._last_maintenance: datetime | None = None
        self.started_at = now_iso()

    # -- core state machine ------------------------------------------------

    def tick(self, sample: Sample | None, now: datetime) -> str:
        settings = self.app.settings
        interval = settings.sense.interval_seconds
        threshold = settings.sense.idle_seconds
        max_gap = max(interval * 3, 60)

        if self.current and self.last_tick:
            elapsed = (now - self.last_tick).total_seconds()
            if 0 < elapsed <= max_gap:
                idle = sample.idle_seconds if sample else 0.0
                credit = elapsed if idle < threshold else max(0.0, elapsed - (idle - threshold))
                if credit > 0:
                    self.current.seconds += credit
                    self.app.store.extend_trace(self.current.trace_id, now.isoformat(timespec="seconds"), round(self.current.seconds, 1))
            elif elapsed > max_gap:
                self.current = None  # machine slept or the loop stalled
        self.last_tick = now

        if pause_state(self.app, now)["paused"] or not settings.sense.enabled:
            self.current = None
            return self._state("paused")
        if sample is None:
            self.current = None
            return self._state("error")
        if sample.idle_seconds >= threshold:
            self.current = None
            return self._state("idle")

        subject = describe(
            sample.app,
            sample.title,
            sample.url,
            quiet_apps=settings.sense.quiet_apps,
            quiet_domains=settings.sense.quiet_domains,
            private_apps=settings.sense.private_apps,
            capture_titles=settings.sense.capture_titles,
            capture_urls=settings.sense.capture_urls,
        )
        if subject is None:
            self.current = None
            self.last_label, self.last_app = "", ""
            return self._state("quiet")

        day = now.date().isoformat()
        if self.current and self.current.key == subject.key and self.current.day == day:
            return self._state("observing")
        self.current = OpenTrace(self._open(subject, now, day), subject.key, day, 0.0)
        self.last_label, self.last_app = subject.label, subject.app
        return self._state("observing")

    def _open(self, subject: Subject, now: datetime, day: str) -> int:
        stamp = now.isoformat(timespec="seconds")
        return self.app.store.add_trace(
            day=day,
            started_at=stamp,
            ended_at=stamp,
            seconds=0.0,
            kind=subject.kind,
            source="sensor",
            app=subject.app,
            category=subject.category,
            title=subject.title,
            url=subject.url,
            domain=subject.domain,
            subject_key=subject.key,
            subject=subject.label,
        )

    def _state(self, state: str) -> str:
        self.last_state = state
        return state

    # -- loop --------------------------------------------------------------

    def status(self) -> dict:
        caps = getattr(self.sensor, "capabilities", None)
        return {
            "state": self.last_state,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "heartbeat": now_iso(),
            "subject": self.last_label if self.last_state == "observing" else "",
            "app": self.last_app if self.last_state == "observing" else "",
            "error": self.error,
            "capabilities": caps.to_dict() if caps else {},
            "interval": self.app.settings.sense.interval_seconds,
        }

    def step(self) -> str:
        now = datetime.now().astimezone()
        sample: Sample | None
        try:
            sample = self.sensor.sample()
            self.error = ""
        except SensorError as exc:
            sample, self.error = None, str(exc)
        except Exception as exc:  # a sampler must never kill the loop
            log.exception("sensor failed")
            sample, self.error = None, f"{type(exc).__name__}: {exc}"
        state = self.tick(sample, now)
        self.app.store.put("sensor", self.status())
        self._maintain(now)
        return state

    def run(self, stop: threading.Event) -> None:
        log.info("watcher started (pid %s)", os.getpid())
        while not stop.is_set():
            self.step()
            stop.wait(self.app.settings.sense.interval_seconds)
        self.app.store.put("sensor", {**self.status(), "state": "stopped"})

    def _maintain(self, now: datetime) -> None:
        if self._last_maintenance and now - self._last_maintenance < timedelta(hours=1):
            return
        self._last_maintenance = now
        keep = self.app.settings.sense.retention_days
        cutoff = (now.date() - timedelta(days=keep)).isoformat()
        removed = self.app.store.forget_before(cutoff)
        if removed:
            log.info("expired %d traces older than %s", removed, cutoff)
