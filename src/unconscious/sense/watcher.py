"""The sampling loop that turns a stream of samples into traces with dwell time.

Attribution rule: the time between two ticks belongs to whatever was in front
at the earlier tick. Reading without touching the keyboard still counts until
``idle_seconds`` passes; after that, the idle stretch is not credited.

Energy: the loop glances every ``interval_seconds`` while attention moves,
backs off (2×, then 4×) while it stays on one thing, checks only the idle
clock at slack water, and stretches further on battery. Because credit is
computed from real elapsed time, a longer gap changes how quickly a switch is
noticed, not how much time is counted.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from unconscious.housekeeping import tidy
from unconscious.sense.power import on_battery
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
        self.steady = 0  # consecutive ticks on the same subject
        self.planned_delay = float(settings.sense.interval_seconds)
        self._status_written: tuple[str, str, float] = ("", "", 0.0)

    # -- core state machine ------------------------------------------------

    def tick(self, sample: Sample | None, now: datetime) -> str:
        settings = self.app.settings
        interval = settings.sense.interval_seconds
        threshold = settings.sense.idle_seconds
        max_gap = max(self.planned_delay * 2.5, interval * 3, 60)

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
            self.steady += 1
            return self._state("observing")
        self.steady = 0
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
        if state != "observing":
            self.steady = 0
        self.last_state = state
        return state

    def next_delay(self, battery: bool | None = None) -> float:
        """How long to wait before the next glance."""
        sense = self.app.settings.sense
        if battery is None:
            battery = sense.battery_saver and on_battery()
        base = float(max(sense.interval_seconds, 20) if battery else sense.interval_seconds)
        ceiling = 90.0 if battery else 60.0
        state = self.last_state
        if state == "paused":
            delay = 60.0
        elif state in {"idle", "error"}:
            delay = 45.0 if battery else 30.0
        elif self.steady >= 8:
            delay = base * 4
        elif self.steady >= 4:
            delay = base * 2
        else:
            delay = base
        self.planned_delay = min(max(delay, float(sense.interval_seconds)), ceiling)
        return self.planned_delay

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
            "interval": round(self.planned_delay),
        }

    def step(self) -> str:
        now = datetime.now().astimezone()
        sample: Sample | None
        try:
            # At slack water only the idle clock is read; the full glance waits for your return.
            idle_clock = getattr(self.sensor, "idle_seconds", None)
            idle = idle_clock() if callable(idle_clock) else None
            if idle is not None and idle >= self.app.settings.sense.idle_seconds:
                sample = Sample(app="", idle_seconds=idle)
            else:
                sample = self.sensor.sample()
            self.error = ""
        except SensorError as exc:
            sample, self.error = None, str(exc)
        except Exception as exc:  # a sampler must never kill the loop
            log.exception("sensor failed")
            sample, self.error = None, f"{type(exc).__name__}: {exc}"
        state = self.tick(sample, now)
        # The status row is what the shore reads; write it when something
        # visible changed or once a minute as a heartbeat, not on every glance.
        last_state, last_subject, last_time = self._status_written
        subject = self.last_label if state == "observing" else ""
        if (state, subject) != (last_state, last_subject) or time.monotonic() - last_time >= 60 or self.error:
            self.app.store.put("sensor", self.status())
            self._status_written = (state, subject, time.monotonic())
        self._maintain(now)
        return state

    def run(self, stop: threading.Event) -> None:
        log.info("watcher started (pid %s)", os.getpid())
        while not stop.is_set():
            self.step()
            stop.wait(self.next_delay())
        self.app.store.put("sensor", {**self.status(), "state": "stopped"})

    def _maintain(self, now: datetime) -> None:
        if self._last_maintenance and now - self._last_maintenance < timedelta(hours=1):
            return
        self._last_maintenance = now
        try:
            tidy(self.app, now.date())  # at most once a day, whoever gets there first
        except Exception:  # housekeeping must never stop the watcher
            log.exception("housekeeping failed")
