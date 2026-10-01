"""One clock for everything that moves, and the rules for when it may move.

The waterline, the shoal, the tide line, a dive's progress and the dialogs
all subscribe to a single shared timer instead of each waking the process on
its own. The clock stops completely when no subscriber is visible or the app
is not in front, slows down on battery ("calm"), and can be switched off.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

from unconscious.sense.power import on_battery

FULL_MS = 80  # about twelve frames a second: the most anything asks for (a bobbing bottle, a fin)
SEA_MS = 100  # the dream's waterline: water this slow looks the same at ten frames a second
CALM_MS = 200  # on battery nothing moves more than five times a second
SLOW_MS = 160  # small ornaments (the sidebar's tide line) never need more than this


@dataclass
class _Rider:
    widget: QWidget
    callback: Callable[[float], None]
    every: int
    owed: float = 0.0  # milliseconds of motion not yet drawn


class Ticker(QObject):
    def __init__(self) -> None:
        super().__init__()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.CoarseTimer)  # lets the OS batch wake-ups
        self._timer.timeout.connect(self._tick)
        self._riders: dict[int, _Rider] = {}
        self.mode = "auto"
        self._last = 0.0
        self._checked = 0.0
        app = QGuiApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(lambda _state: self.refresh())

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.refresh()

    def effective(self) -> str:
        if self.mode in {"full", "calm", "off"}:
            return self.mode
        return "calm" if on_battery() else "full"

    def subscribe(self, widget: QWidget, callback: Callable[[float], None], every: int = FULL_MS) -> None:
        """Call ``callback(step)`` about every ``every`` ms while ``widget`` is on screen.
        ``step`` is elapsed time in units of FULL_MS, so speed does not depend on frame rate."""
        key = id(widget)
        self._riders[key] = _Rider(widget, callback, max(every, FULL_MS))
        widget.destroyed.connect(lambda _obj=None, k=key: self._drop(k))
        self.refresh()

    def _drop(self, key: int) -> None:
        self._riders.pop(key, None)
        try:
            self.refresh()
        except RuntimeError:  # shutting down: the clock went first
            pass

    def _active(self) -> list[_Rider]:
        riders = []
        for rider in list(self._riders.values()):
            try:
                if rider.widget.isVisible() and not rider.widget.window().isMinimized():
                    riders.append(rider)
            except RuntimeError:  # the C++ side is already gone
                continue
        return riders

    def _pace(self, rider: _Rider, mode: str) -> int:
        return max(rider.every, CALM_MS) if mode == "calm" else rider.every

    def refresh(self) -> None:
        mode = self.effective()
        app_in_front = QGuiApplication.applicationState() == Qt.ApplicationState.ApplicationActive
        riders = self._active()
        if mode == "off" or not app_in_front or not riders:
            self._timer.stop()
            return
        interval = min(self._pace(rider, mode) for rider in riders)
        if not self._timer.isActive():
            self._last = self._checked = time.monotonic()
            self._timer.start(interval)
        elif self._timer.interval() != interval:
            self._timer.setInterval(interval)

    def _tick(self) -> None:
        now = time.monotonic()
        elapsed = min((now - self._last) * 1000, 500)  # after a stall, do not leap
        self._last = now
        if now - self._checked > 5:  # every few seconds: did the charger come or go?
            self._checked = now
            self.refresh()
            if not self._timer.isActive():
                return
        mode = self.effective()
        riders = self._active()
        if not riders:
            self._timer.stop()
            return
        for rider in riders:
            rider.owed += elapsed
            if rider.owed >= self._pace(rider, mode) * 0.9:
                step, rider.owed = rider.owed / FULL_MS, 0.0
                rider.callback(step)


_ticker: Ticker | None = None


def ticker() -> Ticker:
    global _ticker
    if _ticker is None:
        _ticker = Ticker()
    return _ticker


def still() -> bool:
    """True when motion is switched off: draw a calm, static frame instead."""
    return ticker().effective() == "off"
