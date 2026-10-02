"""Start the desktop app: services in background threads, a window, and a tray icon.

The tray icon is how the app lives with you: closing the window keeps the
sensor observing and the nightly dream scheduled; Quit is in the tray menu.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QCursor, QGuiApplication, QIcon, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from unconscious.store import today
from unconscious.text import clip
from unconscious.ui import theme
from unconscious.ui.i18n import set_language, t
from unconscious.ui.widgets import mac_icon_image, mark_pixmap

log = logging.getLogger(__name__)


class Tray(QSystemTrayIcon):
    def __init__(self, window):
        super().__init__()
        self.window = window
        self._state_key = None
        self.setToolTip("Digital Unconscious")
        self.activated.connect(self._activated)
        self.rebuild()
        self.update_state(window.state)
        self.show()

    def rebuild(self) -> None:
        menu = QMenu()
        self.status_action = QAction("…", menu)
        self.status_action.setEnabled(False)
        menu.addAction(self.status_action)
        menu.addSeparator()
        menu.addAction(t("tray.open"), self.window.show_window)
        menu.addAction(t("tray.jot"), self.window.open_jot)
        menu.addAction(t("tray.dream"), lambda: self.window.dream(today()))
        self.pause_action = menu.addAction(t("tray.pause"), self._toggle_pause)
        menu.addSeparator()
        menu.addAction(t("tray.quit"), self._quit)
        self.menu = menu
        if sys.platform != "darwin":
            self.setContextMenu(menu)
        # On macOS the menu is popped by hand (_activated). A menu attached through Qt makes
        # Qt ask the event that opened it for its click count, and newer macOS aborts the
        # app when that event is not a mouse event: one click on the mark, and the sea is gone.
        self._state_key = None
        self.update_state(self.window.state)

    def update_state(self, state: dict) -> None:
        sensor = state.get("sensor", {})
        dreaming = any(j["kind"] in {"dream", "dive"} for j in state.get("jobs") or [])
        paused = bool(sensor.get("paused"))
        key = (dreaming, paused, sensor.get("state"), sensor.get("subject"))
        if key == self._state_key:
            return
        self._state_key = key
        mode = "dreaming" if dreaming else "paused" if paused or sensor.get("state") in {"stopped", "never"} else "observing"
        icon = QIcon(mark_pixmap(44, mono=True, state=mode))
        icon.setIsMask(True)  # macOS template image: follows light/dark menu bar
        self.setIcon(icon)
        label = t("sensor.dreaming") if dreaming else t(f"sensor.{'paused' if paused else sensor.get('state', 'never')}")
        if sensor.get("subject") and not dreaming and not paused:
            label += f" · {clip(sensor['subject'], 40)}"
        self.status_action.setText(label)
        self.pause_action.setText(t("tray.resume") if paused else t("tray.pause"))
        self.setToolTip(f"Digital Unconscious — {label}")

    def _toggle_pause(self) -> None:
        store = self.window.app.store
        if self.window.state.get("sensor", {}).get("paused"):
            store.put("sensor_paused_until", None)
        else:
            until = datetime.now().astimezone() + timedelta(hours=1)
            store.put("sensor_paused_until", until.isoformat(timespec="seconds"))
        self.window.poll(force=True)

    def _activated(self, reason) -> None:
        if sys.platform == "darwin":  # on macOS a click opens the menu, which is the convention
            area = self.geometry()
            self.menu.popup(area.bottomLeft() if area.isValid() else QCursor.pos())
            return
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.window.show_window()

    def _quit(self) -> None:
        self.window.quitting = True
        QApplication.instance().quit()


def _instance_name(home: Path) -> str:
    digest = hashlib.sha1(str(home.resolve()).encode()).hexdigest()[:10]
    return f"digital-unconscious-{os.getuid() if hasattr(os, 'getuid') else 'user'}-{digest}"


BUNDLED = bool(getattr(sys, "frozen", False))  # running as Digital Unconscious.app, not from a terminal


class _QuitWatch(QObject):
    """⌘Q, the Dock's Quit and logging out close every window first; a window that only hides
    when closed would cancel that. So when the app is asked to quit, the window lets go."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Quit:
            self.window.quitting = True
        return False


def _behave_like_a_mac_app(qt_app: QApplication, window, ctx) -> None:
    qt_app.installEventFilter(_QuitWatch(window))

    def reopen(state) -> None:  # clicking the Dock icon brings the shore back
        if state == Qt.ApplicationState.ApplicationActive and not window.isVisible():
            window.show_window()

    qt_app.applicationStateChanged.connect(reopen)
    if ctx.settings.sense.enabled and ctx.settings.sense.capture_titles:
        from unconscious.sense.macnative import ask_for_accessibility

        ask_for_accessibility()  # macOS shows its own prompt the first time


def run(*, hidden: bool = False, demo: bool = False, watch: bool = True, auto_dream: bool = True) -> int:
    from unconscious.app import App
    from unconscious.jobs import Jobs, run_scheduler
    from unconscious.ui.window import MainWindow

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    qt_app = QApplication.instance() or QApplication(sys.argv)
    qt_app.setApplicationName("Digital Unconscious")
    qt_app.setApplicationDisplayName("Digital Unconscious")
    qt_app.setOrganizationName("Digital Unconscious")
    qt_app.setQuitOnLastWindowClosed(False)

    ctx = App()
    name = _instance_name(ctx.home)
    probe = QLocalSocket()
    probe.connectToServer(name)
    if probe.waitForConnected(300):
        probe.write(b"show")
        probe.flush()
        probe.waitForBytesWritten(300)
        print("Digital Unconscious is already running; brought it to the front.")
        return 0
    QLocalServer.removeServer(name)
    server = QLocalServer()
    server.listen(name)

    theme.load_fonts()
    if sys.platform != "darwin":
        qt_app.setWindowIcon(QIcon(mark_pixmap(256)))
    elif not BUNDLED:
        # On macOS this is the Dock icon. Inside the .app the bundle's own icon is already right;
        # from a terminal, draw the same one on Apple's grid so it is the size of its neighbours.
        qt_app.setWindowIcon(QIcon(QPixmap.fromImage(mac_icon_image(512))))
    set_language(ctx.settings.language)
    theme.apply(qt_app, theme.detect_dark(ctx.settings.ui.theme))
    from unconscious.ui.motion import ticker

    ticker().set_mode(ctx.settings.ui.motion)

    jobs = Jobs(ctx)
    jobs.start()
    stop = threading.Event()
    if watch and not demo and ctx.settings.sense.enabled:
        from unconscious.cli import _other_watcher
        from unconscious.sense.watcher import Watcher

        if not _other_watcher(ctx):
            threading.Thread(target=Watcher(ctx).run, args=(stop,), name="dun-watch", daemon=True).start()
    if auto_dream and not demo:
        threading.Thread(target=run_scheduler, args=(ctx, jobs, stop), name="dun-scheduler", daemon=True).start()

    window = MainWindow(ctx, jobs, demo=demo)
    if QSystemTrayIcon.isSystemTrayAvailable():
        window.tray = Tray(window)
    server.newConnection.connect(lambda: (server.nextPendingConnection(), window.show_window()))

    def follow_system(_scheme) -> None:
        if ctx.settings.ui.theme == "system":
            window.apply_preferences()

    QGuiApplication.styleHints().colorSchemeChanged.connect(follow_system)
    if BUNDLED and sys.platform == "darwin":
        _behave_like_a_mac_app(qt_app, window, ctx)
    if not (hidden or ctx.settings.ui.start_hidden) or window.tray is None:
        window.show_window()
    code = qt_app.exec()
    stop.set()
    server.close()
    return code
