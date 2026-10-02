"""The main window: a quiet sidebar, the current page, and the plumbing between them."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from unconscious import __version__, api, ingest
from unconscious.jobs import dive_job, dream_job
from unconscious.store import today
from unconscious.text import clip
from unconscious.ui import theme
from unconscious.ui.dialogs import JotDialog
from unconscious.ui.i18n import CREW_NAMES, crew_message, set_language, t, trouble_message
from unconscious.ui.motion import ticker
from unconscious.ui.pages.journal import JournalPage
from unconscious.ui.pages.settings import SettingsPage
from unconscious.ui.pages.sparks import SparkPage, SparksPage
from unconscious.ui.pages.threads import ThreadPage, ThreadsPage
from unconscious.ui.pages.today import TodayPage
from unconscious.ui.theme import THEME, font
from unconscious.ui.widgets import TideLine, app_icon, button, label, mark_pixmap, plain_tip, vbox

if TYPE_CHECKING:
    from unconscious.app import App
    from unconscious.jobs import Jobs

log = logging.getLogger(__name__)

PAGES = {
    "today": TodayPage, "threads": ThreadsPage, "thread": ThreadPage, "sparks": SparksPage,
    "spark": SparkPage, "journal": JournalPage, "settings": SettingsPage,
}
NAV = [("today", "nav.today", "1"), ("threads", "nav.threads", "2"), ("sparks", "nav.sparks", "3"),
       ("journal", "nav.journal", "4"), ("settings", "nav.settings", "5")]
SECTION_OF = {"thread": "threads", "spark": "sparks"}


class SensorDot(QWidget):
    def __init__(self):
        super().__init__()
        self.state = "never"
        self.setFixedSize(10, 10)

    def set_state(self, state: str) -> None:
        self.state = state
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = {
            "observing": THEME.color("ok"), "idle": THEME.color("warn"), "quiet": THEME.color("warn"),
            "error": THEME.color("accent"), "dreaming": THEME.color("accent"),
        }.get(self.state, THEME.color("faint"))
        if self.state == "paused":
            p.setPen(THEME.color("muted"))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QRectF(1.5, 1.5, 7, 7))
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawEllipse(QPointF(5, 5), 4, 4)


class Sidebar(QWidget):
    navigate = Signal(str)

    def __init__(self, window: MainWindow):
        super().__init__()
        self.window = window
        self.setObjectName("Sidebar")
        self.setFixedWidth(232)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.buttons: dict[str, object] = {}
        self.build()

    def build(self) -> None:
        if self.layout():
            QWidget().setLayout(self.layout())  # discard the old layout and its children
        box = vbox(spacing=3, margins=(18, 26, 18, 20))
        brand = QHBoxLayout()
        brand.setSpacing(10)
        logo = QLabel()
        pixmap = mark_pixmap(64)
        pixmap.setDevicePixelRatio(2)
        logo.setPixmap(pixmap)
        brand.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)
        words = vbox(spacing=0)
        upright = label("Digital", "wordmark", wrap=False)
        f = upright.font()
        f.setItalic(False)
        upright.setFont(f)
        words.addWidget(upright)
        words.addWidget(label("Unconscious", "wordmark", wrap=False))
        brand.addLayout(words)
        brand.addStretch(1)
        box.addLayout(brand)
        box.addSpacing(10)
        box.addWidget(label(t("app.tagline"), "caption", "muted"))
        box.addSpacing(26)
        self.buttons = {}
        for key, text, shortcut in NAV:
            b = button(t(text), "nav", lambda k=key: self.navigate.emit(k), tip=f"⌘{shortcut}")
            b.setFont(font("nav"))
            self.buttons[key] = b
            box.addWidget(b)
        box.addStretch(1)
        status = QHBoxLayout()
        status.setSpacing(8)
        self.dot = SensorDot()
        status.addWidget(self.dot, 0, Qt.AlignmentFlag.AlignVCenter)
        self.state_label = label("", "caption-l", wrap=False)
        status.addWidget(self.state_label, 1)
        box.addLayout(status)
        self.subject_label = label("", "caption", "muted", wrap=False)
        self.subject_label.setFixedWidth(190)
        box.addWidget(self.subject_label)
        box.addSpacing(6)
        self.tide = TideLine()
        box.addWidget(self.tide)
        box.addSpacing(12)
        jot = button(t("jot.button"), "jot", self.window.open_jot, tip="⌘J")
        box.addWidget(jot)
        box.addSpacing(14)
        box.addWidget(label(f"v{__version__}", "caption", "faint", wrap=False))
        self.setLayout(box)

    def set_active(self, key: str) -> None:
        for name, b in self.buttons.items():
            b.setProperty("active", "true" if name == key else "false")
            b.style().unpolish(b)
            b.style().polish(b)

    def set_status(self, state: dict) -> None:
        sensor = state.get("sensor", {})
        name = sensor.get("state", "never")
        jobs = state.get("jobs") or []
        if self.window.demo:
            name_key, dot = "sensor.demo", "paused"
        elif any(j["kind"] == "dream" for j in jobs):
            name_key, dot = "sensor.dreaming", "dreaming"
        elif any(j["kind"] == "dive" for j in jobs):
            name_key, dot = "sensor.diving", "dreaming"
        elif sensor.get("paused"):
            name_key, dot = "sensor.paused", "paused"
        else:
            name_key, dot = f"sensor.{name}", name
        self.dot.set_state(dot)
        self.tide.set_state(dot)
        self.state_label.setText(t(name_key))
        subject = sensor.get("subject") or ""
        if not subject and state.get("today_stats", {}).get("seconds"):
            from unconscious.ui.charts import human

            subject = human(state["today_stats"]["seconds"]) + " · " + t("common.today").lower()
        shown = self.subject_label.fontMetrics().elidedText(subject, Qt.TextElideMode.ElideRight, 188)
        self.subject_label.setText(shown)
        self.subject_label.setToolTip(plain_tip(subject) if shown != subject else "")


class Toast(QLabel):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setFont(font("body"))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text: str, error: bool = False) -> None:
        background = THEME.c["accent_ink"] if error else THEME.c["ink"]
        self.setStyleSheet(f"background: {background}; color: {THEME.c['paper']}; border-radius: 6px; padding: 10px 16px;")
        self.setText(text)
        self.setMaximumWidth(460)
        self.adjustSize()
        self._place()
        self.show()
        self.raise_()
        self._timer.start(7000 if error else 3200)

    def _place(self) -> None:
        parent = self.parentWidget()
        if parent:
            self.move((parent.width() - self.width()) // 2 + 112, parent.height() - self.height() - 26)


class MainWindow(QMainWindow):
    def __init__(self, app: App, jobs: Jobs, *, demo: bool = False):
        super().__init__()
        self.app = app
        self.jobs = jobs
        self.demo = demo
        self.tray: QSystemTrayIcon | None = None
        self.failures: dict[str, str] = {}
        self.history: list[tuple[str, dict]] = []
        self.route: tuple[str, dict] = ("today", {})
        self.page = None
        self.state = api.state(app)
        self._active_jobs = {j["id"]: j for j in self.state.get("jobs", [])}
        self._signature = None
        self._told_hidden = False

        self.setWindowTitle("Digital Unconscious")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1000, 700)
        self.resize(1320, 900)
        central = QWidget()
        self.row = QHBoxLayout(central)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(0)
        self.sidebar = Sidebar(self)
        self.sidebar.navigate.connect(lambda key: self.go(key))
        self.row.addWidget(self.sidebar)
        self.holder = QVBoxLayout()
        self.holder.setContentsMargins(0, 0, 0, 0)
        self.row.addLayout(self.holder, 1)
        self.setCentralWidget(central)
        self.toast_label = Toast(central)

        for key, _text, number in NAV:
            QShortcut(QKeySequence(f"Ctrl+{number}"), self, activated=lambda k=key: self.go(k))
        QShortcut(QKeySequence("Ctrl+J"), self, activated=self.open_jot)
        QShortcut(QKeySequence("Ctrl+["), self, activated=self.back)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=lambda: self.page and self.page.refresh())
        QShortcut(QKeySequence("Ctrl+,"), self, activated=lambda: self.go("settings"))
        close = QAction(self)
        close.setShortcut(QKeySequence.StandardKey.Close)
        close.triggered.connect(self.close)
        self.addAction(close)

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.CoarseTimer)
        self.timer.timeout.connect(self.poll)
        self.timer.start(5000)
        self._pace()  # started hidden in the menu bar: poll slowly until the window opens
        self.sidebar.set_status(self.state)
        # The first page is built when the window first opens: a sea that starts in the menu bar
        # at login does not pay for a page nobody has looked at yet.

    # -- navigation ----------------------------------------------------------

    def go(self, name: str, *, replace: bool = False, **params) -> None:
        if name == "today" and params.get("day") == today():
            params.pop("day")
        if self.page is not None and not replace and (name, params) != self.route:
            self.history.append(self.route)
            self.history = self.history[-50:]
        self.route = (name, params)
        page = PAGES[name](self, **params)
        page.refresh()
        if self.page is not None:
            self.holder.removeWidget(self.page)
            self.page.hide()  # off the screen now, not whenever the event loop gets round to deleting it
            self.page.deleteLater()
        self.page = page
        self.holder.addWidget(page)
        self.sidebar.set_active(SECTION_OF.get(name, name))

    def back(self) -> None:
        if self.history:
            name, params = self.history.pop()
            self.go(name, replace=True, **params)

    def refresh(self) -> None:
        if self.page is not None:
            try:
                self.state = api.state(self.app)  # pages read the full state when they rebuild
            except Exception:
                log.exception("state refresh failed")
            self.page.refresh()

    # -- polling -------------------------------------------------------------

    def _pace(self) -> None:
        """Poll often only when someone is looking and something is happening."""
        if not self.isVisible() or self.isMinimized():
            interval = 30_000  # only the menu-bar mark needs news
        elif self._active_jobs:
            interval = 1_500
        else:
            interval = 5_000
        if self.timer.interval() != interval:
            self.timer.setInterval(interval)

    def poll(self, force: bool = False) -> None:
        try:
            fresh = api.pulse(self.app)
        except Exception:
            log.exception("state poll failed")
            return
        # pulse carries only part of each nested group; keep the rest from the last full state
        state = {**self.state, **fresh}
        for key in ("today_stats", "counts"):
            state[key] = {**self.state.get(key, {}), **fresh[key]}
        previous = self._active_jobs
        current = {j["id"]: j for j in state.get("jobs", [])}
        self.state = state
        self._active_jobs = current
        self._tell_crew(state.get("crew") or [])
        self.sidebar.set_status(state)
        if self.tray is not None:
            self.tray.update_state(state)
        finished = [job_id for job_id in previous if job_id not in current]
        for job_id in finished:
            self._finished(self.app.store.job(job_id) or {"id": job_id, "state": "failed", "error": "", "kind": previous[job_id]["kind"], "ref": previous[job_id]["ref"]})
        # the date too: past midnight the Shore turns to the new day instead of offering to re-dive the old one,
        # but not while that day's dive is still out (it turns when the dive comes back)
        held = self._signature[0] if self._signature and any(
            j["kind"] == "dream" and j["ref"] == self._signature[0] for j in current.values()) else today()
        signature = (held, state.get("latest_dream"), state["counts"]["dreams"], state["counts"]["new_sparks"])
        if finished or force or (self._signature is not None and signature != self._signature):
            self.refresh()
        elif current and self.page is not None:
            self.page.on_state(state)
        self._signature = signature
        self._pace()

    def _tell_crew(self, troubles: list[dict]) -> None:
        """Signing in again is the one trouble only the person can fix: say so once."""
        told = getattr(self, "_told_crew", set())
        for trouble in troubles:
            key = (trouble["who"], trouble["kind"], trouble.get("since"))
            if trouble["kind"] == "sign_in" and key not in told:
                told.add(key)
                name = CREW_NAMES.get(trouble["who"], trouble["who"])
                self.notify(t("crew.signInTitle", who=name), trouble_message(trouble["kind"], trouble["who"]))
        self._told_crew = told

    def _finished(self, job: dict) -> None:
        kind, ref = job.get("kind"), job.get("ref", "")
        if kind == "sort":  # quiet catch-up sorting: nothing to announce, the page just refreshes
            return
        if job.get("state") == "done":
            if kind == "dream":
                self.failures.pop(ref, None)
                result = job.get("result") or {}
                self.notify(t("notify.dream"), result.get("title") or "")
            else:
                self.failures.pop(f"dive:{ref}", None)
                self.notify(t("notify.dive"), "")
        else:
            key = ref if kind == "dream" else f"dive:{ref}"
            self.failures[key] = crew_message(job.get("error") or "") or t("common.error")
            self.toast(clip(self.failures[key], 300), error=True)  # the whole reason stays on the page

    # -- actions -------------------------------------------------------------

    def dream(self, day: str, redigest: bool = False) -> None:
        if self.app.store.trace_count(day) == 0:
            self.toast(t("today.thin"), error=True)
            return
        self.failures.pop(day, None)
        self.jobs.submit("dream", day, dream_job(self.app, day, redigest))
        self.poll(force=True)

    def dive(self, spark_id: int) -> None:
        self.failures.pop(f"dive:{spark_id}", None)
        self.jobs.submit("dive", str(spark_id), dive_job(self.app, spark_id))
        self.poll(force=True)

    def spark_feedback(self, spark_id: int, status: str, reason: str) -> None:
        self.app.store.set_spark_status(spark_id, status, reason)
        self.app.store.log_event("feedback", str(spark_id), {"status": status, "reason": reason})
        if status == "pursuing" and self.route[0] != "spark":
            self.go("spark", id=spark_id)
        else:
            self.refresh()

    def save_jot(self, text: str) -> None:
        try:
            ingest.jot(self.app, text)
        except ingest.IngestError as exc:
            self.toast(str(exc), error=True)
            return
        self.toast(t("jot.saved"))
        if self.route[0] == "today":
            self.refresh()

    def open_jot(self) -> None:
        self.show_window()
        dialog = JotDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.value():
            self.save_jot(dialog.value())

    def toast(self, text: str, error: bool = False) -> None:
        self.toast_label.show_message(text, error)

    def notify(self, title: str, message: str) -> None:
        if self.tray is not None and self.tray.isVisible():
            self.tray.showMessage(title, message or title, mark_icon(), 6000)
        self.toast(f"{title}{' — ' + message if message else ''}")

    def apply_preferences(self) -> None:
        from PySide6.QtWidgets import QApplication

        settings = self.app.settings
        set_language(settings.language)
        theme.apply(QApplication.instance(), theme.detect_dark(settings.ui.theme))
        ticker().set_mode(settings.ui.motion)
        self.sidebar.build()
        self.sidebar.set_status(self.state)
        if self.tray is not None:
            self.tray.rebuild()
        name, params = self.route
        self.go(name, replace=True, **params)

    def show_window(self) -> None:
        self.show()
        self.setWindowState((self.windowState() & ~Qt.WindowState.WindowMinimized) | Qt.WindowState.WindowActive)
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        if self.tray is not None and self.tray.isVisible() and not getattr(self, "quitting", False):
            event.ignore()
            self.hide()
            if not self._told_hidden:
                self._told_hidden = True
                self.tray.showMessage("Digital Unconscious", t("tray.hidden"), mark_icon(), 4000)
            return
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.page is None:
            name, params = self.route
            self.go(name, replace=True, **params)
        self._pace()
        ticker().refresh()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._pace()
        ticker().refresh()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.WindowStateChange, QEvent.Type.ActivationChange):
            self._pace()
            ticker().refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.toast_label.isVisible():
            self.toast_label._place()


def mark_icon():
    from PySide6.QtGui import QIcon

    return QIcon(mark_pixmap(64))
