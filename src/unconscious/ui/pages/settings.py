"""Settings: who you are, what the sensor may see, when it dreams, which models think, and memory."""

from __future__ import annotations

import platform
from datetime import date, timedelta

from PySide6.QtCore import QDate, Qt, QTime, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QTimeEdit,
    QWidget,
)

from unconscious import api, ingest
from unconscious.llm.router import ROLE_PREFERENCES, ROLES
from unconscious.ui.dialogs import SharkDialog
from unconscious.ui.i18n import crew_message, region_name, t, trouble_message
from unconscious.ui.pages.base import Page
from unconscious.ui.theme import font
from unconscious.ui.widgets import FoldList, SectionHead, button, eyebrow, hbox, label, vbox, wrap

MODEL_CHOICES = [
    "claude:claude-opus-5-5", "claude:claude-sonnet-5-5", "codex", "anthropic:claude-opus-5-5", "anthropic:claude-sonnet-5-5",
    "deepseek:deepseek-v4-flash", "deepseek:deepseek-v4-pro", "glm", "kimi", "openai", "ollama",
]


def field(title: str, widget: QWidget, hint: str = "") -> QWidget:
    box = vbox(spacing=6, margins=(0, 0, 0, 16))
    box.addWidget(eyebrow(title))
    box.addWidget(widget)
    if hint:
        box.addWidget(label(hint, "small", "muted"))
    box.addStretch(1)
    return wrap(box)


class TopGrid(QGridLayout):
    """A grid whose cells hug the top, so labels line up when one field has a hint."""

    def addWidget(self, widget, row, column, rowspan=1, colspan=1, alignment=Qt.AlignmentFlag.AlignTop):  # noqa: N802
        super().addWidget(widget, row, column, rowspan, colspan, alignment)


def lines_edit(values: list[str], height: int = 84) -> QPlainTextEdit:
    edit = QPlainTextEdit("\n".join(values))
    edit.setFixedHeight(height)
    edit.setFont(font("body"))
    return edit


def lines_of(edit: QPlainTextEdit) -> list[str]:
    return [line.strip() for line in edit.toPlainText().splitlines() if line.strip()]


def spin(value: int, low: int, high: int) -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    box.setFixedWidth(120)
    return box


class Segmented(QWidget):
    def __init__(self, options: list[tuple[str, str]], value: str):
        super().__init__()
        self.value = value
        self.buttons = {}
        row = hbox(spacing=0)
        for key, text in options:
            b = button(text, "on" if key == value else "line", lambda k=key: self.pick(k))
            b.setFont(font("small"))
            self.buttons[key] = b
            row.addWidget(b)
        row.addStretch(1)
        self.setLayout(row)

    def pick(self, key: str) -> None:
        self.value = key
        for k, b in self.buttons.items():
            b.setProperty("kind", "on" if k == key else "line")
            b.style().unpolish(b)
            b.style().polish(b)


class DropZone(QFrame):
    def __init__(self, on_file):
        super().__init__()
        self.on_file = on_file
        self.setObjectName("Drop")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        box = vbox(spacing=4, margins=(24, 26, 24, 26))
        title = label(t("settings.drop"), "card-title-s")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = label(t("settings.dropHint"), "small", "muted")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(title)
        box.addWidget(hint)
        self.setLayout(box)

    def _over(self, on: bool) -> None:
        self.setProperty("over", "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._over(True)

    def dragLeaveEvent(self, _event) -> None:
        self._over(False)

    def dropEvent(self, event) -> None:
        self._over(False)
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.on_file(url.toLocalFile())

    def mouseReleaseEvent(self, _event) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, t("settings.drop"), "", "Documents (*.pdf *.md *.markdown *.txt *.json *.jsonl *.rst *.tex)")
        for path in paths:
            self.on_file(path)


class SettingsPage(Page):
    def build(self) -> None:
        self.view = api.settings_view(self.app)
        s = self.view["settings"]
        self.add(self.page_head(t("settings.title"), t("settings.sub")))
        self._you(s)
        self._senses(s)
        self._dreaming(s)
        self._models(s)
        self._memory()

    def _panel(self, title: str, note: str) -> None:
        self.add(SectionHead(title), 8)
        self.body.addSpacing(6)
        text = label(note, "body", "muted")
        text.setMaximumWidth(640)
        self.add(text)
        self.body.addSpacing(18)

    def _save_row(self, on_save) -> None:
        self.add(hbox(button(t("common.save"), "ink", on_save), "stretch"), 4)
        self.body.addSpacing(44)

    def _save(self, patch: dict) -> None:
        changed = self.app.update_settings(patch)
        self.window.toast(t("common.saved"))
        if any(key.startswith(("you.language", "ui.theme", "ui.motion")) for key in changed):
            self.window.apply_preferences()
        else:
            self.refresh()

    # -- you -----------------------------------------------------------------

    def _you(self, s: dict) -> None:
        self._panel(t("settings.you"), t("settings.youNote"))
        name = QLineEdit(s["you"]["name"])
        persona = QPlainTextEdit(s["you"]["persona"])
        persona.setFixedHeight(96)
        persona.setFont(font("serif"))
        persona.setPlaceholderText(t("settings.personaHint"))
        focus = QLineEdit(", ".join(s["you"]["focus"]))
        language = Segmented([("auto", "Auto"), ("en", "English"), ("zh", "中文")], s["you"]["language"])
        theme = Segmented([("system", t("settings.theme.system")), ("light", t("settings.theme.light")), ("dark", t("settings.theme.dark"))], s["ui"]["theme"])
        motion = Segmented([("auto", t("settings.motion.auto")), ("full", t("settings.motion.full")),
                            ("calm", t("settings.motion.calm")), ("off", t("settings.motion.off"))], s["ui"]["motion"])
        grid = TopGrid()
        grid.setHorizontalSpacing(24)
        grid.addWidget(field(t("settings.name"), name), 0, 0)
        grid.addWidget(field(t("settings.focus"), focus, t("settings.focusHint")), 0, 1)
        grid.addWidget(field(t("settings.persona"), persona), 1, 0, 1, 2)
        grid.addWidget(field(t("settings.language"), language), 2, 0)
        grid.addWidget(field(t("settings.theme"), theme), 2, 1)
        grid.addWidget(field(t("settings.motion"), motion, t("settings.motionHint")), 3, 0, 1, 2)
        self.add(grid)
        self._save_row(lambda: self._save({
            "you": {"name": name.text(), "persona": persona.toPlainText(), "focus": focus.text(), "language": language.value},
            "ui": {"theme": theme.value, "motion": motion.value},
        }))

    # -- senses --------------------------------------------------------------

    def _senses(self, s: dict) -> None:
        self._panel(t("settings.senses"), t("settings.sensesNote"))
        sensor = self.view["sensor"]
        caps = sensor.get("capabilities") or {}
        state_line = t(f"sensor.{sensor.get('state', 'never')}")
        if sensor.get("subject"):
            state_line += f" · {sensor['subject']}"
        self.add(label(state_line, "card-title-s"))
        self.body.addSpacing(10)
        rows = [(t("cap.app"), caps.get("app")), (t("cap.titles"), caps.get("titles")), (t("cap.idle"), caps.get("idle"))]
        urls = caps.get("urls") or {}
        if urls:
            rows += [(t("cap.urls", b=b), v == "ok") for b, v in urls.items()]
        else:
            rows.append((t("cap.urlsUnknown"), None))
        for text, ok in rows:
            mark = "✓" if ok else ("·" if ok is None else "✕")
            tone = "ink2" if ok else ("muted" if ok is None else "accent")
            line = hbox(label(mark, "mono-n", tone, wrap=False), label(text, "body", "ink2" if ok else "muted"), "stretch", spacing=10)
            self.add(line, 4)
        if sensor.get("error") or (caps and not caps.get("titles") and platform.system() == "Darwin"):
            self.add(label(sensor.get("error") or t("cap.permission"), "small", "error", selectable=True), 10)
        pause = hbox(spacing=8)
        if sensor.get("paused"):
            pause.addWidget(button(t("settings.resume"), "ink", lambda: self._pause(None)))
        else:
            pause.addWidget(button(t("settings.pause1"), "line", lambda: self._pause(1)))
            pause.addWidget(button(t("settings.pauseDay"), "line", lambda: self._pause("tomorrow")))
        pause.addStretch(1)
        self.add(pause, 14)
        self.body.addSpacing(18)
        sense = s["sense"]
        titles = QCheckBox(t("settings.titles"))
        titles.setChecked(sense["capture_titles"])
        urls_box = QCheckBox(t("settings.urls"))
        urls_box.setChecked(sense["capture_urls"])
        saver = QCheckBox(t("settings.batterySaver"))
        saver.setChecked(sense["battery_saver"])
        self.add(titles)
        self.add(urls_box, 8)
        self.add(saver, 8)
        self.body.addSpacing(16)
        interval, idle, retention = spin(sense["interval_seconds"], 5, 300), spin(sense["idle_seconds"], 30, 3600), spin(sense["retention_days"], 7, 3650)
        quiet_apps, private_apps, quiet_domains = lines_edit(sense["quiet_apps"]), lines_edit(sense["private_apps"], 120), lines_edit(sense["quiet_domains"])
        grid = TopGrid()
        grid.setHorizontalSpacing(24)
        grid.addWidget(field(t("settings.interval"), interval), 0, 0)
        grid.addWidget(field(t("settings.idle"), idle), 0, 1)
        grid.addWidget(field(t("settings.quietApps"), quiet_apps), 1, 0)
        grid.addWidget(field(t("settings.quietDomains"), quiet_domains), 1, 1)
        grid.addWidget(field(t("settings.privateApps"), private_apps), 2, 0)
        grid.addWidget(field(t("settings.retention"), retention, t("settings.retentionHint")), 2, 1)
        self.add(grid)
        self._save_row(lambda: self._save({"sense": {
            "capture_titles": titles.isChecked(), "capture_urls": urls_box.isChecked(), "battery_saver": saver.isChecked(),
            "interval_seconds": interval.value(), "idle_seconds": idle.value(), "retention_days": retention.value(),
            "quiet_apps": lines_of(quiet_apps), "private_apps": lines_of(private_apps), "quiet_domains": lines_of(quiet_domains),
        }}))

    def _pause(self, hours) -> None:
        from datetime import datetime

        if hours is None:
            self.app.store.put("sensor_paused_until", None)
        elif hours == "tomorrow":
            tomorrow = datetime.combine(date.today() + timedelta(days=1), datetime.min.time()).astimezone().replace(hour=7)
            self.app.store.put("sensor_paused_until", tomorrow.isoformat(timespec="seconds"))
        else:
            until = datetime.now().astimezone() + timedelta(hours=hours)
            self.app.store.put("sensor_paused_until", until.isoformat(timespec="seconds"))
        self.window.poll(force=True)
        self.refresh()

    # -- dreaming ------------------------------------------------------------

    def _dreaming(self, s: dict) -> None:
        self._panel(t("settings.dreaming"), t("settings.dreamingNote"))
        d = s["dream"]
        when = QTimeEdit(QTime.fromString(d["time"], "HH:mm"))
        when.setDisplayFormat("HH:mm")
        when.setFixedWidth(120)
        auto = QCheckBox(t("settings.auto"))
        auto.setChecked(d["auto"])
        critique = QCheckBox(t("settings.critique"))
        critique.setChecked(d["critique"])
        sparks, candidates = spin(d["sparks"], 1, 6), spin(d["candidates"], 1, 10)
        grid = TopGrid()
        grid.setHorizontalSpacing(24)
        grid.addWidget(field(t("settings.time"), when), 0, 0)
        grid.addWidget(field(t("settings.sparks"), sparks), 0, 1)
        grid.addWidget(field(t("settings.candidates"), candidates), 0, 2)
        grid.setColumnStretch(3, 1)
        self.add(grid)
        self.add(auto)
        self.add(critique, 8)
        self._save_row(lambda: self._save({"dream": {
            "time": when.time().toString("HH:mm"), "auto": auto.isChecked(), "critique": critique.isChecked(),
            "sparks": sparks.value(), "candidates": candidates.value(),
        }}))

    # -- models --------------------------------------------------------------

    def _models(self, s: dict) -> None:
        self._panel(t("settings.models"), t("settings.modelsNote"))
        models = self.view["models"]
        available = models["available"]
        row = hbox(spacing=16)
        row.addWidget(eyebrow(t("settings.available")))
        for name, ok in available.items():
            if name == "ollama" and not ok:
                continue
            row.addWidget(label(("✓ " if ok else "· ") + name, "mono-n", "ink2" if ok else "faint", wrap=False))
        row.addStretch(1)
        self.add(row)
        self.body.addSpacing(16)
        grid = TopGrid()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        grid.addWidget(eyebrow(t("settings.choice")), 0, 1)
        grid.addWidget(eyebrow(t("settings.chain")), 0, 2)
        combos = {}
        for index, role in enumerate(ROLES, 1):
            grid.addWidget(label(t(f"settings.role.{role}"), "body", wrap=False), index, 0)
            combo = QComboBox()
            combo.addItem(t("settings.autoModel"), "auto")
            configured = s["models"][role]
            choices = list(dict.fromkeys(MODEL_CHOICES + ROLE_PREFERENCES[role] + ([configured] if configured != "auto" else [])))
            for spec in choices:
                name = spec.split(":")[0]
                combo.addItem(spec + ("" if available.get(name) else "  (not aboard)"), spec)
            combo.setCurrentIndex(max(0, combo.findData(configured)))
            combo.setMinimumWidth(240)
            combos[role] = combo
            grid.addWidget(combo, index, 1)
            chain = " → ".join(models["routes"][role]["chain"]) or t("settings.nothing")
            grid.addWidget(label(chain, "mono-n", "ink2" if models["routes"][role]["chain"] else "accent"), index, 2)
        grid.setColumnStretch(2, 1)
        self.add(grid)
        fallback = QCheckBox(t("settings.fallback"))
        fallback.setChecked(s["models"]["fallback"])
        self.add(fallback, 14)
        research = QCheckBox(t("settings.research"))
        research.setChecked(s["models"]["research"])
        self.add(research, 6)
        self.add(label(t("settings.researchHint"), "small", "muted"), 4)
        guard = QCheckBox(t("settings.regionGuard"))
        guard.setChecked(s["models"]["region_guard"])
        self.add(guard, 10)
        region = models.get("region") or {}
        country = region.get("country")
        if not country:
            status = t("region.never")
        elif country == "?":
            status = t("region.unknown")
        elif region.get("ashore"):
            status = t("region.held", region=region_name(country))
        else:
            status = t("region.clear", region=region_name(country))
        self.add(label(t("settings.regionHint") + " " + status, "small", "muted"), 4)
        troubles = [tr for tr in (self.window.state.get("crew") or []) if tr["kind"] != "region"]
        if troubles:
            self.add(eyebrow(t("crew.troubles")), 16)
            for trouble in troubles:
                self.add(label(trouble_message(trouble["kind"], trouble["who"], trouble["detail"]), "small", "ink2", selectable=True), 4)
        usage = self.view.get("usage") or []
        if usage:
            self.add(eyebrow(t("settings.usage")), 18)
            lines = []
            for item in usage:
                tokens = int((item.get("tokens_in") or 0) + (item.get("tokens_out") or 0))
                lines.append(label(f"{item['role']:<9} {item['provider']}:{item['model']}  ·  " + t("settings.calls", n=item["calls"], tokens=f"{tokens:,}"),
                                   "mono-n", "ink2"))
            self.add(FoldList(lines, keep=6, key="usage", spacing=4), 4)
        error = self.view.get("last_error")
        if error:
            self.add(eyebrow(t("settings.lastError")), 16)
            self.add(label(f"{error['ts'][:16]} · {error['provider']} · {crew_message(error['error'])}", "small", "error", selectable=True), 6)
        self._save_row(lambda: self._save({"models": {
            **{role: combo.currentData() for role, combo in combos.items()}, "fallback": fallback.isChecked(),
            "region_guard": guard.isChecked(), "research": research.isChecked(),
        }}))

    # -- memory --------------------------------------------------------------

    def _memory(self) -> None:
        self._panel(t("settings.memory"), t("settings.memoryNote"))
        self.add(DropZone(self._feed))
        self.body.addSpacing(22)
        aw_day, forget_day = QDateEdit(QDate.currentDate()), QDateEdit(QDate.currentDate())
        for edit in (aw_day, forget_day):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd")
            edit.setFixedWidth(150)
        grid = TopGrid()
        grid.setHorizontalSpacing(24)
        grid.addWidget(field(t("settings.aw"), wrap(hbox(aw_day, button(t("settings.awButton"), "line", lambda: self._import_aw(aw_day)), "stretch"))), 0, 0)
        grid.addWidget(field(t("settings.forgetDay"), wrap(hbox(forget_day, button(t("settings.forgetButton"), "line", lambda: self._forget_day(forget_day)), "stretch"))), 0, 1)
        self.add(grid)
        taste = self.view.get("taste") or {}
        line = t("settings.tasteLine", k=taste.get("kept", 0), d=taste.get("dismissed", 0)) if (taste.get("kept") or taste.get("dismissed")) else t("settings.tasteNone")
        self.add(field(t("settings.taste"), label(line, "body", "ink2")))
        size = (self.view.get("memory") or {}).get("bytes", 0)
        amount = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{max(1, round(size / 1e3))} KB"
        self.add(field(t("settings.size"), label(t("settings.sizeLine", size=amount), "body", "ink2")))
        home = self.view["home"]
        self.add(field(t("settings.home"), wrap(hbox(label(home, "mono-n", "ink2", selectable=True),
                                                       button(t("settings.openFolder"), "ghost", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(home))),
                                                       "stretch"))))
        self.body.addSpacing(20)
        self.add(SectionHead(t("settings.forgetAll")))
        self.add(label(t("settings.forgetAllHint"), "body", "muted"), 8)
        confirm = QLineEdit()
        confirm.setPlaceholderText(t("settings.forgetPhrase"))
        confirm.setFixedWidth(260)
        erase = button(t("settings.forgetAll"), "accent", lambda: self._forget_all(confirm))
        erase.setEnabled(False)
        confirm.textChanged.connect(lambda text: erase.setEnabled(text.strip() == t("settings.forgetPhrase")))
        self.add(hbox(confirm, erase, "stretch"), 12)

    def _feed(self, path: str) -> None:
        try:
            ingest.feed_path(self.app, path)
            self.window.toast(t("settings.fed", name=path.rsplit("/", 1)[-1]))
        except ingest.IngestError as exc:
            self.window.toast(str(exc), error=True)

    def _import_aw(self, edit: QDateEdit) -> None:
        try:
            result = ingest.import_activitywatch(self.app, edit.date().toString("yyyy-MM-dd"))
            self.window.toast(f"ActivityWatch: {result['traces']}")
        except ingest.IngestError as exc:
            self.window.toast(str(exc), error=True)

    def _forget_day(self, edit: QDateEdit) -> None:
        day = edit.date().toString("yyyy-MM-dd")
        if SharkDialog(self, t("shark.title", day=day), t("shark.body")).exec() == SharkDialog.DialogCode.Accepted:
            self.app.store.forget_day(day)
            self.window.toast(t("settings.forgot"))

    def _forget_all(self, confirm: QLineEdit) -> None:
        if confirm.text().strip() != t("settings.forgetPhrase"):
            return
        if SharkDialog(self, t("shark.allTitle"), t("shark.allBody")).exec() == SharkDialog.DialogCode.Accepted:
            self.app.store.forget_everything()
            self.window.toast(t("settings.forgot"))
            self.window.go("today")
