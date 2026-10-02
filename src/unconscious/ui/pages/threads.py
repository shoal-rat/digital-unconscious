"""Threads (the strata of recurring concerns) and a single thread's page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QComboBox, QInputDialog, QMessageBox, QWidget

from unconscious import api
from unconscious.ui.charts import HistoryBars, Strata, human
from unconscious.ui.i18n import t
from unconscious.ui.pages.base import Page
from unconscious.ui.pages.today import short_day
from unconscious.ui.theme import font
from unconscious.ui.widgets import (
    Clickable,
    ElidedLabel,
    FlowLayout,
    FoldList,
    Rule,
    SectionHead,
    Stamp,
    Swatch,
    button,
    eyebrow,
    hbox,
    label,
    para,
    vbox,
    wrap,
)


class ThreadsPage(Page):
    def build(self) -> None:
        view = api.threads_view(self.app)
        self.add(self.page_head(t("threads.title"), t("threads.sub")))
        shown = [s for s in view["signals"] if s["kind"] != "steady"]
        if shown:
            self.add(SectionHead(t("threads.signals"), note=t("threads.signalsNote")))
            self.body.addSpacing(8)
            for signal in shown:
                row = Clickable()
                line = hbox(spacing=16, margins=(0, 11, 0, 11))
                stamp_box = QWidget()
                stamp_box.setFixedWidth(118)
                stamp_box.setLayout(hbox(Stamp(signal["kind"]), "stretch"))
                line.addWidget(stamp_box, 0, Qt.AlignmentFlag.AlignTop)
                line.addWidget(label(_without_kind(signal["text"]), "serif"), 1)
                row.setLayout(line)
                target = signal["threads"][0]
                row.clicked.connect(lambda tid=target: self.window.go("thread", id=tid))
                self.add(row)
                self.add(Rule("rule2"))
            self.body.addSpacing(44)
        if not view["threads"]:
            self.add(label(t("threads.none"), "quote-s", "muted"))
            return
        strata = Strata(view)
        strata.thread_clicked.connect(lambda tid: self.window.go("thread", id=tid))
        self.add(strata)


class ThreadPage(Page):
    def build(self) -> None:
        thread_id = int(self.params["id"])
        th = api.thread_view(self.app, thread_id)
        if not th:
            self.add(label(t("common.error"), "body", "muted"))
            return
        self.thread = th
        back = button(f"← {t('threads.title')}", "ghost", lambda: self.window.go("threads"))
        back.setFont(font("small"))
        self.add(hbox(back, "stretch"))
        self.body.addSpacing(10)

        title_row = hbox(spacing=14)
        title_row.addWidget(Swatch(th.get("hue"), 16), 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addWidget(label(th["name"], "display-l"), 1)
        actions = FlowLayout(spacing=6, one_line=True)  # the buttons wrap under each other in a small window
        pinned, muted = th.get("state") == "pinned", th.get("state") == "muted"
        actions.addWidget(button(t("thread.rename"), "line", self._rename))
        actions.addWidget(button(t("thread.unpin") if pinned else t("thread.pin"), "on" if pinned else "line",
                                 lambda: self._set_state("active" if pinned else "pinned")))
        actions.addWidget(button(t("thread.unmute") if muted else t("thread.mute"), "on" if muted else "line",
                                 lambda: self._set_state("active" if muted else "muted")))
        if th.get("others"):
            merge = QComboBox()
            # sized to a short name, not the longest current's, so the row fits a small window
            merge.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            merge.setMinimumContentsLength(14)
            merge.addItem(t("thread.merge"), None)
            for other in th["others"]:
                merge.addItem(other["name"], other["id"])
            merge.currentIndexChanged.connect(lambda _i, box=merge: self._merge(box))
            actions.addWidget(merge)
        for i in range(actions.count()):
            actions.itemAt(i).widget().setFont(font("small"))
        holder = QWidget()
        holder.setLayout(actions)
        title_row.addWidget(holder, 0, Qt.AlignmentFlag.AlignVCenter)
        self.add(title_row)
        if th.get("gist"):
            gist = para(th["gist"], "reading", "ink2", 150)
            gist.setMaximumWidth(700)
            self.add(gist, 8)
        if th.get("keywords"):
            flow = FlowLayout(spacing=6)
            for keyword in th["keywords"]:
                tag = ElidedLabel(keyword, "mono-n", "ink2")
                tag.setStyleSheet("padding: 2px 7px; border-radius: 3px; background: palette(alternate-base);")
                flow.addWidget(tag)
            holder = QWidget()
            holder.setLayout(flow)
            self.add(holder, 12)
        if muted:
            self.add(label(t("thread.mutedNote"), "small", "error"), 14)
        if th.get("signals"):
            row = hbox(*[Stamp(k) for k in th["signals"]], "stretch", spacing=6)
            self.add(row, 16)

        stats = hbox(spacing=0)
        for value, key in (
            (str(th.get("active_14", 0)), "thread.stats.days"),
            (human(th.get("total", 0)), "thread.stats.total"),
            (human(th.get("median", 0)), "thread.stats.median"),
            (short_day(th["first_day"]) if th.get("first_day") else "—", "thread.stats.first"),
        ):
            cell = vbox(label(value, "num", wrap=False), eyebrow(t(key)), spacing=2, margins=(0, 14, 24, 12))
            stats.addLayout(cell)
            stats.addStretch(1)
        self.add(Rule("rule"), 26)
        self.add(stats)
        self.add(Rule("rule"))

        self.add(SectionHead(t("thread.history")), 40)
        self.body.addSpacing(16)
        self.add(HistoryBars(th.get("series") or [], th.get("days") or [], th.get("hue")))

        left = vbox(spacing=0)
        left.addWidget(SectionHead(t("thread.notes")))
        left.addSpacing(8)
        history = th.get("history") or []
        whens = [f"{short_day(entry['day'])} · {human(entry['seconds'])}" for entry in history]
        when_width = max([0] + [QFontMetrics(font("mono-n")).horizontalAdvance(w) for w in whens]) + 4
        notes = []
        for entry, stamp in zip(history, whens, strict=True):
            line = hbox(spacing=14, margins=(0, 9, 0, 9))
            when = label(stamp, "mono-n", "muted", wrap=False)
            when.setFixedWidth(when_width)  # as wide as the longest date, in either language
            line.addWidget(when, 0, Qt.AlignmentFlag.AlignTop)
            text = vbox(spacing=3)
            if entry.get("note"):
                text.addWidget(label(entry["note"], "body", "ink2"))
            if entry.get("subjects"):
                text.addWidget(label(" · ".join(entry["subjects"][:4]), "small", "muted"))
            line.addLayout(text, 1)
            notes.append(wrap(vbox(wrap(line), Rule("rule2"), spacing=0)))
        if notes:
            left.addWidget(FoldList(notes, keep=20, key=f"history:{thread_id}"))
        left.addStretch(1)
        right = vbox(spacing=0)
        right.addWidget(SectionHead(t("thread.subjects")))
        right.addSpacing(8)
        counts = th.get("subjects") or []
        count_width = max([0] + [QFontMetrics(font("mono-n")).horizontalAdvance(f"{c}×") for _, c in counts]) + 4
        for name, count in counts:
            line = hbox(spacing=12, margins=(0, 8, 0, 8))
            n = label(f"{count}×", "mono-n", "muted", wrap=False)
            n.setFixedWidth(count_width)
            line.addWidget(n)
            line.addWidget(para(name, "body", None, 140), 1)
            right.addWidget(wrap(line))
            right.addWidget(Rule("rule2"))
        right.addStretch(1)
        columns = hbox(spacing=44)
        columns.addLayout(left, 6)
        columns.addLayout(right, 5)
        self.add(columns, 44)

        if th.get("sparks"):
            self.add(SectionHead(t("thread.sparks"), len(th["sparks"])), 48)
            self.body.addSpacing(18)
            self.add(self.cards(th["sparks"], compact=True))

    def _set_state(self, state: str) -> None:
        self.app.store.update_thread(self.thread["id"], state=state)
        self.refresh()

    def _rename(self) -> None:
        name, ok = QInputDialog.getText(self, t("thread.renameTitle"), t("thread.rename"), text=self.thread["name"])
        if ok and name.strip():
            self.app.store.update_thread(self.thread["id"], name=name.strip()[:80])
            self.refresh()

    def _merge(self, box: QComboBox) -> None:
        target = box.currentData()
        if target is None:
            return
        answer = QMessageBox.question(self, t("thread.merge"), t("thread.mergeConfirm", a=self.thread["name"], b=box.currentText()))
        if answer == QMessageBox.StandardButton.Yes:
            self.app.store.merge_threads(self.thread["id"], int(target))
            self.window.go("thread", id=int(target))
        else:
            box.setCurrentIndex(0)


def _without_kind(text: str) -> str:
    """The stamp already names the pattern; drop the sentence's own 'An eddy:' prefix."""
    for mark in ("：", ": "):
        head, sep, rest = text.partition(mark)
        if sep and len(head) <= 16:
            return rest[:1].upper() + rest[1:] if rest[:1].isascii() else rest
    return text
