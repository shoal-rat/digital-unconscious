"""Journal: one line per night, like the contents page of a diary."""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from unconscious import api
from unconscious.ui.charts import ShoreThumb, shore_threads
from unconscious.ui.i18n import language, t
from unconscious.ui.pages.base import Page
from unconscious.ui.widgets import Clickable, Rule, hbox, label, vbox


class JournalPage(Page):
    def build(self) -> None:
        self.add(self.page_head(t("journal.title"), t("journal.sub")))
        dreams = api.dreams_list(self.app)
        if not dreams:
            self.add(label(t("journal.empty"), "quote-s", "muted"))
            return
        for dream in dreams:
            d = date.fromisoformat(dream["day"])
            row = Clickable()
            line = hbox(spacing=26, margins=(0, 18, 0, 18))
            full = self.app.store.dream(dream["day"]) or {}
            payload = full.get("payload") or {}
            line.addWidget(ShoreThumb(shore_threads(payload.get("topics") or [], payload.get("signals") or []), dream["day"]), 0, Qt.AlignmentFlag.AlignTop)
            when = vbox(spacing=2)
            when.addWidget(label(str(d.day), "display-m", wrap=False))
            if language() == "zh":
                sub = f"{d.month}月 · 周{'一二三四五六日'[d.weekday()]}"
            else:
                sub = f"{d.strftime('%b')} · {d.strftime('%a')}"
            when.addWidget(label(sub, "caption", "muted", wrap=False))
            stamp = QWidget()
            stamp.setFixedWidth(110)
            stamp.setLayout(when)
            line.addWidget(stamp, 0, Qt.AlignmentFlag.AlignTop)
            text = vbox(spacing=8)
            text.addWidget(label(dream["title"], "display-s"))
            if dream.get("undercurrent"):
                text.addWidget(label(dream["undercurrent"], "quote-s", "ink2"))
            line.addLayout(text, 1)
            line.addWidget(label(t("journal.sparks", n=dream["spark_count"]), "caption", "muted", wrap=False), 0, Qt.AlignmentFlag.AlignTop)
            row.setLayout(line)
            row.clicked.connect(lambda day=dream["day"]: self.window.go("today", day=day))
            self.add(row)
            self.add(Rule("rule"))
