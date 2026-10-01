"""The notebook of sparks, and one spark with its literature dive."""

from __future__ import annotations

import webbrowser

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QLineEdit, QWidget

from unconscious import api
from unconscious.mind.dive import STEPS as DIVE_STEPS
from unconscious.ui.cards import SparkCard
from unconscious.ui.charts import human
from unconscious.ui.i18n import t
from unconscious.ui.pages.base import Page
from unconscious.ui.pages.today import long_day
from unconscious.ui.theme import THEME, font
from unconscious.ui.widgets import (
    Rule,
    SectionHead,
    Steps,
    button,
    eyebrow,
    hbox,
    label,
    para,
    vbox,
    wrap,
)

TABS = [("new", "spark.new"), ("kept", "spark.kept"), ("pursuing", "spark.pursuing"), ("dismissed", "spark.dismissed"), ("", "sparks.all")]
KIND_MARK = {"search": "⌕", "jot": "✎", "reading": "❡", "note": "·"}


class SparksPage(Page):
    def build(self) -> None:
        status = self.params.get("status", "new")
        query = self.params.get("q", "")
        self.add(self.page_head(t("sparks.title"), t("sparks.sub")))
        counts = {key: len(self.app.store.sparks(status=key or None, limit=1000)) for key, _ in TABS}
        tabs = hbox(spacing=4)
        for key, name in TABS:
            tab = button(f"{t(name)}  {counts[key]}", "on" if key == status else "ghost",
                         lambda k=key: self.window.go("sparks", status=k, q=query))
            tab.setFont(font("small"))
            tabs.addWidget(tab)
        tabs.addStretch(1)
        search = QLineEdit(query)
        search.setPlaceholderText(t("sparks.search"))
        search.setClearButtonEnabled(True)
        search.setFixedWidth(260)
        timer = QTimer(search)
        timer.setSingleShot(True)
        timer.setInterval(350)
        timer.timeout.connect(lambda: self.window.go("sparks", status=status, q=search.text().strip(), replace=True))
        search.textChanged.connect(lambda _text: timer.start())
        tabs.addWidget(search)
        self.add(tabs)
        self.add(Rule("rule"), 10)
        self.body.addSpacing(24)
        sparks = api.sparks_list(self.app, status or None, query or None)
        if not sparks:
            self.add(label(t("sparks.empty"), "quote-s", "muted"))
            return
        self.add(self.cards(sparks, compact=True))
        if query:
            QTimer.singleShot(0, lambda: (search.setFocus(), search.setCursorPosition(len(query))))


class SparkPage(Page):
    def build(self) -> None:
        spark_id = int(self.params["id"])
        spark = api.spark_view(self.app, spark_id)
        if not spark:
            self.add(label(t("common.error"), "body", "muted"))
            return
        self.spark = spark
        self.steps = None
        back = button(f"← {t('sparks.title')}", "ghost", lambda: self.window.back())
        back.setFont(font("small"))
        self.add(hbox(back, "stretch"))
        self.body.addSpacing(12)

        main = vbox(spacing=0)
        card = SparkCard(spark, linked=False)
        card.open_thread.connect(lambda tid: self.window.go("thread", id=tid))
        card.open_spark.connect(lambda sid: self.window.go("spark", id=sid))
        card.feedback.connect(self.window.spark_feedback)
        main.addWidget(card)
        main.addSpacing(40)
        main.addWidget(self._dive_section(spark))
        main.addStretch(1)

        side = vbox(spacing=0)
        if spark.get("dream"):
            side.addWidget(self._side_title(t("spark.fromDream", day=long_day(spark["dream"]["day"]))))
            link = button(spark["dream"]["title"], "link", lambda: self.window.go("today", day=spark["dream"]["day"]))
            link.setFont(font("serif"))
            side.addWidget(_wrapping(link, spark["dream"]["title"]))
            side.addSpacing(28)
        side.addWidget(self._side_title(t("spark.rubric")))
        scores = spark.get("scores") or {}
        if scores:
            for key in ("grounded", "sharp", "fresh", "doable", "fit"):
                value = int(scores.get(key) or 0)
                row = hbox(spacing=10, margins=(0, 4, 0, 4))
                name = label(t(f"rubric.{key}"), "small", "ink2", wrap=False)
                name.setFixedWidth(86)
                row.addWidget(name)
                row.addWidget(_Pips(value), 1)
                row.addWidget(label(str(value), "mono-n", "muted", wrap=False))
                side.addLayout(row)
            if scores.get("generic"):
                side.addWidget(label(t("rubric.generic"), "small", "accent"))
        else:
            side.addWidget(label(t("rubric.none"), "small", "muted"))
        side.addSpacing(28)
        evidence = spark.get("evidence") or []
        if evidence:
            side.addWidget(self._side_title(t("spark.evidence")))
            for index, item in enumerate(evidence, 1):
                text = f"“{item.get('body') or item['label']}”" if item.get("kind") in {"jot", "note"} else item["label"]
                line = hbox(spacing=8, margins=(0, 6, 0, 6))
                line.addWidget(label(str(index), "mono-s", "accent", wrap=False), 0, Qt.AlignmentFlag.AlignTop)
                block = vbox(label(text, "small"), spacing=2)
                meta = " · ".join(x for x in (KIND_MARK.get(item.get("kind"), "") + (" " + item["domain"] if item.get("domain") else ""),
                                              human(item["seconds"]) if item.get("seconds") else "") if x.strip())
                if meta:
                    block.addWidget(label(meta, "mono-s", "muted"))
                line.addLayout(block, 1)
                side.addLayout(line)
                side.addWidget(Rule("rule2"))
            side.addSpacing(28)
        if spark.get("search_terms"):
            side.addWidget(self._side_title(t("spark.searchTerms")))
            for term in spark["search_terms"]:
                side.addWidget(label(term, "mono-n", "ink2"))
        side.addStretch(1)

        side_box = QWidget()
        side_box.setLayout(side)
        side_box.setFixedWidth(300)
        columns = hbox(spacing=44)
        columns.addLayout(main, 1)
        columns.addWidget(side_box, 0, Qt.AlignmentFlag.AlignTop)
        self.add(columns)

    def _side_title(self, text: str) -> QWidget:
        box = vbox(Rule("ink"), eyebrow(text), spacing=10, margins=(0, 0, 0, 10))
        return wrap(box)

    def _dive_section(self, spark: dict) -> QWidget:
        holder = QWidget()
        box = vbox(spacing=0)
        holder.setLayout(box)
        box.addWidget(SectionHead(t("dive.title")))
        box.addSpacing(14)
        job = next((j for j in self.window.state.get("jobs", []) if j["kind"] == "dive" and j["ref"] == str(spark["id"])), None)
        dives = spark.get("dives") or []
        failure = self.window.failures.get(f"dive:{spark['id']}")
        if job:
            frame = QFrame()
            frame.setObjectName("Drop")
            inner = vbox(spacing=8, margins=(18, 16, 18, 16))
            self.steps = Steps(DIVE_STEPS, job.get("step"), night=False)
            inner.addWidget(self.steps)
            frame.setLayout(inner)
            box.addWidget(frame)
            return holder
        if not dives:
            intro = para(t("dive.intro"), "body", "ink2", 150)
            intro.setMaximumWidth(620)
            box.addWidget(intro)
            if failure:
                box.addSpacing(10)
                box.addWidget(label(failure, "small", "error", selectable=True))
            box.addSpacing(16)
            box.addLayout(hbox(button(t("dive.start"), "ink", lambda: self.window.dive(spark["id"])), "stretch"))
            return holder

        dive = dives[0]
        report, papers = dive.get("report") or {}, dive.get("papers") or []
        verdict = report.get("verdict", "unclear")
        head = hbox(spacing=12)
        tag = label(t(f"dive.verdict.{verdict}"), "caption-l", None, wrap=False)
        color = {"open": THEME.c["ok"], "active": THEME.hue(2).name(), "crowded": THEME.c["accent"]}.get(verdict, THEME.c["muted"])
        tag.setStyleSheet(f"color: #ffffff; background: {color}; padding: 4px 12px; border-radius: 12px;")
        head.addWidget(tag)
        head.addWidget(label(t("dive.when", day=(dive.get("created_at") or "")[:10], model=dive.get("model") or ""), "mono-n", "muted", wrap=False))
        head.addStretch(1)
        again = button(t("dive.again"), "line", lambda: self.window.dive(spark["id"]))
        again.setFont(font("small"))
        head.addWidget(again)
        box.addLayout(head)
        if report.get("sharpened_question"):
            box.addSpacing(16)
            box.addWidget(label(report["sharpened_question"], "quote"))
        if report.get("summary"):
            box.addSpacing(12)
            box.addWidget(para(report["summary"], "body", "ink2", 155))
        if report.get("known"):
            box.addWidget(self._dive_label(t("dive.known")))
            for item in report["known"]:
                refs = "".join(f"[{r}]" for r in item.get("refs") or [])
                box.addWidget(_bullet(f"{item['point']}  {refs}"))
        if report.get("gap"):
            box.addWidget(self._dive_label(t("dive.gap")))
            box.addWidget(para(report["gap"], "body", "ink2", 155))
        if report.get("approaches"):
            box.addWidget(self._dive_label(t("dive.approaches")))
            for approach in report["approaches"]:
                row = hbox(spacing=18, margins=(0, 8, 0, 8))
                row.addLayout(vbox(eyebrow(t("dive.design")), para(approach.get("design", ""), "body", None, 145), "stretch", spacing=3), 1)
                row.addLayout(vbox(eyebrow(t("dive.data")), para(approach.get("data", ""), "body", None, 145), "stretch", spacing=3), 1)
                box.addWidget(wrap(row))
                box.addWidget(Rule("rule2"))
        if report.get("next_steps"):
            box.addWidget(self._dive_label(t("dive.next")))
            for index, step in enumerate(report["next_steps"], 1):
                box.addWidget(_bullet(step, f"{index}."))
        if report.get("risks"):
            box.addWidget(self._dive_label(t("dive.risks")))
            for risk in report["risks"]:
                box.addWidget(_bullet(risk))
        if report.get("novelty_note"):
            box.addSpacing(18)
            note = QFrame()
            note.setObjectName("Objection")
            note.setLayout(vbox(label(report["novelty_note"], "question", "muted"), margins=(12, 0, 0, 0)))
            box.addWidget(note)
        if papers:
            box.addWidget(self._dive_label(t("dive.refs")))
            cited = set(report.get("cited") or [])
            for index, paper in enumerate(papers, 1):
                row = hbox(spacing=10, margins=(0, 7, 0, 7))
                row.addWidget(label(f"[{index}]", "mono-n", "accent", wrap=False), 0, Qt.AlignmentFlag.AlignTop)
                block = vbox(spacing=2)
                url = paper.get("url") or ""
                title = button(paper["title"], "link", (lambda u=url: webbrowser.open(u)) if url else None)
                title.setFont(font("body"))
                block.addWidget(_wrapping(title, paper["title"]))
                meta = " · ".join(str(x) for x in (", ".join(paper.get("authors") or []), paper.get("venue"), paper.get("year")) if x)
                if meta:
                    block.addWidget(label(meta, "small", "muted"))
                row.addLayout(block, 1)
                line = wrap(row)
                if cited and index not in cited:
                    line.setStyleSheet("QLabel, QPushButton { color: palette(placeholder-text); }")
                box.addWidget(line)
                box.addWidget(Rule("rule2"))
        return holder

    def _dive_label(self, text: str) -> QWidget:
        return wrap(vbox(eyebrow(text), margins=(0, 24, 0, 8)))

    def on_state(self, state: dict) -> None:
        job = next((j for j in state.get("jobs", []) if j["kind"] == "dive" and j["ref"] == str(self.params.get("id"))), None)
        if job and self.steps is not None:
            self.steps.set_current(job.get("step"))
        elif job and self.steps is None:
            self.refresh()


class _Pips(QWidget):
    def __init__(self, value: int):
        super().__init__()
        self.value = value
        self.setFixedHeight(8)

    def paintEvent(self, _event) -> None:
        from PySide6.QtGui import QPainter

        p = QPainter(self)
        gap = 3
        width = (self.width() - gap * 4) / 5
        for i in range(5):
            p.fillRect(int(i * (width + gap)), 1, int(width), 6, THEME.color("ink" if i < self.value else "rule2"))


def _bullet(text: str, mark: str = "—") -> QWidget:
    row = hbox(spacing=10, margins=(0, 3, 0, 3))
    bullet = label(mark, "body", "muted", wrap=False)
    bullet.setFixedWidth(18)
    row.addWidget(bullet, 0, Qt.AlignmentFlag.AlignTop)
    row.addWidget(para(text, "body", "ink2", 150), 1)
    return wrap(row)


def _wrapping(button_widget, text: str) -> QWidget:
    """QPushButton cannot wrap, so titles become clickable wrapped labels."""
    from unconscious.ui.cards import TitleLabel

    title = TitleLabel(text, "serif")
    title.clicked.connect(button_widget.click)
    return title
