"""Today (or any day): the dream at night-panel scale, its sparks, then the day's residue."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLineEdit, QWidget

from unconscious import api
from unconscious.jobs import dream_moment
from unconscious.mind.dream import STEPS as DREAM_STEPS
from unconscious.store import today
from unconscious.ui.charts import Pebbles, Ribbon, human, shore_threads
from unconscious.ui.i18n import language, t, trouble_message
from unconscious.ui.pages.base import Page
from unconscious.ui.theme import font
from unconscious.ui.widgets import (
    ElidedLabel,
    FlowLayout,
    Fold,
    FoldList,
    MiniBar,
    NightPanel,
    SectionHead,
    Stamp,
    Steps,
    Swatch,
    TitleLabel,
    Unfold,
    button,
    capped,
    eyebrow,
    hbox,
    label,
    para,
    vbox,
    wrap,
)

KIND_MARK = {"search": "⌕", "jot": "✎", "reading": "❡", "note": "·"}


def long_day(day: str) -> str:
    d = date.fromisoformat(day)
    if language() == "zh":
        weekdays = "一二三四五六日"
        return f"{d.year}年{d.month}月{d.day}日 星期{weekdays[d.weekday()]}"
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B %Y')}"


def short_day(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d.month}月{d.day}日" if language() == "zh" else f"{d.day} {d.strftime('%b')}"


class TodayPage(Page):
    def build(self) -> None:
        app, state = self.app, self.window.state
        day = self.params.get("day") or today()
        self.day = day
        is_today = day == today()
        dream = api.dream_view(app, day)
        view = api.day_view(app, day)
        threads = api.threads_view(app, day)
        signals = {th["id"]: th["signals"] for th in threads["threads"]}
        job = next((j for j in state.get("jobs", []) if j["kind"] == "dream" and j["ref"] == day), None)
        self.steps = None

        # masthead
        prev_day = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        next_day = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        nav = hbox(spacing=4)
        nav.addWidget(button(f"← {short_day(prev_day)}", "ghost", lambda: self.window.go("today", day=prev_day)))
        if not is_today:
            nav.addWidget(button(f"{short_day(next_day)} →", "ghost", lambda: self.window.go("today", day=next_day)))
            nav.addWidget(button(t("common.today"), "line", lambda: self.window.go("today")))
        for i in range(nav.count()):
            nav.itemAt(i).widget().setFont(font("small"))
        self.add(hbox(eyebrow(long_day(day)), "stretch", nav))
        self.body.addSpacing(14)

        if dream:
            self.add(self._dream_panel(dream, is_today, job))
            if job:
                self.add(self._steps_panel(job), 12)
            self.add(SectionHead(t("today.sparks"), len(dream["sparks"]), t("today.sparksNote")), 52)
            self.body.addSpacing(20)
            self.add(self.cards(dream["sparks"]))
            if dream.get("earlier"):  # the day dived more than once: nothing of the earlier dives is lost
                self.add(SectionHead(t("today.earlier"), len(dream["earlier"]), t("today.earlierNote")), 52)
                for dive in dream["earlier"]:
                    self.add(self._earlier_dive(dive), 12)
        elif job:
            self.add(self._steps_panel(job, standalone=True))
        elif not state.get("has_memory"):
            self.add(self._onboarding(state))
        else:
            self.add(self._waiting(state, view, day))
            latest = state.get("latest_dream")
            if latest and latest != day:
                previous = api.dream_view(app, latest)
                if previous:
                    self._last_dream(previous)

        self._residue(view, signals, is_today)

    # -- night panels --------------------------------------------------------

    def _dream_panel(self, dream: dict, is_today: bool, job: dict | None) -> QWidget:
        panel = NightPanel()
        b = panel.body
        stats = (dream.get("payload") or {}).get("stats") or {}
        models = dream.get("models") or {}
        night = datetime.fromisoformat(dream["created_at"]).astimezone() >= dream_moment(self.app, dream["day"])
        heading = t("today.ofDay") if not is_today else t("today.tonight") if night else t("today.todays")
        b.addWidget(label(f"{heading} · {t('today.dream', n=dream.get('number') or '')} · {short_day(dream['day'])}", "caption-l", "muted"))
        b.addSpacing(22)
        title = label(dream.get("title") or "", "display-xl")
        title.setMaximumWidth(860)
        b.addWidget(title)
        b.addSpacing(22)
        reflection = para(dream.get("reflection") or "", "reading", "body", 165)
        reflection.setMaximumWidth(660)
        b.addWidget(Fold(reflection, lines=6, key=f"reflection:{dream['day']}"))
        if dream.get("undercurrent"):
            b.addSpacing(34)
            mark = label("“", "quote", "accent", wrap=False)
            f = mark.font()
            f.setPixelSize(72)
            mark.setFont(f)
            mark.setFixedWidth(40)
            mark.setAlignment(Qt.AlignmentFlag.AlignTop)
            quote = vbox(eyebrow(t("today.undercurrent"), "accent"), label(dream["undercurrent"], "quote"), spacing=6)
            row = hbox(spacing=8)
            row.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
            row.addLayout(quote, 1)
            b.addLayout(capped(wrap(row), 760))
        b.addSpacing(30)
        meta = hbox(spacing=18)
        parts = []
        if stats.get("seconds"):
            parts.append(t("today.meta.attention", time=human(stats["seconds"])))
        if stats.get("threads"):
            parts.append(t("today.meta.threads", n=stats["threads"]))
        if stats.get("candidates"):
            parts.append(t("today.meta.kept", k=stats.get("kept", len(dream.get("sparks") or [])), c=stats["candidates"]))
        if (dream.get("payload") or {}).get("demo"):
            parts.append(t("today.meta.sample"))
        elif models.get("dream"):
            parts.append(t("today.meta.models", dream=models["dream"], critique=models.get("critique") or "—"))
        flow = FlowLayout(spacing=18)  # the parts wrap rather than push the panel wider than the window
        for part in parts:  # a part wider than the room ends in … and says it all on hover
            flow.addWidget(ElidedLabel(part, "caption", "muted", longest=10_000))
        facts = QWidget()
        facts.setLayout(flow)
        meta.addWidget(facts, 1, Qt.AlignmentFlag.AlignVCenter)
        if is_today and not job:
            meta.addWidget(button(t("today.redream"), "", lambda: self.window.dream(self.day)), 0,
                           Qt.AlignmentFlag.AlignTop)
        b.addLayout(meta)
        failure = self.window.failures.get(dream["day"])
        if failure:  # a re-dive that failed: the dream stays, and the reason is readable in full
            b.addSpacing(12)
            b.addWidget(label(failure, "small", "error", selectable=True))
        payload = dream.get("payload") or {}
        pebbles = shore_threads(payload.get("topics") or [], payload.get("signals") or [])
        if pebbles:
            shore = Pebbles(pebbles, dream["day"], height=136)
            panel.body.setContentsMargins(52, 46, 52, 22)
            shore.thread_clicked.connect(lambda tid: self.window.go("thread", id=tid))
            panel.set_shore(shore)
            panel.set_fish(len(dream.get("sparks") or []))
        return panel

    def _earlier_dive(self, dive: dict) -> QWidget:
        """An earlier dive of the day: its number, time and headline, and the rest folded away."""
        box = vbox(spacing=8, margins=(0, 6, 0, 6))
        when = datetime.fromisoformat(dive["created_at"]).astimezone().strftime("%H:%M")
        box.addWidget(eyebrow(t("today.earlierAt", n=dive["number"], time=when), wrap=True))
        box.addWidget(label(dive.get("title") or "", "display-s"))
        inner = vbox(spacing=14)
        if dive.get("reflection"):
            reflection = para(dive["reflection"], "reading", "ink2", 160)
            reflection.setMaximumWidth(660)
            inner.addWidget(Fold(reflection, lines=6, key=f"reflection:{dive['id']}"))
        if dive.get("undercurrent"):
            inner.addLayout(capped(label(dive["undercurrent"], "quote-s", "ink2"), 760))
        if dive.get("sparks"):
            inner.addSpacing(6)
            inner.addWidget(self.cards(dive["sparks"], compact=True))
        box.addWidget(Unfold(wrap(inner), t("fold.dive"), t("fold.diveLess"), key=f"dive:{dive['id']}"))
        box.addWidget(_hairline())
        return wrap(box)

    def _steps_panel(self, job: dict, standalone: bool = False) -> QWidget:
        panel = NightPanel(padding=(48, 38, 48, 34))
        if standalone:
            panel.body.addWidget(eyebrow(t("sensor.dreaming"), "muted"))
            panel.body.addSpacing(18)
            panel.body.addWidget(label(t("today.dreamingTitle"), "display-m"))
            panel.body.addSpacing(22)
        self.steps = Steps(DREAM_STEPS, job.get("step"))
        panel.body.addWidget(self.steps)
        return panel

    def _waiting(self, state: dict, view: dict, day: str) -> QWidget:
        panel = NightPanel()
        b = panel.body
        b.addWidget(eyebrow(t("today.notYet"), "muted"))
        b.addSpacing(20)
        b.addWidget(label(t("today.notYetTitle", time=state.get("dream_time", "21:30")), "display-m"))
        b.addSpacing(14)
        thin = len(view["subjects"]) < 3
        when = state.get("dream_time", "21:30")
        if thin:
            text = t("today.thin")
        elif view["seconds"] < 60:  # only jots and notes so far
            text = t("today.notYetNotes", n=len(view["subjects"]), time2=when)
        else:
            text = t("today.notYetBody", time=human(view["seconds"]), n=len(view["subjects"]), time2=when)
        body = para(text, "reading", "body", 155)
        body.setMaximumWidth(620)
        b.addWidget(body)
        failure = self.window.failures.get(day)
        if failure:
            b.addSpacing(14)
            b.addWidget(label(failure, "small", "error", selectable=True))
        for trouble in state.get("crew") or []:  # why the crew is not sailing, in plain words
            message = trouble_message(trouble["kind"], trouble["who"], trouble["detail"])
            if message != failure:
                b.addSpacing(10)
                b.addWidget(label(message, "small", "ink2", selectable=True))
        if not state.get("models_ready"):
            b.addSpacing(14)
            b.addWidget(label(t("error.noModel"), "small", "error"))
        b.addSpacing(24)
        go = button(t("today.dreamNow"), "accent", lambda: self.window.dream(day))
        go.setEnabled(bool(view["subjects"]))
        b.addLayout(hbox(go, "stretch"))
        return panel

    def _onboarding(self, state: dict) -> QWidget:
        panel = NightPanel()
        b = panel.body
        b.addWidget(label("Digital Unconscious", "caption-l", "muted", wrap=False))
        b.addSpacing(20)
        b.addWidget(label(t("onboard.title"), "display-m"))
        b.addSpacing(12)
        intro = para(t("onboard.body"), "reading", "body", 155)
        intro.setMaximumWidth(620)
        b.addWidget(intro)
        b.addSpacing(26)
        grid = QGridLayout()
        grid.setSpacing(12)
        sensor_ok = state.get("sensor", {}).get("state") in {"observing", "idle", "quiet", "paused"}
        items = [
            (t("onboard.observe"), t("onboard.observeBody"), sensor_ok),
            (t("onboard.you"), t("onboard.youBody"), state.get("personalised")),
            (t("onboard.feed"), t("onboard.feedBody"), state.get("today_stats", {}).get("traces", 0) > 0),
            (t("onboard.model"), t("onboard.modelBody"), state.get("models_ready")),
        ]
        for index, (title, text, ok) in enumerate(items):
            tile = QFrame()
            tile.setStyleSheet("QFrame { border: 1px solid rgba(168,162,148,0.28); border-radius: 6px; background: rgba(32,38,56,0.6); }"
                               " QLabel { border: none; background: transparent; }")
            inner = vbox(spacing=5, margins=(16, 14, 16, 14))
            inner.addWidget(label(f"{index + 1:02d}", "mono", "accent"))
            inner.addWidget(label(title, "card-title-s"))
            inner.addWidget(label(text, "small", "muted"))
            inner.addStretch(1)
            inner.addWidget(label(t("onboard.ok") if ok else t("onboard.missing"), "mono-s", "body" if ok else "accent"))
            tile.setLayout(inner)
            grid.addWidget(tile, index // 2, index % 2)
        b.addLayout(grid)
        b.addSpacing(22)
        b.addLayout(hbox(button(t("jot.button"), "accent", self.window.open_jot),
                         button(t("nav.settings"), "", lambda: self.window.go("settings")), "stretch"))
        return panel

    def _last_dream(self, dream: dict) -> None:
        self.add(SectionHead(t("today.lastDream"), note=long_day(dream["day"])), 52)
        self.body.addSpacing(16)
        title = label(dream["title"], "display-m")
        title.setCursor(Qt.CursorShape.PointingHandCursor)
        title.mousePressEvent = lambda _e: self.window.go("today", day=dream["day"])
        self.add(title)
        if dream.get("undercurrent"):
            self.add(label(dream["undercurrent"], "quote-s", "ink2"), 8)
        if dream.get("sparks"):
            self.body.addSpacing(22)
            self.add(self.cards(dream["sparks"], compact=True))

    # -- residue -------------------------------------------------------------

    def _residue(self, view: dict, signals: dict, is_today: bool) -> None:
        note = (t("today.residueNote", time=human(view["seconds"]), n=len(view["subjects"])) if view["seconds"] >= 60
                else t("today.residueNotes", n=len(view["subjects"])))
        self.add(SectionHead(t("today.residue"), note=note), 56)
        self.body.addSpacing(14)
        if view["segments"] or view["marks"]:
            self.add(Ribbon(view))
        else:
            self.add(label(t("today.nothingYet"), "quote-s", "muted"))
        self.body.addSpacing(24)

        left = vbox(spacing=12)
        left.addWidget(eyebrow(t("today.threadsToday")))
        left.addSpacing(4)
        totals: dict[int, dict] = {}
        for s in view["subjects"]:
            if s["thread_id"] is None:
                continue
            entry = totals.setdefault(s["thread_id"], {"id": s["thread_id"], "name": s["thread"], "hue": s["hue"], "seconds": 0})
            entry["seconds"] += s["seconds"]
        rows = sorted(totals.values(), key=lambda r: -r["seconds"])
        peak = max([1] + [r["seconds"] for r in rows])
        if not rows:
            left.addWidget(label(t("today.unsorted"), "question", "muted"))
        for row in rows:
            line = hbox(spacing=8)
            name = TitleLabel(row["name"], "body")  # wraps, where a button would push the column wider
            dot = (name.fontMetrics().height() - 10) // 2  # on the first line, however many the name takes
            line.addLayout(vbox(Swatch(row["hue"]), "stretch", spacing=0, margins=(0, max(0, dot), 0, 0)))
            name.clicked.connect(lambda tid=row["id"]: self.window.go("thread", id=tid))
            line.addWidget(name, 1)
            for kind in [k for k in signals.get(row["id"], []) if k != "steady"][:2]:
                line.addWidget(Stamp(kind, small=True), 0, Qt.AlignmentFlag.AlignTop)
            line.addWidget(label(human(row["seconds"]), "mono-n", "muted", wrap=False), 0, Qt.AlignmentFlag.AlignTop)
            left.addLayout(vbox(line, MiniBar(row["seconds"] / peak, row["hue"]), spacing=5))
        left.addStretch(1)

        right = vbox(spacing=0)
        right.addWidget(eyebrow(t("today.subjects")))
        right.addSpacing(10)
        if not view["subjects"]:
            right.addWidget(label(t("today.nothingYet"), "question", "muted"))
        rows = []
        for s in view["subjects"]:
            line = hbox(spacing=10, margins=(0, 7, 0, 7))
            mark = label(KIND_MARK.get(s["kind"], ""), "mono-n", "accent" if s["kind"] == "jot" else "muted", wrap=False)
            mark.setFixedWidth(14)
            if not KIND_MARK.get(s["kind"]):
                mark = Swatch(s["hue"], 7)
                holder = QWidget()
                holder.setFixedWidth(14)
                inner = hbox(mark, spacing=0)
                inner.setAlignment(Qt.AlignmentFlag.AlignCenter)
                holder.setLayout(inner)
                mark = holder
            line.addWidget(mark)
            text = f"“{s['body'] or s['label']}”" if s["kind"] == "jot" else s["label"]
            whole = s["body"] if s["kind"] == "jot" and s.get("body") else s["label"]
            tip = "\n".join(x for x in (whole, s.get("thread") or "", s.get("domain") or "") if x)
            line.addWidget(ElidedLabel(text, "body", tip=tip), 1)  # ends in … at whatever width it gets
            if s["seconds"]:
                line.addWidget(label(human(s["seconds"]), "mono-n", "muted", wrap=False))
            rows.append(wrap(vbox(wrap(line), _hairline(), spacing=0)))
        if rows:
            right.addWidget(FoldList(rows, keep=14, key=f"subjects:{view['day']}"))
        right.addStretch(1)

        columns = hbox(spacing=44)
        columns.addLayout(left, 6)
        columns.addLayout(right, 5)
        self.add(columns)

        if is_today:
            self.body.addSpacing(26)
            box = QLineEdit()
            box.setPlaceholderText(t("jot.inline"))
            box.setFont(font("serif"))
            box.setMaxLength(2000)
            keep = button(t("jot.save"), "line")

            def submit() -> None:
                if box.text().strip():
                    self.window.save_jot(box.text())
                    box.clear()

            box.returnPressed.connect(submit)
            keep.clicked.connect(submit)
            self.add(hbox(box, keep))

    def on_state(self, state: dict) -> None:
        job = next((j for j in state.get("jobs", []) if j["kind"] == "dream" and j["ref"] == getattr(self, "day", None)), None)
        if job and self.steps is not None:
            self.steps.set_current(job.get("step"))
        elif job and self.steps is None:
            self.refresh()


def _hairline() -> QWidget:
    from unconscious.ui.widgets import Rule

    return Rule("rule2")
