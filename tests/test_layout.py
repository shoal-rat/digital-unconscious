"""Nothing on a page is cut off: long texts fit or fold behind a pill, and no page is wider than its window."""

from __future__ import annotations

import json
import os
import sqlite3
import unittest

from helpers import TempApp

from unconscious.demo import seed_demo

SIZES = [(1000, 700), (1320, 900), (1600, 1000), (1920, 1080)]


def _longer(text, times: float, sep: str):
    if not isinstance(text, str) or not text.strip():
        return text
    whole = int(times)
    parts = [text] * whole + ([text[: max(1, int(len(text) * (times - whole)))]] if times > whole else [])
    return sep.join(parts)


def lengthen(db_path, language: str) -> None:
    """Make the demo's texts two or three times longer, as a talkative model would write them."""
    sep = "" if language == "zh" else " "
    db = sqlite3.connect(db_path)
    for i, title, reflection, under in db.execute("select id, title, reflection, undercurrent from dreams").fetchall():
        db.execute("update dreams set title=?, reflection=?, undercurrent=? where id=?",
                   (_longer(title, 2, sep), _longer(reflection, 3, sep), _longer(under, 2.5, sep), i))
    for i, title, insight, evidence in db.execute("select id, title, insight, evidence from sparks").fetchall():
        items = json.loads(evidence or "[]")
        items = (items * 3)[:9]  # more evidence than a card shows
        db.execute("update sparks set title=?, insight=?, evidence=? where id=?",
                   (_longer(title, 2, sep), _longer(insight, 3, sep), json.dumps(items, ensure_ascii=False), i))
    for i, name, gist in db.execute("select id, name, gist from threads").fetchall():
        db.execute("update threads set name=?, gist=? where id=?", (_longer(name, 2, sep), _longer(gist, 2.5, sep), i))
    db.commit()
    db.close()


@unittest.skipUnless(os.environ.get("DUN_UI_TESTS", "1") == "1", "UI tests disabled")
class LayoutTests(TempApp):
    def setUp(self) -> None:
        super().setUp()
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest("PySide6 not installed")
        self.qt = QApplication.instance() or QApplication([])
        from unconscious.ui import theme

        theme.load_fonts()
        theme.apply(self.qt, False)

    def settle(self) -> None:
        from PySide6.QtCore import QCoreApplication, QEvent

        for _ in range(6):
            QCoreApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)
            self.qt.processEvents()

    def window(self, language: str):
        from unconscious.app import App
        from unconscious.jobs import Jobs
        from unconscious.store import today
        from unconscious.ui.i18n import set_language
        from unconscious.ui.widgets import UNFOLDED
        from unconscious.ui.window import MainWindow

        seed_demo(self.home, language, reset=False)
        lengthen(self.home / "memory.db", language)
        ctx = App(self.home)
        ctx._router = self.app._router
        first = ctx.store.dream(today())  # dive the day a second time: the first folds away on its shore, opened
        fish = [{k: s[k] for k in ("title", "mechanism", "question", "insight", "evidence", "score")}
                for s in ctx.store.sparks(dream_id=first["id"])[:2]]
        ctx.store.save_dream(today(), title=first["title"] + " (again)", reflection=first["reflection"],
                             undercurrent=first["undercurrent"], payload=first["payload"], models=first["models"], sparks=fish)
        UNFOLDED.add(f"dive:{first['id']}")
        self.addCleanup(UNFOLDED.discard, f"dive:{first['id']}")
        set_language(language)
        window = MainWindow(ctx, Jobs(ctx), demo=True)
        self.addCleanup(window.close)
        self.addCleanup(set_language, "en")
        return ctx, window

    def cut_widgets(self, page) -> list[str]:
        from PySide6.QtWidgets import QLabel, QPushButton, QStyle, QStyleOptionButton, QWidget

        container = page.widget()
        problems = []
        if container.width() > page.viewport().width():
            problems.append(f"page {container.width()}px wide in a {page.viewport().width()}px window")
        for widget in [container, *container.findChildren(QWidget)]:
            if not widget.isVisible():
                continue
            name = type(widget).__name__
            layout = widget.layout()
            wrapped = isinstance(widget, QLabel) and widget.wordWrap()
            if wrapped or widget.hasHeightForWidth() or (layout is not None and layout.hasHeightForWidth()):
                needed = widget.heightForWidth(widget.width())
                if layout is not None and layout.hasHeightForWidth():
                    needed = max(needed, layout.totalHeightForWidth(widget.width()))
                if widget.height() + 1 < needed:
                    problems.append(f"{name} cut at the bottom: {widget.height()} of {needed}px")
            if isinstance(widget, QPushButton) and widget.text():
                option = QStyleOptionButton()
                widget.initStyleOption(option)
                contents = widget.style().subElementRect(QStyle.SubElement.SE_PushButtonContents, option, widget)
                icon = widget.iconSize().width() + 4 if not widget.icon().isNull() else 0
                need = widget.fontMetrics().horizontalAdvance(widget.text().replace("&", "")) + icon + widget.width() - contents.width()
                if widget.width() + 1 < need:
                    problems.append(f"button cut sideways: {widget.width()} of {need}px")
            texts = [c for c in widget.children() if isinstance(c, (QLabel, QPushButton)) and c.isVisible()]
            for i, a in enumerate(texts):
                for b in texts[i + 1:]:
                    shared = a.geometry().intersected(b.geometry())
                    if shared.width() > 1 and shared.height() > 1:
                        problems.append(f"{type(a).__name__} and {type(b).__name__} overlap in {name}")
        return problems

    def check_every_page(self, language: str) -> None:
        ctx, window = self.window(language)
        thread_id = ctx.store.threads()[0]["id"]
        spark_id = ctx.store.sparks()[0]["id"]
        routes = [("today", {}), ("threads", {}), ("thread", {"id": thread_id}), ("sparks", {"status": ""}),
                  ("spark", {"id": spark_id}), ("journal", {}), ("settings", {})]
        window.show()
        for width, height in SIZES:
            window.resize(width, height)
            self.settle()
            for name, params in routes:
                window.go(name, **params)
                self.settle()
                with self.subTest(language=language, size=f"{width}x{height}", page=name):
                    self.assertEqual(self.cut_widgets(window.page), [])

    def test_nothing_is_cut_off_in_english(self):
        self.check_every_page("en")

    def test_nothing_is_cut_off_in_chinese(self):
        self.check_every_page("zh")

    def test_a_fish_page_leads_back_to_the_dive_it_came_from(self):
        from unconscious.store import today
        from unconscious.ui.widgets import UNFOLDED, TitleLabel

        ctx, window = self.window("en")
        earlier = ctx.store.day_dives(today())[1]
        fish = ctx.store.sparks(dream_id=earlier["id"])[0]["id"]
        UNFOLDED.discard(f"dive:{earlier['id']}")
        window.show()
        window.go("spark", id=fish)
        self.settle()
        link = next(w for w in window.page.widget().findChildren(TitleLabel) if w.text() == earlier["title"])
        link.clicked.emit()
        self.settle()
        self.assertEqual((window.route[0], window.page.day), ("today", today()))
        self.assertIn(f"dive:{earlier['id']}", UNFOLDED, "the earlier dive is opened")

    def test_capped_text_is_measured_at_its_own_width(self):
        """The bug behind a dream cut off halfway: a box measured a capped title at the full column
        width, came out short, and squeezed the difference out of the reflection below it."""
        from PySide6.QtWidgets import QWidget

        from unconscious.ui.widgets import capped, label, para, vbox, wrap

        host = QWidget()
        self.addCleanup(host.close)
        title = label("A title that wraps onto more lines once it is held narrow " * 3, "display-xl")
        title.setMaximumWidth(320)
        quote = wrap(vbox(label("An undercurrent said in a long quiet sentence that wraps " * 3, "quote")))
        body = para("The reflection under the title, long enough to be squeezed. " * 12)
        box = vbox(title, spacing=12)
        box.addLayout(capped(quote, 320))
        box.addWidget(body)
        host.setLayout(box)
        host.show()
        host.resize(900, box.totalHeightForWidth(900))
        self.settle()
        for widget in (title, quote, body):
            with self.subTest(widget=type(widget).__name__):
                self.assertLessEqual(widget.width(), 900)
                self.assertGreaterEqual(widget.height() + 1, widget.heightForWidth(widget.width())
                                        if widget.layout() is None else widget.layout().totalHeightForWidth(widget.width()))

    def test_long_text_folds_behind_a_pill_and_unfolds(self):
        from PySide6.QtWidgets import QWidget

        from unconscious.ui.widgets import UNFOLDED, Fold, para, vbox

        UNFOLDED.discard("test:fold")
        host = QWidget()
        self.addCleanup(host.close)
        long_text = para("The tide came in and went out again. " * 40)
        short_text = para("A short line.")
        long_fold, short_fold = Fold(long_text, lines=4, key="test:fold"), Fold(short_text, lines=4)
        host.setLayout(vbox(long_fold, short_fold, "stretch"))
        host.resize(520, 2000)
        host.show()
        self.settle()
        self.assertTrue(long_fold.toggle.isVisible(), "a long text shows the pill")
        self.assertFalse(short_fold.toggle.isVisible(), "a short text has nothing to unfold")
        whole = long_text._layout(long_text.width())[1]
        folded = long_text.height()
        self.assertLess(folded, whole / 2)
        long_fold.toggle.click()
        self.settle()
        self.assertGreaterEqual(long_text.height() + 1, whole, "unfolded, every line has room")
        self.assertIn("test:fold", UNFOLDED)
        again = Fold(para("The tide came in and went out again. " * 40), lines=4, key="test:fold")
        self.addCleanup(again.deleteLater)
        self.assertFalse(again.text.folded, "a fold a person opened stays open when the page is rebuilt")
        long_fold.toggle.click()
        self.settle()
        self.assertEqual(long_text.height(), folded)
        self.assertNotIn("test:fold", UNFOLDED)

    def test_the_pill_follows_the_width_the_text_really_gets(self):
        from PySide6.QtWidgets import QWidget

        from unconscious.ui.widgets import Fold, para, vbox

        text = para("word " * 120)
        fold = Fold(text, lines=2)  # built before it has a place: Qt's default width is 640
        self.assertTrue(text.exceeds(640))
        host = QWidget()
        self.addCleanup(host.close)
        host.setLayout(vbox(fold, "stretch"))
        host.resize(2400, 600)
        host.show()
        self.settle()
        self.assertFalse(text.exceeds(text.width()))
        self.assertFalse(fold.toggle.isVisible(), "no pill under text that is shown whole")
        host.resize(500, 600)
        self.settle()
        self.assertTrue(fold.toggle.isVisible(), "narrowed until it folds, the pill comes back")

    def test_a_folded_list_never_opens_a_window_of_its_own(self):
        from PySide6.QtCore import QEvent, QObject
        from PySide6.QtWidgets import QPushButton, QWidget

        from unconscious.ui.widgets import FoldList, label, vbox

        windows = []

        class Watch(QObject):
            def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
                if event.type() == QEvent.Type.Show and isinstance(obj, QPushButton) and obj.isWindow():
                    windows.append(obj)
                return False

        watch = Watch()
        self.qt.installEventFilter(watch)
        self.addCleanup(self.qt.removeEventFilter, watch)
        host = QWidget()
        self.addCleanup(host.close)
        host.setLayout(vbox())
        host.show()
        self.settle()
        rows = FoldList([label(f"row {i}") for i in range(20)], keep=5)
        host.layout().addWidget(rows)
        self.settle()
        self.assertEqual(windows, [], "the pill must not flash up as a window and take focus from the app")
        self.assertTrue(rows.toggle.isVisible())


if __name__ == "__main__":
    unittest.main()
