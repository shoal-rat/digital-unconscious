"""The scheduler, the view models over demo memory, the CLI, and an offscreen UI smoke test."""

import io
import os
import unittest
from contextlib import redirect_stdout
from datetime import datetime

from helpers import TempApp

from unconscious import api
from unconscious.demo import seed_demo
from unconscious.jobs import due_dreams, record_attempt
from unconscious.store import today


class SchedulerTests(TempApp):
    def test_due_after_dream_time_and_attempted_at_most_twice(self):
        self.trace("2026-09-30", "deep work", seconds=3600)
        self.app.update_settings({"dream": {"time": "21:30", "auto": True}})
        before = datetime(2026, 9, 30, 20, 0).astimezone()
        after = datetime(2026, 9, 30, 22, 0).astimezone()
        self.assertEqual(due_dreams(self.app, before), [])
        self.assertEqual(due_dreams(self.app, after), ["2026-09-30"])
        record_attempt(self.app, "2026-09-30")
        record_attempt(self.app, "2026-09-30")
        self.assertEqual(due_dreams(self.app, after), [])

    def test_yesterday_is_caught_up_and_auto_can_be_off(self):
        self.trace("2026-09-29", "deep work", seconds=3600)
        morning = datetime(2026, 9, 30, 8, 0).astimezone()
        self.assertEqual(due_dreams(self.app, morning), ["2026-09-29"])
        self.app.update_settings({"dream": {"auto": False}})
        self.assertEqual(due_dreams(self.app, morning), [])


class DemoViewTests(TempApp):
    def setUp(self):
        super().setUp()
        seed_demo(self.home, "en", reset=False)
        self.app._settings = None
        from unconscious.app import App

        fresh = App(self.home)
        fresh._router = self.app._router
        self.app = fresh

    def test_views_have_what_the_interface_needs(self):
        state = api.state(self.app)
        self.assertTrue(state["has_memory"])
        self.assertTrue(state["dreamt_today"])
        day = api.day_view(self.app, today())
        self.assertTrue(day["segments"] and day["subjects"])
        dream = api.dream_view(self.app, today())
        self.assertEqual(len(dream["sparks"]), 3)
        threads = api.threads_view(self.app)
        kinds = {s["kind"] for s in threads["signals"]}
        self.assertTrue({"orbit", "return", "surge", "seed"} <= kinds)
        bike = next(t for t in threads["threads"] if t["name"].startswith("Bike"))
        self.assertIn("orbit", api.thread_view(self.app, bike["id"])["signals"])
        pursuing = self.app.store.sparks(status="pursuing")[0]
        self.assertTrue(api.spark_view(self.app, pursuing["id"])["dives"])
        self.assertIn("routes", api.settings_view(self.app)["models"])


class CliTests(TempApp):
    def run_cli(self, *args):
        from unconscious.cli import main

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(["--home", str(self.home), *args])
        return code, buffer.getvalue()

    def test_jot_config_and_sparks(self):
        code, _ = self.run_cli("jot", "a", "half-formed", "thought")
        self.assertEqual(code, 0)
        self.assertEqual(self.app.store.traces(today())[0]["kind"], "jot")
        code, _ = self.run_cli("config", "set", "dream.time", "22:00")
        self.assertEqual(code, 0)
        self.assertEqual(self.app.settings.dream.time, "22:00")
        code, out = self.run_cli("sparks")
        self.assertEqual(code, 0)


@unittest.skipUnless(os.environ.get("DUN_UI_TESTS", "1") == "1", "UI tests disabled")
class InterfaceSmokeTests(TempApp):
    def test_every_page_builds_and_actions_work(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest("PySide6 not installed")
        seed_demo(self.home, "en", reset=False)
        qt = QApplication.instance() or QApplication([])
        from unconscious.app import App
        from unconscious.jobs import Jobs
        from unconscious.ui import theme
        from unconscious.ui.window import MainWindow

        theme.load_fonts()
        theme.apply(qt, False)
        ctx = App(self.home)
        ctx._router = self.app._router
        window = MainWindow(ctx, Jobs(ctx), demo=True)
        window.resize(1280, 900)
        thread_id = ctx.store.threads()[0]["id"]
        spark_id = ctx.store.sparks()[0]["id"]
        for name, params in [("today", {}), ("threads", {}), ("thread", {"id": thread_id}), ("sparks", {"status": "new"}),
                             ("spark", {"id": spark_id}), ("journal", {}), ("settings", {}), ("today", {"day": "2026-01-01"})]:
            window.go(name, **params)
            qt.processEvents()
            self.assertIsNotNone(window.page)
            self.assertGreater(window.page.body.count(), 1)
        window.poll()
        self.assertIn("threads", window.state["counts"], "a poll must not drop the full counts")
        self.assertIn("traces", window.state["today_stats"])
        window.spark_feedback(spark_id, "kept", "")
        self.assertEqual(ctx.store.spark(spark_id)["status"], "kept")
        window.save_jot("a thought from the window")
        ctx.update_settings({"you": {"language": "zh"}})
        window.apply_preferences()
        self.assertEqual(window.sidebar.buttons["today"].text(), "海岸")
        window.back()

        from unconscious.ui.motion import CALM_MS, SLOW_MS, ticker

        clock = ticker()
        self.assertEqual(window.timer.interval(), 30000, "never shown: poll slowly")
        window.show()
        qt.processEvents()
        self.assertEqual(window.timer.interval(), 5000)
        moving = window.isVisible() and qt.applicationState().name == "ApplicationActive"
        clock.set_mode("full")
        self.assertEqual(clock._timer.isActive(), bool(moving and clock._active()))
        if clock._timer.isActive():
            self.assertLessEqual(clock._timer.interval(), SLOW_MS)
            clock.set_mode("calm")
            self.assertEqual(clock._timer.interval(), CALM_MS)
        clock.set_mode("off")
        self.assertFalse(clock._timer.isActive())
        clock.set_mode("full")
        window.hide()
        qt.processEvents()
        clock.refresh()
        self.assertFalse(clock._timer.isActive(), "no motion while the window is away")
        self.assertEqual(window.timer.interval(), 30000)
        window.close()


if __name__ == "__main__":
    unittest.main()
