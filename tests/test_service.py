"""Setting out at login: the login item, with a stand-in launchctl, and the Harbour's switch."""

from __future__ import annotations

import json
import os
import platform
import stat
import unittest
from pathlib import Path

from helpers import TempApp

from unconscious import service

LAUNCHCTL = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["FAKE_LAUNCHCTL_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1:2] == ["bootstrap"] and os.environ.get("FAKE_LAUNCHCTL_REFUSE"):
    sys.stderr.write("Bootstrap failed: 5: Input/output error\\n")
    sys.exit(5)
"""


@unittest.skipUnless(platform.system() == "Darwin", "the login item is a launchd agent on macOS")
class LoginItemTests(TempApp):
    def setUp(self) -> None:
        super().setUp()
        self.saved = {k: os.environ.get(k) for k in ("HOME", "PATH", "XPC_SERVICE_NAME", "FAKE_LAUNCHCTL_LOG", "FAKE_LAUNCHCTL_REFUSE")}
        bin_dir = self.home / "bin"
        bin_dir.mkdir()
        tool = bin_dir / "launchctl"
        tool.write_text(LAUNCHCTL)
        tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
        self.log = self.home / "launchctl.log"
        os.environ.update({"HOME": str(self.home), "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                           "FAKE_LAUNCHCTL_LOG": str(self.log)})
        for key in ("XPC_SERVICE_NAME", "FAKE_LAUNCHCTL_REFUSE"):
            os.environ.pop(key, None)
        self.plist = Path(self.home) / "Library" / "LaunchAgents" / f"{service.LABEL}.plist"

    def tearDown(self) -> None:
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        super().tearDown()

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_setting_out_at_login_and_staying_ashore(self):
        self.assertFalse(service.installed())
        service.set_at_login(True)
        self.assertTrue(service.installed())
        text = self.plist.read_text()
        self.assertIn("<string>app</string>", text)
        self.assertIn("<string>--hidden</string>", text)
        self.assertEqual([c[0] for c in self.calls()], ["bootout", "bootstrap"])
        service.set_at_login(False)
        self.assertFalse(service.installed())
        self.assertEqual(self.calls()[-1][0], "bootout")

    def test_the_app_started_by_the_login_item_is_never_stopped_by_its_own_switch(self):
        os.environ["XPC_SERVICE_NAME"] = service.LABEL
        self.assertTrue(service.running_as_job())
        service.set_at_login(True)
        service.set_at_login(False)
        self.assertFalse(service.installed())
        self.assertEqual(self.calls(), [], "no bootout of the job that is this very app")

    def test_a_refusal_is_said_in_the_systems_own_words(self):
        os.environ["FAKE_LAUNCHCTL_REFUSE"] = "1"
        with self.assertRaises(RuntimeError) as caught:
            service.set_at_login(True)
        self.assertIn("Input/output error", str(caught.exception))
        self.assertFalse(service.installed(), "a refused item leaves nothing behind for the next login")

    def test_the_harbour_switch_works_off_the_interface_thread(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtWidgets import QApplication, QCheckBox
        except ImportError:
            self.skipTest("PySide6 not installed")
        import time

        from unconscious.app import App
        from unconscious.jobs import Jobs
        from unconscious.ui import theme
        from unconscious.ui.i18n import t
        from unconscious.ui.window import MainWindow

        qt = QApplication.instance() or QApplication([])
        theme.apply(qt, False)
        ctx = App(self.home)
        ctx._router = self.app._router

        def harbour(demo: bool):
            window = MainWindow(ctx, Jobs(ctx), demo=demo)
            self.addCleanup(window.close)
            window.show()
            window.go("settings")
            qt.processEvents()
            return window

        def box(window, key: str) -> QCheckBox:
            return next(b for b in window.page.widget().findChildren(QCheckBox) if b.text() == t(key))

        borrowed = harbour(demo=True)
        self.assertFalse(box(borrowed, "settings.atLogin").isEnabled(), "a borrowed sea never takes over the login item")
        window = harbour(demo=False)
        self.assertFalse(box(window, "settings.atLogin").isChecked())
        box(window, "settings.titles").setChecked(False)  # an edit not saved yet
        switch = box(window, "settings.atLogin")
        switch.setChecked(True)
        self.assertFalse(switch.isEnabled(), "busy while the system answers")
        deadline = time.monotonic() + 5
        while not switch.isEnabled():
            qt.processEvents()
            self.assertLess(time.monotonic(), deadline, "the switch never came back")
        self.assertTrue(service.installed() and switch.isChecked(), "the switch shows what is really set")
        self.assertFalse(box(window, "settings.titles").isChecked(), "the unsaved edit next to it is still there")
        self.assertEqual([c[0] for c in self.calls()], ["bootout", "bootstrap"])


class SecondStartTests(TempApp):
    def test_a_start_at_login_leaves_the_open_app_where_it_is(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PySide6.QtNetwork import QLocalServer
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest("PySide6 not installed")
        import time

        from unconscious.ui import app as ui_app

        qt = QApplication.instance() or QApplication([])
        name = ui_app._instance_name(self.home)
        QLocalServer.removeServer(name)
        server = QLocalServer()
        self.assertTrue(server.listen(name))
        self.addCleanup(server.close)

        class OpenApp:  # the running app's window, as the server sees it
            shown = 0

            def show_window(self) -> None:
                OpenApp.shown += 1

        server.newConnection.connect(lambda: ui_app.answer_second_start(server, OpenApp()))

        def settle() -> None:
            end = time.monotonic() + 0.5
            while time.monotonic() < end:
                qt.processEvents()

        self.assertEqual(ui_app.run(hidden=True), 0)
        settle()
        self.assertEqual(OpenApp.shown, 0, "a start at login leaves the window where it is")
        self.assertEqual(ui_app.run(hidden=False), 0)
        settle()
        self.assertEqual(OpenApp.shown, 1, "a start by hand brings it forward")

if __name__ == "__main__":
    unittest.main()
