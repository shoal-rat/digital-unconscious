"""Shared fixtures: a throwaway app home wired to the offline model."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from unconscious.app import App
from unconscious.llm.fake import fake_router


class TempApp(unittest.TestCase):
    """Each test gets a fresh home directory and an app that talks to FakeLLM."""

    def setUp(self) -> None:
        self.home = Path(tempfile.mkdtemp(prefix="dun-test-"))
        self._old_home = os.environ.get("DUN_HOME")
        os.environ["DUN_HOME"] = str(self.home)
        self.app = App(self.home)
        self.app.update_settings({"you": {"language": "en"}})
        self.app._router = fake_router(self.app.settings, self.app.store)
        self.app._router.pinned = True

    def tearDown(self) -> None:
        if self._old_home is None:
            os.environ.pop("DUN_HOME", None)
        else:
            os.environ["DUN_HOME"] = self._old_home
        shutil.rmtree(self.home, ignore_errors=True)

    def trace(self, day: str, subject: str, *, seconds: float = 600, key: str | None = None, kind: str = "focus",
              at: str = "10:00", category: str = "browser", body: str = "") -> int:
        start = datetime.fromisoformat(f"{day}T{at}:00").astimezone()
        end = start + timedelta(seconds=seconds)
        return self.app.store.add_trace(
            day=day, started_at=start.isoformat(timespec="seconds"), ended_at=end.isoformat(timespec="seconds"),
            seconds=seconds, kind=kind, source="test", app="Browser", category=category, title=subject,
            subject_key=key or f"web:test:{subject.lower()}", subject=subject, body=body,
        )
