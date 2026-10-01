"""Process-wide context: where the home is, current settings, the store, the router."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from unconscious.config import Settings, app_home, load_settings
from unconscious.llm.router import Router
from unconscious.store import Store

log = logging.getLogger(__name__)


class App:
    def __init__(self, home: Path | None = None, *, router: Router | None = None):
        self.home = Path(home) if home else app_home()
        self.home.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.home / "config.toml"
        self._settings = load_settings(self.settings_path)
        if not self.settings_path.exists():
            self._settings.save()
        self._mtime = self._stat()
        self._lock = threading.Lock()
        self.store = Store(self.home / "memory.db")
        if router is not None:
            router.pinned = True  # injected (tests, demo): survives settings reloads
        self._router = router

    def _stat(self) -> float:
        try:
            return self.settings_path.stat().st_mtime
        except OSError:
            return 0.0

    @property
    def settings(self) -> Settings:
        """Settings, reloaded when the file changes (the dashboard or a person edited it)."""
        mtime = self._stat()
        if mtime != self._mtime:
            with self._lock:
                try:
                    self._settings = load_settings(self.settings_path)
                    self._mtime = mtime
                    self._router = None if not getattr(self._router, "pinned", False) else self._router
                except Exception:
                    log.exception("could not reload settings; keeping the previous ones")
        return self._settings

    def update_settings(self, patch: dict) -> list[str]:
        with self._lock:
            settings = self._settings
            changed = settings.update(patch)
            if changed:
                settings.save()
                self._mtime = self._stat()
                if not getattr(self._router, "pinned", False):
                    self._router = None
        return changed

    @property
    def router(self) -> Router:
        if self._router is None:
            if os.environ.get("DUN_FAKE_LLM"):
                from unconscious.llm.fake import fake_router

                self._router = fake_router(self.settings, self.store)
            else:
                self._router = Router(self.settings, self.store)
        return self._router
