from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from du_research.config import AppConfig
from du_research.utils import iso_now


class WorkspaceMaintenance:
    def __init__(self, workspace_dir: Path, config: AppConfig):
        self.workspace_dir = workspace_dir
        self.config = config

    def run(self) -> dict[str, object]:
        removed_observation = self._prune_observation()
        removed_daily = self._prune_daily_cycles()
        removed_browser = self._prune_browser_artifacts()
        trimmed_service_log = self._trim_service_log()
        return {
            "timestamp": iso_now(),
            "removed_observation_files": removed_observation,
            "removed_daily_cycles": removed_daily,
            "removed_browser_artifacts": removed_browser,
            "trimmed_service_log": trimmed_service_log,
        }

    def _prune_observation(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, self.config.retention.observation_days))
        observation_dir = self.workspace_dir / "observation"
        return self._prune_files_older_than(observation_dir, cutoff)

    def _prune_daily_cycles(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, self.config.retention.daily_cycle_days))
        daily_dir = self.workspace_dir / "daily"
        removed = 0
        if not daily_dir.exists():
            return removed
        for path in daily_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("cycle_"):
                continue
            if self._mtime_utc(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    def _prune_browser_artifacts(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, self.config.retention.browser_artifact_days))
        removed = 0
        for raw in [self.config.automation.download_dir, self.config.automation.screenshot_dir]:
            path = self._workspace_path(raw)
            removed += self._prune_files_older_than(path, cutoff)
        return removed

    def _trim_service_log(self) -> bool:
        log_path = self._workspace_path(self.config.service.log_path)
        if not log_path.exists():
            return False
        max_bytes = max(1, self.config.retention.service_log_max_mb) * 1024 * 1024
        size = log_path.stat().st_size
        if size <= max_bytes:
            return False
        keep_bytes = max_bytes // 2
        with log_path.open("rb") as handle:
            handle.seek(max(0, size - keep_bytes))
            data = handle.read()
        newline_index = data.find(b"\n")
        if newline_index >= 0:
            data = data[newline_index + 1 :]
        log_path.write_bytes(data)
        return True

    def _prune_files_older_than(self, directory: Path, cutoff: datetime) -> int:
        removed = 0
        if not directory.exists():
            return removed
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if self._mtime_utc(path) < cutoff:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
        return removed

    def _workspace_path(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        parts = list(path.parts)
        if parts and parts[0].lower() == "workspace":
            parts = parts[1:]
        return self.workspace_dir.joinpath(*parts)

    def _mtime_utc(self, path: Path) -> datetime:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
