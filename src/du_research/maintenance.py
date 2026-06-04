from __future__ import annotations

import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from du_research.config import AppConfig
from du_research.utils import iso_now


def _safe_version(path: Path) -> int:
    try:
        return int(path.name.split(".", 1)[0].lstrip("v"))
    except ValueError:
        return 0


class WorkspaceMaintenance:
    def __init__(self, workspace_dir: Path, config: AppConfig):
        self.workspace_dir = workspace_dir
        self.config = config

    def run(self) -> dict[str, object]:
        removed_observation = self._prune_observation()
        removed_daily = self._prune_daily_cycles()
        removed_browser = self._prune_browser_artifacts()
        trimmed_service_log = self._trim_service_log()
        capped_stores = self._cap_growing_stores()
        pruned_prompt_versions = self._prune_prompt_history()
        return {
            "timestamp": iso_now(),
            "removed_observation_files": removed_observation,
            "removed_daily_cycles": removed_daily,
            "removed_browser_artifacts": removed_browser,
            "trimmed_service_log": trimmed_service_log,
            "capped_stores": capped_stores,
            "pruned_prompt_versions": pruned_prompt_versions,
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

    def _cap_growing_stores(self) -> dict[str, int]:
        """Trim the append-only knowledge/backlog stores to their retention caps.

        A safety net that also reclaims files which grew before per-write caps
        existed (e.g. an already-long-running install).
        """
        retention = self.config.retention
        return {
            "rag_documents": self._cap_jsonl(
                self.workspace_dir / "knowledge" / "rag_documents.jsonl",
                retention.rag_max_documents,
            ),
            "idea_backlog": self._cap_jsonl(
                self.workspace_dir / "ideas" / "idea_backlog.jsonl",
                retention.idea_backlog_max,
            ),
        }

    def _cap_jsonl(self, path: Path, max_lines: int) -> int:
        """Keep only the newest ``max_lines`` non-empty lines; return removed count."""
        if max_lines <= 0 or not path.exists():
            return 0
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) <= max_lines:
            return 0
        removed = len(lines) - max_lines
        path.write_text("\n".join(lines[-max_lines:]) + "\n", encoding="utf-8")
        return removed

    def _prune_prompt_history(self) -> int:
        """Keep newest N prompt versions per agent and rotate evolution logs."""
        retention = self.config.retention
        prompts_dir = self.workspace_dir / "prompts"
        if not prompts_dir.exists():
            return 0
        keep = max(0, retention.prompt_versions_kept)
        removed = 0
        for agent_dir in prompts_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            versions = sorted(agent_dir.glob("v*_proposed.txt"), key=_safe_version)
            stale = versions[:-keep] if keep else versions
            for old in stale:
                try:
                    old.unlink()
                    removed += 1
                except OSError:
                    continue
            self._cap_jsonl(agent_dir / "evolution_log.jsonl", retention.evolution_log_max_lines)
        return removed

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
