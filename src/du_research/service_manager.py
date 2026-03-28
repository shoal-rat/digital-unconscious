from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from du_research.config import AppConfig
from du_research.utils import iso_now


class ServiceManager:
    def __init__(self, *, config: AppConfig, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.workspace_dir = Path(config.pipeline.workspace_dir).resolve()
        self.pid_path = self._workspace_path(config.service.pid_path)
        self.log_path = self._workspace_path(config.service.log_path)
        self.status_path = self._workspace_path(config.service.status_path)
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.parent.mkdir(parents=True, exist_ok=True)

    def start_background(
        self,
        *,
        config_path: Path,
        log_file: str | None = None,
        interval_minutes: int | None = None,
    ) -> dict[str, Any]:
        status = self.status()
        if status["running"]:
            return status

        cmd = [sys.executable, "-m", "du_research.cli", "--config", str(config_path), "start"]
        if log_file:
            cmd.extend(["--log-file", log_file])
        if interval_minutes is not None:
            cmd.extend(["--interval-minutes", str(interval_minutes)])

        env = dict(os.environ)
        src_path = str(self.project_root / "src")
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with self.log_path.open("ab") as handle:
            popen_kwargs: dict[str, Any] = {
                "cwd": str(self.project_root),
                "stdout": handle,
                "stderr": handle,
                "stdin": subprocess.DEVNULL,
                "env": env,
            }
            if os.name == "nt":
                creationflags = 0
                creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
                creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
                popen_kwargs["creationflags"] = creationflags
            else:
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(cmd, **popen_kwargs)

        payload = {
            "pid": process.pid,
            "started_at": iso_now(),
            "command": cmd,
            "log_path": str(self.log_path),
            "status_path": str(self.status_path),
        }
        self.pid_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status_path.write_text(
            json.dumps({"running": True, "started_at": payload["started_at"], "pid": process.pid}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "running": True,
            "pid": process.pid,
            "pid_path": str(self.pid_path),
            "log_path": str(self.log_path),
            "status_path": str(self.status_path),
        }

    def stop(self) -> dict[str, Any]:
        metadata = self._load_pid_metadata()
        if not metadata:
            return {"running": False, "stopped": False, "reason": "no_pid_file"}
        pid = int(metadata.get("pid", 0) or 0)
        if pid <= 0 or not self._pid_exists(pid):
            self._cleanup_pid_file()
            return {"running": False, "stopped": False, "reason": "process_not_found"}
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
        else:
            os.kill(pid, signal.SIGTERM)
        self._cleanup_pid_file()
        self.status_path.write_text(
            json.dumps({"running": False, "stopped_at": iso_now(), "pid": pid}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {"running": False, "stopped": True, "pid": pid}

    def restart(
        self,
        *,
        config_path: Path,
        log_file: str | None = None,
        interval_minutes: int | None = None,
    ) -> dict[str, Any]:
        self.stop()
        return self.start_background(config_path=config_path, log_file=log_file, interval_minutes=interval_minutes)

    def status(self) -> dict[str, Any]:
        metadata = self._load_pid_metadata()
        pid = int(metadata.get("pid", 0) or 0) if metadata else 0
        running = bool(pid and self._pid_exists(pid))
        latest_status = {}
        if self.status_path.exists():
            try:
                latest_status = json.loads(self.status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                latest_status = {}
        if metadata and not running:
            self._cleanup_pid_file()
        return {
            "running": running,
            "pid": pid if running else None,
            "pid_path": str(self.pid_path),
            "log_path": str(self.log_path),
            "status_path": str(self.status_path),
            "latest_status": latest_status,
        }

    def _load_pid_metadata(self) -> dict[str, Any] | None:
        if not self.pid_path.exists():
            return None
        try:
            return json.loads(self.pid_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _cleanup_pid_file(self) -> None:
        if self.pid_path.exists():
            self.pid_path.unlink()

    def _pid_exists(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _workspace_path(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        parts = list(path.parts)
        if parts and parts[0].lower() == "workspace":
            parts = parts[1:]
        return self.workspace_dir.joinpath(*parts)
