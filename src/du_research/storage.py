from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from du_research.models import RunManifest
from du_research.utils import iso_now


class RunStorage:
    def __init__(self, workspace_dir: Path, run_id: str):
        self.workspace_dir = workspace_dir
        self.runs_dir = workspace_dir / "runs"
        self.learning_dir = workspace_dir / "learning"
        self.run_dir = self.runs_dir / run_id
        self.trace_file = self.run_dir / "execution_trace.jsonl"
        self.manifest_file = self.run_dir / "run_manifest.json"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, order: int, name: str) -> Path:
        path = self.run_dir / f"{order:02d}_{name}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def append_trace(
        self,
        stage: str,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "timestamp": iso_now(),
            "stage": stage,
            "event": event,
            "details": details or {},
        }
        with self.trace_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_manifest(self, manifest: RunManifest) -> None:
        self.write_json(self.manifest_file, manifest.to_dict())


def load_manifest(workspace_dir: Path, run_id: str) -> dict[str, Any]:
    manifest_file = workspace_dir / "runs" / run_id / "run_manifest.json"
    return json.loads(manifest_file.read_text(encoding="utf-8"))

