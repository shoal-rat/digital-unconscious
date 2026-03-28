from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from du_research.utils import iso_now


class SubmissionManager:
    def __init__(self, approvals_dir: Path, pending_timeout_hours: int = 72):
        self.approvals_dir = approvals_dir
        self.pending_timeout_hours = pending_timeout_hours
        self.approvals_dir.mkdir(parents=True, exist_ok=True)

    def create_submission_package(
        self,
        run_dir: Path,
        *,
        final_manuscript_path: Path | None = None,
        final_markdown_path: Path | None = None,
    ) -> Path:
        history = [
            {
                "timestamp": iso_now(),
                "status": "pending_approval",
                "notes": "Submission package created.",
            }
        ]
        package = {
            "created_at": iso_now(),
            "run_dir": str(run_dir),
            "status": "pending_approval",
            "pending_timeout_hours": self.pending_timeout_hours,
            "history": history,
            "artifacts": {
                "manuscript_pdf": str(final_manuscript_path or (run_dir / "06_review" / "final_manuscript.pdf")),
                "manuscript_md": str(final_markdown_path or (run_dir / "06_review" / "final_manuscript.md")),
                "review_report": str(run_dir / "06_review" / "review_report.json"),
                "review_history": str(run_dir / "06_review" / "review_history.jsonl"),
            },
        }
        path = self.approvals_dir / f"{run_dir.name}_submission.json"
        path.write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def update_status(self, run_id: str, status: str, notes: str = "") -> Path:
        path = self.approvals_dir / f"{run_id}_submission.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = status
        payload["updated_at"] = iso_now()
        payload["notes"] = notes
        payload.setdefault("history", []).append(
            {
                "timestamp": payload["updated_at"],
                "status": status,
                "notes": notes,
            }
        )
        if status == "approved":
            payload["approved_at"] = payload["updated_at"]
        if status == "submitted":
            payload["submitted_at"] = payload["updated_at"]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
