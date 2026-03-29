"""Offline task queue — saves pending AI work when LLM is unavailable.

When the AI backend hits rate limits, quota exhaustion, or network issues,
tasks are saved to a local JSONL queue file. On the next run (or when
manually triggered), the queue is drained by re-submitting to the LLM.

This replaces all heuristic fallbacks. The system is AI-only: if the LLM
can't be reached right now, the work waits until it can.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from du_research.utils import iso_now

logger = logging.getLogger(__name__)


class TaskQueue:
    """Persistent JSONL queue for pending AI tasks."""

    def __init__(self, workspace_dir: Path):
        self.queue_path = workspace_dir / "queue" / "pending_tasks.jsonl"
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        priority: int = 0,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Add a task to the queue. Returns the task id."""
        task_id = f"{task_type}_{iso_now().replace(':', '').replace('-', '')}"
        entry = {
            "id": task_id,
            "type": task_type,
            "status": "pending",
            "priority": priority,
            "created_at": iso_now(),
            "payload": payload,
            "context": context or {},
            "attempts": 0,
            "last_error": None,
        }
        with self.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Queued task %s (type=%s)", task_id, task_type)
        return task_id

    def pending_tasks(self) -> list[dict[str, Any]]:
        """Load all pending tasks, sorted by priority (highest first)."""
        if not self.queue_path.exists():
            return []
        tasks = []
        for line in self.queue_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                task = json.loads(line)
                if task.get("status") == "pending":
                    tasks.append(task)
            except json.JSONDecodeError:
                continue
        return sorted(tasks, key=lambda t: t.get("priority", 0), reverse=True)

    def drain(
        self,
        handler: Callable[[dict[str, Any]], dict[str, Any] | None],
        max_tasks: int = 50,
    ) -> dict[str, Any]:
        """Process pending tasks with the given handler.

        The handler receives a task dict and returns a result dict (success)
        or None (failure, will be re-queued).
        """
        tasks = self.pending_tasks()
        if not tasks:
            return {"processed": 0, "succeeded": 0, "failed": 0, "remaining": 0}

        succeeded = 0
        failed = 0
        still_pending: list[dict[str, Any]] = []
        completed_ids: set[str] = set()

        for task in tasks[:max_tasks]:
            try:
                result = handler(task)
                if result is not None:
                    completed_ids.add(task["id"])
                    succeeded += 1
                else:
                    task["attempts"] = task.get("attempts", 0) + 1
                    task["last_error"] = "handler returned None"
                    still_pending.append(task)
                    failed += 1
            except Exception as exc:
                task["attempts"] = task.get("attempts", 0) + 1
                task["last_error"] = str(exc)
                still_pending.append(task)
                failed += 1

        # Rewrite queue with remaining tasks
        all_lines = []
        if self.queue_path.exists():
            for line in self.queue_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    t = json.loads(line)
                    if t.get("id") in completed_ids:
                        t["status"] = "completed"
                    all_lines.append(json.dumps(t, ensure_ascii=False))
                except json.JSONDecodeError:
                    all_lines.append(line)

        self.queue_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")

        remaining = len(self.pending_tasks())
        return {
            "processed": succeeded + failed,
            "succeeded": succeeded,
            "failed": failed,
            "remaining": remaining,
        }

    def count_pending(self) -> int:
        return len(self.pending_tasks())
