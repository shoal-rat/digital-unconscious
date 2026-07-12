"""Bounded, de-duplicated storage for research ideas.

The backlog is deliberately a JSONL file: it remains inspectable, portable,
and easy to repair.  This module is the single owner of its storage rules so
the CLI, dashboard, tray, daily engine, and paper ideation workflow agree on
identity, ordering, and retention.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from du_research.utils import iso_now


class IdeaBacklog:
    """Small repository around ``ideas/idea_backlog.jsonl``."""

    def __init__(self, workspace: str | Path, max_entries: int = 500):
        self.workspace = Path(workspace).resolve()
        self.path = self.workspace / "ideas" / "idea_backlog.jsonl"
        self.max_entries = max_entries

    @staticmethod
    def title_of(idea: dict[str, Any]) -> str:
        return str(idea.get("title") or idea.get("idea_text") or "").strip()

    @staticmethod
    def id_of(idea: dict[str, Any]) -> str:
        return str(idea.get("id") or idea.get("idea_id") or "").strip()

    @staticmethod
    def key(title: str) -> str:
        normalized = " ".join(title.casefold().split())
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest() if normalized else ""

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        ideas: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                ideas.append(value)
        return ideas

    def recent_titles(self, limit: int = 50) -> list[str]:
        titles = [self.title_of(idea) for idea in self.load()]
        return [title for title in titles if title][-max(0, limit) :]

    def get(self, idea_id: str) -> dict[str, Any] | None:
        for idea in self.load():
            if self.id_of(idea) == idea_id:
                return idea
        return None

    def contains(
        self,
        title: str,
        *,
        idea_id: str | None = None,
        ideas: Iterable[dict[str, Any]] | None = None,
    ) -> bool:
        wanted_key = self.key(title)
        for idea in ideas if ideas is not None else self.load():
            if idea_id and self.id_of(idea) == idea_id:
                return True
            if wanted_key and self.key(self.title_of(idea)) == wanted_key:
                return True
        return False

    def append_unique(
        self,
        ideas: Iterable[dict[str, Any]],
        *,
        date: str | None = None,
        origin: str | None = None,
    ) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized = [self.normalize(raw_idea) for raw_idea in ideas]

        # Lock a stable sidecar rather than the backlog inode: the backlog is
        # atomically replaced below, so locking it directly would not protect
        # processes that opened the replacement file.
        with self._exclusive_lock():
            lines = self._read_lines()
            existing = self._ideas_from_lines(lines)
            known_ids = {self.id_of(idea) for idea in existing if self.id_of(idea)}
            known_keys = {
                self.key(self.title_of(idea))
                for idea in existing
                if self.title_of(idea)
            }
            additions: list[dict[str, Any]] = []

            for idea in normalized:
                title = self.title_of(idea)
                idea_id = self.id_of(idea)
                title_key = self.key(title)
                if not title or idea_id in known_ids or title_key in known_keys:
                    continue
                entry = {"timestamp": iso_now(), **idea}
                if date and not entry.get("date"):
                    entry["date"] = date
                if origin and not entry.get("origin"):
                    entry["origin"] = origin
                additions.append(entry)
                known_ids.add(idea_id)
                known_keys.add(title_key)

            if additions:
                lines.extend(json.dumps(entry, ensure_ascii=False) for entry in additions)
                if self.max_entries > 0:
                    lines = lines[-self.max_entries :]
                self._atomic_write_lines(lines)
            return len(additions)

    def normalize(self, idea: dict[str, Any]) -> dict[str, Any]:
        """Replace response-local model IDs with stable content identity.

        Models commonly emit ``idea_001`` on every call. Treating that value as
        global identity caused later cycles to be silently discarded.
        """
        entry = dict(idea)
        title = self.title_of(entry)
        question = str(entry.get("research_question") or "").strip()
        canonical_id = self.canonical_id(title, question)
        supplied_id = self.id_of(entry)
        if supplied_id and supplied_id != canonical_id:
            entry.setdefault("source_model_id", supplied_id)
        entry["id"] = canonical_id
        entry["idea_id"] = canonical_id
        return entry

    @staticmethod
    def canonical_id(title: str, research_question: str = "") -> str:
        canonical = hashlib.sha256(
            f"{title.casefold()}\n{research_question.casefold()}".encode()
        ).hexdigest()[:12]
        return f"idea_{canonical}"

    def top(self, *, exclude_researched: bool = True) -> dict[str, Any] | None:
        ideas = self.load()
        researched_ids, researched_keys = self._researched_identity() if exclude_researched else (set(), set())
        candidates = [
            idea
            for idea in ideas
            if not (
                (self.id_of(idea) and self.id_of(idea) in researched_ids)
                or self.key(self.title_of(idea)) in researched_keys
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=self._score)

    def _score(self, idea: dict[str, Any]) -> float:
        raw = idea.get("total_score", idea.get("score", 0))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        return value * 100 if 0 <= value <= 1 else value

    def _cap(self) -> None:
        if self.max_entries <= 0:
            return
        with self._exclusive_lock():
            if not self.path.exists():
                return
            lines = self._read_lines()
            if len(lines) > self.max_entries:
                self._atomic_write_lines(lines[-self.max_entries :])

    def _read_lines(self) -> list[str]:
        if not self.path.exists():
            return []
        return [
            line
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _ideas_from_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
        ideas: list[dict[str, Any]] = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                ideas.append(value)
        return ideas

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # POSIX, including macOS and Linux
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _atomic_write_lines(self, lines: Iterable[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(line.rstrip("\r\n") + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        except BaseException:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise

    def _researched_identity(self) -> tuple[set[str], set[str]]:
        ids: set[str] = set()
        keys: set[str] = set()
        runs_dir = self.workspace / "runs"
        if not runs_dir.exists():
            return ids, keys
        for path in runs_dir.glob("*/run_manifest.json"):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if manifest.get("status") not in {"running", "completed"}:
                continue
            if manifest.get("idea_id"):
                ids.add(str(manifest["idea_id"]))
            key = self.key(str(manifest.get("idea_text") or ""))
            if key:
                keys.add(key)
        return ids, keys
