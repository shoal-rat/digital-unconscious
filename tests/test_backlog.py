from __future__ import annotations

import json
import multiprocessing
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from du_research.backlog import IdeaBacklog  # noqa: E402


def _append_batch(
    workspace: str,
    max_entries: int,
    batch: list[dict[str, str]],
    ready_queue,
    start_event,
    result_queue,
) -> None:
    """Start several real processes at the same point in append_unique()."""
    ready_queue.put(True)
    if not start_event.wait(timeout=20):
        raise TimeoutError("backlog concurrency test did not start")
    try:
        added = IdeaBacklog(workspace, max_entries=max_entries).append_unique(batch)
    except BaseException as exc:
        result_queue.put(("error", repr(exc)))
        raise
    result_queue.put(("ok", added))


class IdeaBacklogWriteTests(unittest.TestCase):
    def _run_concurrent_batches(
        self,
        workspace: str,
        batches: list[list[dict[str, str]]],
        *,
        max_entries: int,
    ) -> list[int]:
        context = multiprocessing.get_context("spawn")
        ready_queue = context.Queue()
        start_event = context.Event()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_append_batch,
                args=(
                    workspace,
                    max_entries,
                    batch,
                    ready_queue,
                    start_event,
                    result_queue,
                ),
            )
            for batch in batches
        ]

        try:
            for process in processes:
                process.start()
            for _ in processes:
                ready_queue.get(timeout=20)
            start_event.set()
            for process in processes:
                process.join(timeout=20)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail("concurrent backlog writer did not exit")
                self.assertEqual(process.exitcode, 0)

            results = [result_queue.get(timeout=5) for _ in processes]
        except queue.Empty as exc:
            self.fail(f"concurrent backlog writer did not report a result: {exc}")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            ready_queue.close()
            result_queue.close()

        errors = [value for status, value in results if status == "error"]
        self.assertEqual(errors, [])
        return [int(value) for status, value in results if status == "ok"]

    def test_concurrent_unique_batches_do_not_lose_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batches = [
                [
                    {
                        "id": f"model_{worker}_{index}",
                        "title": f"Worker {worker} idea {index}",
                    }
                    for index in range(20)
                ]
                for worker in range(4)
            ]

            added = self._run_concurrent_batches(tmpdir, batches, max_entries=100)
            backlog = IdeaBacklog(tmpdir, max_entries=100)
            values = backlog.load()

            self.assertEqual(sum(added), 80)
            self.assertEqual(len(values), 80)
            self.assertEqual(len({backlog.id_of(value) for value in values}), 80)
            for line in backlog.path.read_text(encoding="utf-8").splitlines():
                self.assertIsInstance(json.loads(line), dict)

    def test_concurrent_duplicate_is_appended_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batches = [
                [{"id": f"model_{worker}", "title": "The same research question"}]
                for worker in range(6)
            ]

            added = self._run_concurrent_batches(tmpdir, batches, max_entries=100)
            values = IdeaBacklog(tmpdir, max_entries=100).load()

            self.assertEqual(sum(added), 1)
            self.assertEqual(len(values), 1)
            self.assertEqual(values[0]["title"], "The same research question")

    def test_concurrent_writes_preserve_retention_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batches = [
                [{"title": f"Batch {worker} idea {index}"} for index in range(20)]
                for worker in range(4)
            ]

            self._run_concurrent_batches(tmpdir, batches, max_entries=25)
            backlog = IdeaBacklog(tmpdir, max_entries=25)
            values = backlog.load()

            self.assertEqual(len(values), 25)
            self.assertEqual(len({backlog.id_of(value) for value in values}), 25)
            self.assertTrue(backlog.path.read_bytes().endswith(b"\n"))

    def test_append_keeps_order_while_capping_oldest_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backlog = IdeaBacklog(tmpdir, max_entries=3)
            backlog.append_unique(
                [{"title": "First"}, {"title": "Second"}],
                date="2026-07-12",
                origin="test",
            )
            backlog.append_unique([{"title": "Third"}, {"title": "Fourth"}])

            values = backlog.load()
            self.assertEqual([value["title"] for value in values], ["Second", "Third", "Fourth"])
            self.assertEqual(values[0]["date"], "2026-07-12")
            self.assertEqual(values[0]["origin"], "test")

    def test_failed_replace_leaves_previous_backlog_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backlog = IdeaBacklog(tmpdir, max_entries=10)
            backlog.append_unique([{"title": "Existing idea"}])
            original = backlog.path.read_bytes()

            with mock.patch("du_research.backlog.os.replace", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    backlog.append_unique([{"title": "New idea"}])

            self.assertEqual(backlog.path.read_bytes(), original)
            temporary_files = list(backlog.path.parent.glob(f".{backlog.path.name}.*.tmp"))
            self.assertEqual(temporary_files, [])


if __name__ == "__main__":
    unittest.main()
