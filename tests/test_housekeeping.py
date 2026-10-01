"""Housekeeping keeps the harbour small and never forgets the main line."""

import json
import unittest
from datetime import date, timedelta

from helpers import TempApp

from unconscious.housekeeping import MAIN_LINE, tidy, trim_log
from unconscious.jobs import due_sorting, sort_job
from unconscious.mind.dream import DreamError, run_dream
from unconscious.mind.signals import compute_signals

END = date(2026, 9, 30)


def day(offset: int) -> str:
    return (END - timedelta(days=offset)).isoformat()


class HousekeepingTests(TempApp):
    def sail(self, days: int = 30) -> None:
        """A month at sea: the same few subjects, visited again and again, dreamt every night."""
        subjects = ["Decoy pricing SaaS pages", "Bike lane ridership data", "Sleep and impulse buying", "Churn panel models"]
        for offset in range(days - 1, -1, -1):
            d = day(offset)
            for visit in range(6):
                subject = subjects[(offset + visit) % len(subjects)]
                self.trace(d, subject, seconds=300 + 40 * visit, at=f"{9 + visit:02d}:{(offset * 7) % 50:02d}")
            self.trace(d, f"bottle {d}", seconds=0, kind="jot", key=f"jot:{d}", at="12:00", body="a half-formed thought")
            try:
                run_dream(self.app, d)
            except DreamError:  # the paper crew repeats itself; the day is still sorted into currents
                pass
        sparks = self.app.store.sparks()
        self.app.store.set_spark_status(sparks[0]["id"], "kept")
        self.app.store.set_spark_status(sparks[1]["id"], "dismissed", "generic")

    def main_line(self) -> dict[str, list]:
        with self.app.store.connect() as db:
            return {table: [tuple(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")] for table in MAIN_LINE}

    def what_a_dive_reads(self, d: str) -> list[tuple]:
        return [(s["subject_key"], round(s["seconds"], 1), s["visits"], s["body"]) for s in self.app.store.subjects(d)]

    def signals(self, d: str) -> str:
        found, stats = compute_signals(self.app, d)
        return json.dumps([s.to_dict() for s in found], sort_keys=True, default=str) + repr([(s.thread["id"], s.series) for s in stats])

    def test_the_main_line_survives_a_year_of_tidying(self):
        self.sail()
        tomorrow = (END + timedelta(days=1)).isoformat()  # undercurrents measured from history alone
        before_line, before_signals, before_history = self.main_line(), self.signals(day(0)), self.signals(tomorrow)
        old_day = day(20)
        before_reads, before_rows = self.what_a_dive_reads(old_day), self.app.store.trace_count(old_day)

        done = tidy(self.app, END + timedelta(days=1))
        self.assertGreater(done["merged"], 0)
        self.assertLess(self.app.store.trace_count(old_day), before_rows, "repeat visits merge after two weeks")
        self.assertEqual(self.what_a_dive_reads(old_day), before_reads, "a dive reads exactly the same subjects")
        recent = day(3)
        self.assertEqual(self.app.store.trace_count(recent), 7, "the last two weeks keep every visit")
        self.assertEqual(self.main_line(), before_line)
        self.assertEqual(self.signals(day(0)), before_signals, "tonight's undercurrents are unchanged")
        self.assertEqual(self.signals(tomorrow), before_history)
        # the digest cache stays valid for the merged day: re-sorting it would not call the crew again
        self.assertEqual(self.app.store.digest(old_day)["trace_count"], self.app.store.trace_count(old_day))

        tidy(self.app, END + timedelta(days=365), force=True)
        self.assertEqual(self.app.store.trace_count(), 0, "raw driftlines wash away after retention_days")
        with self.app.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM digests").fetchone()[0], 0)
        self.assertEqual(self.main_line(), before_line, "currents, dreams, fish and the net are never tidied away")
        self.assertEqual(self.signals(tomorrow), before_history, "the history undercurrents are read from is whole")

    def test_tidy_runs_once_a_day(self):
        self.assertIsNotNone(tidy(self.app, END))
        self.assertIsNone(tidy(self.app, END))
        self.assertIsNotNone(tidy(self.app, END + timedelta(days=1)))

    def test_the_shark_gives_the_space_back(self):
        filler = "x" * 900
        with self.app.store.tx() as db:
            for i in range(6000):
                db.execute(
                    "INSERT INTO traces(day, started_at, ended_at, seconds, kind, source, title, body, subject_key, subject) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (day(i % 30), f"{day(i % 30)}T10:00:00+00:00", f"{day(i % 30)}T10:01:00+00:00", 60, "focus", "test",
                     f"page {i}", filler, f"web:{i}", f"page {i}"),
                )
        full = self.app.store.size()
        self.assertGreater(full, 5_000_000)
        self.app.store.forget_everything()
        self.assertLess(self.app.store.size(), full * 0.1)

    def test_old_logs_leave_and_recent_ones_stay(self):
        with self.app.store.tx() as db:
            for stamp in ("2025-01-01T09:00:00+00:00", "2026-09-29T09:00:00+00:00"):
                db.execute("INSERT INTO jobs(id, kind, ref, state, created_at) VALUES(?,?,?,?,?)", (stamp, "dream", "d", "done", stamp))
                db.execute("INSERT INTO llm_calls(ts, role, provider, model, ok, ms) VALUES(?,?,?,?,?,?)", (stamp, "dream", "fake", "m", 1, 5))
        self.assertEqual(self.app.store.prune_logs(END.isoformat()), 2)
        with self.app.store.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0], 1)

    def test_a_long_log_keeps_its_tail(self):
        path = self.home / "logs" / "dun.log"
        path.parent.mkdir()
        path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(200_000)))
        self.assertTrue(trim_log(path))
        tail = path.read_bytes()
        self.assertLessEqual(len(tail), 256 * 1024)
        self.assertTrue(tail.startswith(b"line ") and tail.endswith(b"line 199999\n"))
        self.assertFalse(trim_log(path))


class CatchUpSortingTests(TempApp):
    def test_every_day_is_sorted_into_its_currents_before_it_can_wash_away(self):
        for offset in (5, 4, 1, 0):
            for visit in range(4):
                self.trace(day(offset), "Decoy pricing SaaS pages", seconds=400, at=f"1{visit}:00")
        now = __import__("datetime").datetime.fromisoformat(f"{day(0)}T22:00:00").astimezone()
        self.assertEqual(due_sorting(self.app, now), [day(5)], "oldest first; yesterday is the dream catch-up's")
        sort_job(self.app, day(5))(lambda _step: None)
        self.assertTrue(self.app.store.thread_days(day(5)), "sorting writes the day into a current's history")
        self.assertEqual(due_sorting(self.app, now), [day(4)])
        self.app.update_settings({"dream": {"auto": False}})
        self.assertEqual(due_sorting(self.app, now), [], "no automatic crew calls when night diving is off")


if __name__ == "__main__":
    unittest.main()
