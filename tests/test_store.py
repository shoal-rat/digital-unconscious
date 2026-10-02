import unittest

from helpers import TempApp


class StoreTests(TempApp):
    def test_day_assignments_are_replaced_not_accumulated(self):
        store = self.app.store
        tid = store.create_thread("Pricing", "", ["pricing"], "2026-09-30")
        for _ in range(2):
            store.replace_day_assignments("2026-09-30", [{"thread_id": tid, "seconds": 600, "visits": 3, "subjects": ["a"]}], {"k": tid})
        [row] = store.thread_days("2026-09-01")
        self.assertEqual(row["seconds"], 600)
        self.assertEqual(store.subject_threads("2026-09-30"), {"k": tid})

    def test_merge_moves_history(self):
        store = self.app.store
        a = store.create_thread("A", "", [], "2026-09-28")
        b = store.create_thread("B", "", [], "2026-09-29")
        store.replace_day_assignments("2026-09-28", [{"thread_id": a, "seconds": 100, "visits": 1, "subjects": ["x"]}], {"x": a})
        store.replace_day_assignments("2026-09-29", [{"thread_id": a, "seconds": 50, "visits": 1, "subjects": ["y"]},
                                                     {"thread_id": b, "seconds": 70, "visits": 1, "subjects": ["z"]}], {"y": a, "z": b})
        store.merge_threads(a, b)
        days = {r["day"]: r["seconds"] for r in store.thread_days("2026-09-01", b)}
        self.assertEqual(days, {"2026-09-28": 100, "2026-09-29": 120})
        self.assertEqual(store.thread(a)["state"], "merged")
        self.assertEqual(store.thread(b)["first_day"], "2026-09-28")

    def test_a_second_dive_keeps_the_first_whole(self):
        store = self.app.store
        first = store.save_dream("2026-09-30", title="t1", reflection="r1", undercurrent="u1?", payload={}, models={},
                                 sparks=[{"title": "kept", "mechanism": "orbit"}, {"title": "untouched", "mechanism": "seed"}])
        kept, untouched = sorted(s["id"] for s in store.sparks(dream_id=first))
        store.set_spark_status(kept, "kept")
        second = store.save_dream("2026-09-30", title="t2", reflection="r2", undercurrent="u2?", payload={}, models={},
                                  sparks=[{"title": "fresh", "mechanism": "gap"}])
        self.assertNotEqual(second, first)
        self.assertEqual([d["title"] for d in store.day_dives("2026-09-30")], ["t2", "t1"], "both dives stay, latest first")
        self.assertEqual(store.dream("2026-09-30")["title"], "t2")
        self.assertEqual({s["title"]: s["status"] for s in store.sparks(dream_id=first)}, {"kept": "kept", "untouched": "drifted"})
        self.assertEqual([s["title"] for s in store.sparks(status="new")], ["fresh"])
        store.set_spark_status(untouched, "kept")
        store.set_spark_status(untouched, "new")  # undone: an earlier dive's fish drifts out again, never back to new
        self.assertEqual(store.spark(untouched)["status"], "drifted")
        fresh = store.sparks(dream_id=second)[0]["id"]
        store.set_spark_status(fresh, "kept")
        store.set_spark_status(fresh, "new")
        self.assertEqual(store.spark(fresh)["status"], "new")
        [entry] = store.dreams()
        self.assertEqual((entry["title"], entry["dives"], entry["spark_count"]), ("t2", 2, 2), "one logbook line a day")
        self.assertEqual((store.dive_number(first), store.dive_number(second), store.dream_number("2026-09-30")), (1, 2, 2))
        store.save_dream("2026-09-29", title="yesterday, later", reflection="", undercurrent="", payload={}, models={})
        self.assertEqual(store.dream_number("2026-09-30"), 2, "a dive's number never moves")
        with self.assertRaises(ValueError):
            store.set_spark_status(kept, "bogus")

    def test_memories_with_one_dive_a_day_open_unchanged(self):
        import sqlite3

        from unconscious.store import Store

        path = self.home / "old.db"
        db = sqlite3.connect(path)
        db.executescript("""
            CREATE TABLE dreams(id INTEGER PRIMARY KEY, day TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
              title TEXT NOT NULL DEFAULT '', reflection TEXT NOT NULL DEFAULT '', undercurrent TEXT NOT NULL DEFAULT '',
              payload TEXT NOT NULL DEFAULT '{}', models TEXT NOT NULL DEFAULT '{}');
            INSERT INTO dreams(id, day, created_at, title) VALUES (4, '2026-09-29', '2026-09-29T21:30:00+08:00', 'old one');
            INSERT INTO dreams(id, day, created_at, title) VALUES (7, '2026-09-30', '2026-09-30T21:30:00+08:00', 'old two');
        """)
        db.commit()
        db.close()
        store = Store(path)
        self.assertEqual([(d["id"], d["title"]) for d in store.dreams()], [(7, "old two"), (4, "old one")])
        store.save_dream("2026-09-30", title="new", reflection="", undercurrent="", payload={}, models={})
        self.assertEqual([d["title"] for d in store.day_dives("2026-09-30")], ["new", "old two"])
        Store(path)  # opening again changes nothing
        self.assertEqual(len(store.day_dives("2026-09-30")), 2)
        with store.connect() as db:
            self.assertEqual(db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0], "3")
        self.assertTrue(store.get("stale_since"), "an upgrade leaves the days dived before it as they were sorted")

    def test_retention_forgets_raw_traces_only(self):
        self.trace("2026-01-01", "old page")
        self.trace("2026-09-30", "new page")
        tid = self.app.store.create_thread("T", "", [], "2026-01-01")
        self.app.store.replace_day_assignments("2026-01-01", [{"thread_id": tid, "seconds": 600, "visits": 1}], {})
        removed = self.app.store.forget_before("2026-07-01")
        self.assertEqual(removed, 1)
        self.assertEqual(self.app.store.trace_count(), 1)
        self.assertEqual(len(self.app.store.thread_days("2025-01-01")), 1)

    def test_jobs_lifecycle(self):
        store = self.app.store
        job = store.create_job("dream", "2026-09-30")
        store.update_job(job, state="running", step="digest")
        self.assertEqual(store.active_jobs()[0]["step"], "digest")
        store.fail_stale_jobs()
        self.assertEqual(store.job(job)["state"], "failed")


if __name__ == "__main__":
    unittest.main()
