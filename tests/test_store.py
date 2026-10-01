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

    def test_redream_keeps_sparks_the_person_acted_on(self):
        store = self.app.store
        dream_id = store.save_dream("2026-09-30", title="t", reflection="r", undercurrent="u?", payload={}, models={})
        kept = store.add_spark(dream_id=dream_id, day="2026-09-30", title="kept", mechanism="orbit")
        store.add_spark(dream_id=dream_id, day="2026-09-30", title="fresh", mechanism="seed")
        store.set_spark_status(kept, "kept")
        again = store.save_dream("2026-09-30", title="t2", reflection="r", undercurrent="u?", payload={}, models={})
        self.assertEqual(again, dream_id)
        self.assertEqual([s["title"] for s in store.sparks(dream_id=dream_id)], ["kept"])
        with self.assertRaises(ValueError):
            store.set_spark_status(kept, "bogus")

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
