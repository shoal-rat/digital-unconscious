import unittest
from datetime import date, timedelta

from helpers import TempApp

from unconscious.llm.base import LLMResult
from unconscious.mind.digest import apply_digest, digest_day, prepare_subjects
from unconscious.mind.dive import run_dive
from unconscious.mind.dream import DreamError, run_dream, select
from unconscious.mind.signals import compute_signals

DAY = "2026-09-30"


def days_before(n: int) -> str:
    return (date.fromisoformat(DAY) - timedelta(days=n)).isoformat()


class DigestTests(TempApp):
    def test_refs_are_validated_and_duplicates_reuse_threads(self):
        self.trace(DAY, "Decoy pricing on SaaS pages", seconds=900)
        self.trace(DAY, "Bike lane ridership", seconds=300, at="12:00")
        existing = self.app.store.create_thread("SaaS pricing psychology", "", ["pricing", "decoy"], days_before(3))
        subjects = prepare_subjects(self.app, DAY)
        by_ref = {s["ref"]: s for s in subjects}
        data = {"topics": [
            {"label": "Pricing", "gist": "g", "subjects": ["S1", "S99"], "thread": f"T{existing}",
             "thread_name": "", "thread_gist": "", "keywords": ["pricing"]},
            {"label": "Bikes", "gist": "g", "subjects": ["S2", "S1"], "thread": "new",
             "thread_name": "Bike lanes", "thread_gist": "lanes", "keywords": ["bike"]},
            {"label": "Again", "gist": "g", "subjects": ["S2"], "thread": "new",
             "thread_name": "SaaS Pricing Psychology", "thread_gist": "", "keywords": []},
        ]}
        topics = apply_digest(self.app, DAY, data, by_ref, [])
        self.assertEqual(len(topics), 2)  # the third topic had only already-claimed refs
        names = {t["name"] for t in self.app.store.threads()}
        self.assertEqual(names, {"SaaS pricing psychology", "Bike lanes"})
        self.assertEqual(len(self.app.store.subject_threads(DAY)), 2)

    def test_digest_is_cached_until_new_traces_arrive(self):
        self.trace(DAY, "Decoy pricing", seconds=900)
        first = digest_day(self.app, DAY)
        self.assertFalse(first.cached)
        self.assertTrue(digest_day(self.app, DAY).cached)
        self.trace(DAY, "Menu engineering", seconds=900, at="15:00")
        self.assertFalse(digest_day(self.app, DAY).cached)


class SignalTests(TempApp):
    def history(self, thread_id, days_and_seconds):
        for offset, seconds in days_and_seconds:
            day = days_before(offset)
            self.app.store.replace_day_assignments(day, [{"thread_id": thread_id, "seconds": seconds, "visits": 2}], {})

    def test_orbit_return_surge_seed_and_collision(self):
        store = self.app.store
        orbit = store.create_thread("Bike lanes commuting", "", ["bike", "lanes"], days_before(20))
        back = store.create_thread("Sleep and decisions", "", ["sleep"], days_before(25))
        surge = store.create_thread("Language model evaluation", "", ["llm", "eval"], days_before(20))
        seed = store.create_thread("Menu design", "", ["menu", "restaurant"], DAY)
        for day_offset, items in {
            0: [(orbit, 240), (back, 1500), (surge, 5400), (seed, 1800)],
            1: [(orbit, 200)], 3: [(orbit, 300), (surge, 600)], 5: [(orbit, 180)], 8: [(orbit, 260), (surge, 700)],
            12: [(back, 1200), (surge, 500)], 13: [(back, 1300)], 16: [(back, 900)],
        }.items():
            day = days_before(day_offset)
            store.replace_day_assignments(day, [{"thread_id": tid, "seconds": s, "visits": 2} for tid, s in items], {})
        self.trace(DAY, "menu", seconds=1800, key="m", at="14:00")
        self.trace(DAY, "sleep", seconds=1500, key="s", at="14:40")
        store.replace_day_assignments(DAY, [
            {"thread_id": orbit, "seconds": 240, "visits": 2}, {"thread_id": back, "seconds": 1500, "visits": 2},
            {"thread_id": surge, "seconds": 5400, "visits": 2}, {"thread_id": seed, "seconds": 1800, "visits": 2},
        ], {"m": seed, "s": back})
        signals, _stats = compute_signals(self.app, DAY, "en")
        kinds = {(s.kind, tuple(s.threads)) for s in signals}
        self.assertIn(("orbit", (orbit,)), kinds)
        self.assertIn(("return", (back,)), kinds)
        self.assertIn(("surge", (surge,)), kinds)
        self.assertIn(("seed", (seed,)), kinds)
        collisions = [s for s in signals if s.kind == "collision"]
        self.assertTrue(collisions)
        pair = next(s for s in collisions if set(s.threads) == {seed, back})
        self.assertTrue(pair.facts["adjacent"])
        orbit_text = next(s.text for s in signals if s.kind == "orbit")
        self.assertIn("Bike lanes commuting", orbit_text)

    def test_muted_threads_are_silent(self):
        tid = self.app.store.create_thread("Apartment hunting", "", ["rent"], DAY)
        self.app.store.update_thread(tid, state="muted")
        self.history(tid, [(0, 1800)])
        signals, _ = compute_signals(self.app, DAY, "en")
        self.assertFalse(any(tid in s.threads for s in signals))


class DreamTests(TempApp):
    def seed_day(self):
        self.trace(DAY, "Decoy pricing SaaS pages", seconds=1500, at="09:00")
        self.trace(DAY, "pricing tier decoy experiment", seconds=900, at="10:00", kind="search", key="search:pricing tier decoy")
        self.trace(DAY, "Bike lane ridership data", seconds=400, at="11:00")
        self.trace(DAY, "late night signups pricier plans", seconds=0, at="12:00", kind="jot", key="jot:1",
                   body="Do late night signups pick pricier plans?")

    def test_full_pipeline_with_offline_model(self):
        self.seed_day()
        steps = []
        result = run_dream(self.app, DAY, steps.append)
        self.assertEqual(steps, ["gather", "digest", "signals", "dream", "critique", "save"])
        dream = self.app.store.dream(DAY)
        self.assertTrue(dream["title"])
        sparks = self.app.store.sparks(dream_id=dream["id"])
        self.assertEqual(len(sparks), len(result["sparks"]))
        self.assertTrue(all(s["evidence"] or s["thread_ids"] for s in sparks))
        self.assertTrue(all(0 < s["score"] <= 100 for s in sparks))

    def test_empty_day_refuses_to_dream(self):
        with self.assertRaises(DreamError):
            run_dream(self.app, DAY)

    def test_invented_refs_are_rejected(self):
        self.seed_day()
        fake = self.app.router.providers["fake"]
        original = fake.complete

        def lying(request, model):
            result = original(request, model)
            if request.role == "dream":
                for spark in result.data["sparks"]:
                    spark["evidence"] = ["S404"]
                    spark["threads"] = ["T999"]
            return result

        fake.complete = lying
        with self.assertRaises(DreamError) as caught:
            run_dream(self.app, DAY)
        self.assertIn("cites no valid evidence", str(caught.exception))

    def test_selection_prefers_diversity(self):
        cards = [
            {"title": "a", "mechanism": "orbit", "thread_refs": ["T1"], "score": 90},
            {"title": "b", "mechanism": "orbit", "thread_refs": ["T1"], "score": 85},
            {"title": "c", "mechanism": "collision", "thread_refs": ["T1", "T2"], "score": 60},
            {"title": "d", "mechanism": "seed", "thread_refs": ["T3"], "score": 30},
        ]
        self.assertEqual([c["title"] for c in select(cards, 3)], ["a", "c"])

    def test_taste_nudges_ranking(self):
        self.seed_day()
        run_dream(self.app, DAY)
        spark = self.app.store.sparks()[0]
        for _ in range(2):
            self.app.store.set_spark_status(spark["id"], "kept")
        from unconscious.mind.taste import load_taste

        taste = load_taste(self.app)
        self.assertEqual(taste.kept, 1)
        self.assertIn("Kept or pursued 1", taste.to_prompt())


class DiveTests(TempApp):
    def test_dive_validates_citations(self):
        dream_id = self.app.store.save_dream(DAY, title="t", reflection="r", undercurrent="u?", payload={}, models={})
        spark_id = self.app.store.add_spark(dream_id=dream_id, day=DAY, title="Plan names as load", mechanism="orbit",
                                            question="q?", search_terms=["plan names churn"])
        papers = [{"title": f"Paper {i}", "abstract": f"Abstract {i}", "authors": [], "year": 2020, "venue": "J"} for i in range(6)]
        fake = self.app.router.providers["fake"]
        original = fake.complete

        def with_bad_ref(request, model):
            result = original(request, model)
            result.data["known"].append({"point": "made up", "refs": [42]})
            return LLMResult(True, "", result.data, "fake", "offline")

        fake.complete = with_bad_ref
        outcome = run_dive(self.app, spark_id, search=lambda terms: papers)
        dive = self.app.store.dives(spark_id)[0]
        self.assertEqual(outcome["papers"], 6)
        self.assertTrue(all(ref <= 6 for item in dive["report"]["known"] for ref in item["refs"]))
        self.assertEqual(self.app.store.spark(spark_id)["status"], "pursuing")

    def test_a_research_dive_reads_full_texts_in_a_folder_that_leaves_with_it(self):
        from unconscious import scholar

        dream_id = self.app.store.save_dream(DAY, title="t", reflection="r", undercurrent="u?", payload={}, models={})
        spark_id = self.app.store.add_spark(dream_id=dream_id, day=DAY, title="Decoys", mechanism="seed",
                                            question="q?", search_terms=["decoy pricing"])
        papers = [{"title": f"Paper {i}", "abstract": "A", "authors": [], "year": 2020, "venue": "J",
                   "pdf": f"https://example.org/{i}.pdf" if i != 2 else ""} for i in range(1, 7)]

        def fetch(url):
            if url.endswith("3.pdf"):
                return b"<html>paywall</html>"  # not a PDF: left out
            if url.endswith("4.pdf"):
                raise OSError("dead link")
            return b"%PDF-1.7 " + url.encode()

        seen = {}
        call = self.app.router.call

        def watch(request):
            seen["request"] = request
            seen["files"] = sorted(p.name for p in (request.workdir / "papers").iterdir()) if request.workdir else []
            return call(request)

        self.app.router.call = watch
        self.app.update_settings({"models": {"research": True}})
        run_dive(self.app, spark_id, search=lambda terms: papers, fetch=fetch)
        request = seen["request"]
        self.assertTrue(request.research)
        self.assertEqual(seen["files"], ["01.pdf", "05.pdf", "06.pdf"][: scholar.PDF_LIMIT])
        self.assertIn("FULL TEXTS IN ./papers", request.prompt)
        self.assertIn("[5] papers/05.pdf", request.prompt)
        self.assertFalse(request.workdir.exists(), "the folder and the papers in it are gone")
        self.assertEqual(self.app.store.dives(spark_id)[0]["report"]["full_texts"], [1, 5, 6])

        self.app.update_settings({"models": {"research": False}})
        run_dive(self.app, spark_id, search=lambda terms: papers, fetch=lambda url: self.fail("no downloads"))
        self.assertFalse(seen["request"].research)
        self.assertIsNone(seen["request"].workdir)
        self.assertNotIn("FULL TEXTS", seen["request"].prompt)


if __name__ == "__main__":
    unittest.main()
