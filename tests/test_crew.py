"""When the crew cannot sail: the region guard, signing out, limits, no network.

Nothing here touches the network: region answers and sign-in probes are scripted.
"""

import json
import unittest

from helpers import TempApp

from unconscious.config import Settings
from unconscious.jobs import counted, crew_ready, scheduler_tick
from unconscious.llm.base import LLMRequest, LLMResult
from unconscious.llm.crew import LIMIT, OFFLINE, REGION, SIGN_IN, Crew, classify, parse
from unconscious.llm.region import RegionCheck, _country_is, _plain, _trace
from unconscious.llm.router import Router
from unconscious.ui.i18n import crew_message, set_language


class Sailor:
    """A scripted crew member: replies in order, counts errands, answers sign-in probes."""

    def __init__(self, name, replies=(), signed_in=True):
        self.name = name
        self.replies = list(replies)
        self.calls = 0
        self._signed_in = signed_in

    def available(self):
        return True

    def signed_in(self):
        return self._signed_in

    def complete(self, request, model):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else {"answer": "ok"}
        if isinstance(reply, str):
            return LLMResult(False, provider=self.name, model=model or "", error=reply)
        return LLMResult(True, json.dumps(reply), reply, self.name, model or "m")


class Where:
    """Scripted answers to "where is the connection?", one per lookup."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = 0

    def __call__(self):
        self.asked += 1
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def ask(router, role="dream"):
    return router.call(LLMRequest(role, "s", "p"))


def build(settings=None, where=None, clock=None, **crew):
    settings = settings or Settings()
    return Router(settings, None, providers=crew, region=RegionCheck(fetch=where or Where("US")),
                  crew=Crew(None, clock=clock or Clock()))


class RegionTests(unittest.TestCase):
    def test_reading_the_endpoints(self):
        self.assertEqual(_trace("fl=1\nip=1.2.3.4\nloc=CN\ncolo=HKG\n"), "CN")
        self.assertEqual(_country_is('{"ip":"1.2.3.4","country":"US"}'), "US")
        self.assertEqual(_plain("JP\n"), "JP")
        self.assertIsNone(_plain("<html>"))

    def test_in_mainland_china_claude_and_codex_stay_ashore_and_others_sail(self):
        claude, codex, deepseek = Sailor("claude"), Sailor("codex"), Sailor("deepseek")
        result = ask(build(where=Where("CN"), claude=claude, codex=codex, deepseek=deepseek))
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "deepseek")
        self.assertEqual((claude.calls, codex.calls), (0, 0), "nothing at all is sent to them")

    def test_with_no_one_else_aboard_the_errand_waits_with_a_reason(self):
        claude = Sailor("claude")
        result = ask(build(where=Where("CN"), claude=claude))
        self.assertFalse(result.ok)
        self.assertEqual(parse(result.error), (REGION, "claude", "CN"))
        self.assertEqual(claude.calls, 0)

    def test_the_connection_is_looked_up_before_every_errand(self):
        where, claude = Where("US", "CN", "US"), Sailor("claude")
        router = build(where=where, claude=claude)
        self.assertTrue(ask(router).ok)
        self.assertFalse(ask(router).ok, "switched into China between two errands: held")
        self.assertTrue(ask(router).ok, "and back out: sailing again")
        self.assertEqual((where.asked, claude.calls), (3, 2))

    def test_offline_means_unsure_and_unsure_means_wait(self):
        claude = Sailor("claude")
        result = ask(build(where=Where(None), claude=claude))
        self.assertEqual(parse(result.error), (REGION, "claude", "?"))
        self.assertEqual(claude.calls, 0)

    def test_the_guard_can_be_turned_off(self):
        settings = Settings()
        settings.update({"models": {"region_guard": False}})

        def never():
            raise AssertionError("no lookup when the guard is off")

        self.assertTrue(ask(build(settings, where=never, claude=Sailor("claude"))).ok)

    def test_answers_are_kept_briefly_and_unsure_ones_more_briefly(self):
        where = Where("US")
        check = RegionCheck(fetch=where)
        check.current()
        check.current()
        self.assertEqual(where.asked, 1)
        check._fix.at -= 301
        check.current()
        self.assertEqual(where.asked, 2, "a known answer is kept five minutes")
        unsure = RegionCheck(fetch=Where(None))
        unsure.current()
        self.assertEqual(unsure._keep(unsure._fix), 120)


class TroubleTests(unittest.TestCase):
    def test_errors_are_sorted_into_troubles(self):
        cases = {
            "Invalid API key · Please run /login": SIGN_IN,
            "OAuth token has expired. Please obtain a new token": SIGN_IN,
            "Error: Not logged in": SIGN_IN,
            "Claude AI usage limit reached|1759300000": LIMIT,
            "API Error: 529 Overloaded": LIMIT,
            "API Error: Connection error.": OFFLINE,
            "getaddrinfo ENOTFOUND api.anthropic.com": OFFLINE,
            "timed out": OFFLINE,
            "API Error: 403 Request not allowed": REGION,
            "invalid JSON: answer is required": None,
            "Every candidate spark was rejected": None,
        }
        for error, kind in cases.items():
            self.assertEqual(classify(error), kind, error)

    def test_signed_out_claude_hands_over_and_comes_back_after_signing_in(self):
        clock = Clock()
        claude = Sailor("claude", ["Invalid API key · Please run /login"], signed_in=False)
        codex = Sailor("codex")
        router = build(clock=clock, claude=claude, codex=codex)
        first = ask(router)
        self.assertEqual(first.provider, "codex", "fallback takes the errand")
        self.assertEqual(router.crew.troubles["claude"].kind, SIGN_IN)
        ask(router)
        self.assertEqual(claude.calls, 1, "while it pauses, claude is not asked again")
        clock.now += 301
        ask(router)
        self.assertEqual(claude.calls, 1, "the probe says still signed out: keep waiting, no errand")
        self.assertEqual(router.crew.troubles["claude"].strikes, 2)
        claude._signed_in = True
        clock.now += 10_000
        self.assertEqual(ask(router).provider, "claude", "signed in again: back at sea")
        self.assertNotIn("claude", router.crew.troubles)

    def test_pauses_grow_and_have_a_ceiling(self):
        clock = Clock()
        crew = Crew(None, clock=clock)
        pauses = []
        for _ in range(8):
            crew.failed("claude", "API Error: Connection error.")
            pauses.append(crew.troubles["claude"].until - clock.now)
        self.assertEqual(pauses[:3], [300, 600, 1200])
        self.assertEqual(pauses[-1], 7200)

    def test_a_real_failure_is_not_a_trouble(self):
        claude = Sailor("claude", ["the model said something unusable"])
        result = ask(build(claude=claude))
        self.assertIsNone(parse(result.error))
        self.assertNotIn("claude", build().crew.troubles)

    def test_messages_in_both_languages(self):
        set_language("en")
        self.assertIn("mainland China", crew_message("crew:region:claude:CN"))
        self.assertIn("claude auth login", crew_message("crew:sign_in:claude:Invalid API key"))
        self.assertIn("offline", crew_message("crew:region:claude:?"))
        set_language("zh")
        self.assertIn("中国大陆", crew_message("crew:region:claude:CN"))
        self.assertIn("重新登录", crew_message("crew:sign_in:codex:"))
        set_language("en")
        self.assertEqual(crew_message("plain error"), "plain error")


class NightWatchTests(TempApp):
    def test_waiting_for_the_crew_does_not_spend_attempts(self):
        def held(_progress):
            raise RuntimeError("crew:offline:claude:Connection error")

        def broken(_progress):
            raise RuntimeError("the model said something unusable")

        for work in (held, held, held):
            with self.assertRaises(RuntimeError):
                counted(self.app, "2026-09-30", "auto_dream_attempts", work)(lambda _s: None)
        self.assertEqual(self.app.store.get("auto_dream_attempts", {}) or {}, {})
        with self.assertRaises(RuntimeError):
            counted(self.app, "2026-09-30", "auto_dream_attempts", broken)(lambda _s: None)
        self.assertEqual(self.app.store.get("auto_dream_attempts"), {"2026-09-30": 1})

    def test_no_dive_is_queued_while_the_crew_cannot_sail(self):
        class Jobs:
            def __init__(self):
                self.submitted = []

            def submit(self, kind, ref, work):
                self.submitted.append((kind, ref))

        self.trace("2026-09-30", "Decoy pricing", seconds=3000)
        self.app.update_settings({"dream": {"time": "00:00"}})
        jobs = Jobs()
        self.app._router.ready = lambda role: False
        self.assertFalse(crew_ready(self.app))
        scheduler_tick(self.app, jobs)
        self.assertEqual(jobs.submitted, [])
        self.app._router.ready = lambda role: True
        scheduler_tick(self.app, jobs)
        self.assertTrue(jobs.submitted, "and once it can, the dive is queued")

    def test_the_shore_hears_about_troubles_without_asking_anyone(self):
        from unconscious import api

        crew = Crew(self.app.store)
        crew.failed("claude", "Invalid API key · Please run /login")
        self.app._router.region = RegionCheck(fetch=lambda: "CN")
        self.app._router.region.current()
        status = api.crew_status(self.app)
        self.assertIn(("claude", SIGN_IN), [(s["who"], s["kind"]) for s in status])
        self.assertIn(("crew", REGION), [(s["who"], s["kind"]) for s in status])
        self.assertIn("claude", Crew(self.app.store).troubles, "pauses outlive a restart")


if __name__ == "__main__":
    unittest.main()
