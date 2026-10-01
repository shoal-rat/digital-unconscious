import unittest
from datetime import datetime, timedelta

from helpers import TempApp

from unconscious.sense.sensors import Sample
from unconscious.sense.watcher import Watcher


class FakeSensor:
    def __init__(self):
        from unconscious.sense.sensors import Capabilities

        self.capabilities = Capabilities(platform="test", app=True, titles=True)

    def sample(self):
        raise AssertionError("tests drive tick() directly")


class WatcherTests(TempApp):
    def setUp(self):
        super().setUp()
        self.watcher = Watcher(self.app, sensor=FakeSensor())
        self.t0 = datetime(2026, 9, 30, 10, 0, 0).astimezone()

    def at(self, seconds):
        return self.t0 + timedelta(seconds=seconds)

    def test_dwell_accumulates_and_switches_subject(self):
        page = Sample("Safari", "Pricing Complexity and Churn", "https://www.nber.org/papers/w1", 0)
        other = Sample("Preview", "notes.pdf", "", 0)
        for i in range(5):
            self.watcher.tick(page, self.at(i * 15))
        self.watcher.tick(other, self.at(75))
        self.watcher.tick(other, self.at(90))
        subjects = {s["subject"]: s for s in self.app.store.subjects("2026-09-30")}
        self.assertEqual(subjects["Pricing Complexity and Churn"]["seconds"], 75)
        self.assertEqual(subjects["notes.pdf"]["seconds"], 15)
        self.assertEqual(len(self.app.store.traces("2026-09-30")), 2)

    def test_idle_time_is_not_credited(self):
        page = Sample("Safari", "Long read", "", 0)
        self.watcher.tick(page, self.at(0))
        self.watcher.tick(Sample("Safari", "Long read", "", 30), self.at(15))
        # 200 s idle at the next tick with a 120 s threshold: only (15 - (200 - 120)) clipped to 0 is credited
        state = self.watcher.tick(Sample("Safari", "Long read", "", 200), self.at(30))
        self.assertEqual(state, "idle")
        total = self.app.store.subjects("2026-09-30")[0]["seconds"]
        self.assertEqual(total, 15)

    def test_quiet_paused_and_gaps(self):
        self.assertEqual(self.watcher.tick(Sample("1Password", "vault", "", 0), self.at(0)), "quiet")
        self.app.store.put("sensor_paused_until", "forever")
        self.assertEqual(self.watcher.tick(Sample("Safari", "x", "", 0), self.at(15)), "paused")
        self.app.store.put("sensor_paused_until", None)
        self.watcher.tick(Sample("Safari", "x", "", 0), self.at(30))
        self.watcher.tick(Sample("Safari", "x", "", 0), self.at(30 + 3600))  # laptop slept
        self.assertEqual(self.app.store.subjects("2026-09-30")[0]["seconds"], 0)

    def test_day_rollover_opens_a_new_trace(self):
        late = datetime(2026, 9, 30, 23, 59, 50).astimezone()
        page = Sample("Safari", "Night reading", "", 0)
        self.watcher.tick(page, late)
        self.watcher.tick(page, late + timedelta(seconds=15))
        self.assertEqual(len(self.app.store.traces("2026-10-01")), 1)



class PacingTests(TempApp):
    def setUp(self):
        super().setUp()
        self.watcher = Watcher(self.app, sensor=FakeSensor())
        self.t0 = datetime(2026, 9, 30, 10, 0, 0).astimezone()

    def test_backs_off_while_attention_stays_put(self):
        page = Sample("Safari", "One long paper", "", 0)
        delays = []
        clock = self.t0
        for _ in range(10):
            self.watcher.tick(page, clock)
            delays.append(self.watcher.next_delay(battery=False))
            clock += timedelta(seconds=delays[-1])
        self.assertEqual(delays[0], 15)
        self.assertEqual(delays[5], 30)
        self.assertEqual(delays[-1], 60)
        # long gaps are still credited in full: 9 intervals of real time
        total = self.app.store.subjects("2026-09-30")[0]["seconds"]
        self.assertAlmostEqual(total, sum(delays[:-1]), delta=1)

    def test_a_switch_resets_the_pace(self):
        for i in range(9):
            self.watcher.tick(Sample("Safari", "Paper", "", 0), self.t0 + timedelta(seconds=15 * i))
        self.assertEqual(self.watcher.next_delay(battery=False), 60)
        self.watcher.tick(Sample("Mail", "Inbox", "", 0), self.t0 + timedelta(seconds=200))
        self.assertEqual(self.watcher.next_delay(battery=False), 15)

    def test_battery_and_slack_water(self):
        self.watcher.tick(Sample("Safari", "Paper", "", 0), self.t0)
        self.assertEqual(self.watcher.next_delay(battery=True), 20)
        self.watcher.tick(Sample("Safari", "Paper", "", 500), self.t0 + timedelta(seconds=20))
        self.assertEqual(self.watcher.last_state, "idle")
        self.assertEqual(self.watcher.next_delay(battery=False), 30)
        self.assertEqual(self.watcher.next_delay(battery=True), 45)
        self.app.store.put("sensor_paused_until", "forever")
        self.watcher.tick(Sample("Safari", "Paper", "", 0), self.t0 + timedelta(seconds=60))
        self.assertEqual(self.watcher.next_delay(battery=False), 60)

    def test_power_probe_never_raises(self):
        from unconscious.sense.power import on_battery

        self.assertIn(on_battery(), (True, False))


if __name__ == "__main__":
    unittest.main()
