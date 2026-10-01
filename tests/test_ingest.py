import json
import unittest

from helpers import TempApp

from unconscious import ingest
from unconscious.store import today


class IngestTests(TempApp):
    def test_jot_is_a_conscious_trace(self):
        ingest.jot(self.app, "Do people sign up for pricier plans late at night? mail me@x.com")
        [trace] = self.app.store.traces(today())
        self.assertEqual(trace["kind"], "jot")
        self.assertIn("[email]", trace["body"])
        with self.assertRaises(ingest.IngestError):
            ingest.jot(self.app, "   ")

    def test_plain_text_log_becomes_notes(self):
        text = (
            "Spent 20 minutes comparing SaaS pricing pages.\n"
            "Maybe there is a study here: cognitive load versus churn risk.\n"
            "Opened churn dashboard, enterprise tolerates complexity.\n"
        )
        result = ingest.feed_bytes(self.app, "daily_log.txt", text.encode())
        self.assertEqual(result["kind"], "notes")
        self.assertEqual(len(result["traces"]), 3)
        self.assertTrue(all(t["kind"] == "note" for t in self.app.store.traces(today())))

    def test_markdown_becomes_one_reading_with_its_abstract(self):
        doc = "# Pricing complexity and churn\n\nIntro words.\n\nAbstract: We study how plan names shape churn.\n" + "body " * 400
        result = ingest.feed_bytes(self.app, "paper.md", doc.encode())
        self.assertEqual(result["kind"], "reading")
        [trace] = self.app.store.traces(today())
        self.assertEqual(trace["subject"], "Pricing complexity and churn")
        self.assertTrue(trace["body"].startswith("We study how plan names"))

    def test_unsupported_and_pdf_without_extra(self):
        with self.assertRaises(ingest.IngestError):
            ingest.feed_bytes(self.app, "image.png", b"\x89PNG")

    def test_activitywatch_export_respects_afk_incognito_and_privacy(self):
        export = {"buckets": {
            "aw-watcher-window_mac": {"type": "currentwindow", "events": [
                {"timestamp": "2026-09-30T08:00:00+00:00", "duration": 600, "data": {"app": "Stata", "title": "churn.do"}},
                {"timestamp": "2026-09-30T08:10:00+00:00", "duration": 300, "data": {"app": "Google Chrome", "title": "covered by web"}},
                {"timestamp": "2026-09-30T09:00:00+00:00", "duration": 900, "data": {"app": "Slack", "title": "DM with Bob"}},
                {"timestamp": "2026-09-30T10:00:00+00:00", "duration": 900, "data": {"app": "Stata", "title": "while away"}},
            ]},
            "aw-watcher-web-chrome": {"type": "web.tab.current", "events": [
                {"timestamp": "2026-09-30T08:10:00+00:00", "duration": 300,
                 "data": {"url": "https://www.google.com/search?q=hazard+models", "title": "hazard models - Google Search"}},
                {"timestamp": "2026-09-30T08:20:00+00:00", "duration": 120, "data": {"url": "https://secret.example", "title": "x", "incognito": True}},
            ]},
            "aw-watcher-afk_mac": {"type": "afkstatus", "events": [
                {"timestamp": "2026-09-30T09:55:00+00:00", "duration": 1800, "data": {"status": "afk"}},
            ]},
        }}
        result = ingest.feed_bytes(self.app, "aw-export.json", json.dumps(export).encode())
        self.assertEqual(result["kind"], "activitywatch")
        rows = [t for day in result["days"] for t in self.app.store.traces(day)]
        labels = {r["subject"] for r in rows}
        self.assertIn("hazard models", labels)
        self.assertIn("Slack", labels)  # private app: time only
        self.assertNotIn("covered by web", labels)
        self.assertFalse(any("away" in r["subject"] for r in rows))
        self.assertFalse(any("secret" in r["url"] for r in rows))
        self.assertTrue(all(r["title"] == "" for r in rows if r["subject"] == "Slack"))


if __name__ == "__main__":
    unittest.main()
