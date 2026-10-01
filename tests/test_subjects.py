import unittest

from unconscious.config import Sense
from unconscious.sense.privacy import clean_url, redact
from unconscious.sense.subjects import clean_title, describe, host_matches, search_query

SENSE = Sense()
RULES = {"quiet_apps": SENSE.quiet_apps, "quiet_domains": SENSE.quiet_domains, "private_apps": SENSE.private_apps}


def subject(app, title, url=""):
    return describe(app, title, url, **RULES)


class SubjectTests(unittest.TestCase):
    def test_search_queries_become_search_subjects(self):
        s = subject("Google Chrome", "decoy pricing - Google Search", "https://www.google.com/search?q=decoy+pricing&oq=x")
        self.assertEqual((s.kind, s.label, s.key), ("search", "decoy pricing", "search:decoy pricing"))
        self.assertEqual(s.url, "https://www.google.com/search")  # the query string is never stored

    def test_scholarly_search_is_reading(self):
        s = subject("Safari", "x", "https://scholar.google.com/scholar?q=choice+overload")
        self.assertEqual((s.kind, s.category), ("search", "reading"))

    def test_search_from_title_when_url_is_unavailable(self):
        self.assertEqual(search_query("", "menu anchoring - Google Search"), ("menu anchoring", False))
        self.assertEqual(search_query("", "天气_百度搜索")[0], "天气")

    def test_private_windows_and_quiet_apps_are_never_recorded(self):
        self.assertIsNone(subject("Firefox", "Mozilla Firefox Private Browsing"))
        self.assertIsNone(subject("Google Chrome", "New Incognito Tab"))
        self.assertIsNone(subject("1Password 8", "Vault"))
        self.assertIsNone(subject("Google Chrome", "Accounts", "https://online.mybank.com/login"))

    def test_chat_apps_keep_only_time(self):
        s = subject("WeChat", "Conversation with Alice")
        self.assertEqual((s.label, s.title, s.category), ("WeChat", "", "chat"))
        gmail = subject("Google Chrome", "Inbox (3) - me@x.com - Gmail", "https://mail.google.com/mail/u/0/")
        self.assertEqual((gmail.label, gmail.title, gmail.url), ("mail.google.com", "", ""))

    def test_code_editors_group_by_project(self):
        a = subject("Visual Studio Code", "● engine.py — churn-panel — Visual Studio Code")
        b = subject("Visual Studio Code", "cohorts.py — churn-panel — Visual Studio Code")
        self.assertEqual(a.key, b.key)
        self.assertEqual(a.label, "churn-panel")

    def test_titles_are_cleaned(self):
        self.assertEqual(clean_title("(3) Why plans matter - YouTube - Google Chrome", "Google Chrome", "youtube.com"), "Why plans matter")
        self.assertEqual(clean_title("Report - Personal - Microsoft​ Edge", "Microsoft Edge"), "Report")
        self.assertEqual(clean_title("paper.pdf – Page 3 of 20", "Preview"), "paper.pdf")

    def test_host_matching_respects_label_boundaries(self):
        self.assertTrue(host_matches("x.com", "x.com"))
        self.assertTrue(host_matches("mobile.x.com", "x.com"))
        self.assertFalse(host_matches("dropbox.com", "x.com"))
        self.assertTrue(host_matches("scholar.google.co.uk", "scholar.google."))
        self.assertTrue(host_matches("pubmed.ncbi.nlm.nih.gov", "pubmed"))

    def test_redaction_and_url_cleaning(self):
        self.assertEqual(redact("mail me@site.org about 4111 1111 1111 1111"), "mail [email] about [number]")
        self.assertEqual(redact("ticket 12345678"), "ticket [number]")
        self.assertIn("[id]", redact("token ab12cd34ef56gh78ij90kl12mn"))
        self.assertEqual(clean_url("https://u:p@site.com/a/b?q=1#frag"), "https://site.com/a/b")
        self.assertEqual(clean_url("https://site.com/d/1a2b3c4d5e6f7g8h9i0j1k2l/edit"), "https://site.com/d/:id/edit")
        self.assertEqual(clean_url("file:///etc/passwd"), "")


if __name__ == "__main__":
    unittest.main()
