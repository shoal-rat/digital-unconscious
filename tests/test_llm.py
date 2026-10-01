import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from unconscious.config import Settings
from unconscious.llm.base import STR, LLMRequest, LLMResult, check, extract_json, obj
from unconscious.llm.providers import ClaudeCodeCLI, CodexCLI
from unconscious.llm.region import RegionCheck
from unconscious.llm.router import Router

SCHEMA = obj({"answer": STR})


class Scripted:
    def __init__(self, name, replies, available=True):
        self.name = name
        self.replies = list(replies)
        self.calls = []
        self._available = available

    def available(self):
        return self._available

    def complete(self, request, model):
        self.calls.append((request.prompt, model))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if reply is None:
            return LLMResult(False, provider=self.name, model=model or "", error="boom")
        return LLMResult(True, json.dumps(reply), reply, self.name, model or "m")


def router(settings, **providers):
    # tests never look anything up: the connection is "abroad" unless a test says otherwise
    return Router(settings, None, providers=providers, region=RegionCheck(fetch=lambda: "US"))


class JsonTests(unittest.TestCase):
    def test_extract_json_from_messy_replies(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('Sure!\n```json\n{"a": 2}\n```'), {"a": 2})
        self.assertEqual(extract_json('Here: {"a": 3} hope that helps'), {"a": 3})
        self.assertIsNone(extract_json("no json here"))

    def test_check_finds_schema_violations(self):
        schema = obj({"items": {"type": "array", "items": obj({"n": {"type": "integer"}}), "minItems": 1},
                      "kind": {"type": "string", "enum": ["a", "b"]}})
        self.assertEqual(check({"items": [{"n": 1}], "kind": "a"}, schema), [])
        problems = check({"items": [{"n": "x"}], "kind": "z"}, schema)
        self.assertTrue(any("expected integer" in p for p in problems))
        self.assertTrue(any("not in" in p for p in problems))
        self.assertTrue(check({"items": []}, schema))


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()

    def test_falls_back_to_the_next_available_provider(self):
        first = Scripted("claude", [None])
        second = Scripted("codex", [{"answer": "ok"}])
        result = router(self.settings, claude=first, codex=second).call(LLMRequest("dream", "s", "p", SCHEMA))
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "codex")

    def test_repairs_invalid_json_once(self):
        provider = Scripted("claude", [{"wrong": 1}, {"answer": "fixed"}])
        result = router(self.settings, claude=provider).call(LLMRequest("dream", "s", "p", SCHEMA))
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"answer": "fixed"})
        self.assertIn("did not match", provider.calls[1][0])

    def test_provider_crash_does_not_escape(self):
        provider = Scripted("claude", [RuntimeError("bad")])
        result = router(self.settings, claude=provider).call(LLMRequest("dream", "s", "p"))
        self.assertFalse(result.ok)
        self.assertIn("RuntimeError", result.error)

    def test_no_provider_explains_what_to_do(self):
        result = router(self.settings, claude=Scripted("claude", [], available=False)).call(LLMRequest("dream", "s", "p"))
        self.assertFalse(result.ok)
        self.assertIn("dun doctor", result.error)

    def test_role_preferences_and_independent_critic(self):
        r = router(self.settings, claude=Scripted("claude", []), codex=Scripted("codex", []))
        self.assertEqual(r.chain("dream")[0], ("claude", "claude-opus-5-5"))
        self.assertEqual(r.chain("digest")[0], ("claude", "claude-sonnet-5-5"))
        self.assertEqual(r.chain("critique")[0][0], "codex")  # not the dreamer

    def test_explicit_choice_without_fallback(self):
        self.settings.update({"models": {"dream": "codex:gpt-x", "fallback": False}})
        r = router(self.settings, claude=Scripted("claude", []), codex=Scripted("codex", []))
        self.assertEqual(r.chain("dream"), [("codex", "gpt-x")])


@unittest.skipIf(os.name == "nt", "stand-in executables are POSIX scripts")
class CliRunnerTests(unittest.TestCase):
    """Run the real runners against stand-in `claude` and `codex` executables."""

    def setUp(self):
        self.bin = Path(tempfile.mkdtemp())
        self._path = os.environ["PATH"]
        os.environ["PATH"] = f"{self.bin}{os.pathsep}{self._path}"

    def tearDown(self):
        os.environ["PATH"] = self._path

    def script(self, name, body):
        path = self.bin / name
        path.write_text("#!/usr/bin/env python3\n" + body)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)

    def test_claude_runner_reads_structured_output_and_usage(self):
        self.script("claude", (
            "import sys, json\n"
            "args = sys.argv[1:]\n"
            "assert '--tools' in args and args[args.index('--tools') + 1] == ''\n"
            "assert '--system-prompt' in args and '--safe-mode' in args\n"
            "prompt = sys.stdin.read()\n"
            "print(json.dumps({'result': 'done', 'structured_output': {'answer': prompt.strip()},"
            " 'usage': {'input_tokens': 7, 'output_tokens': 3}, 'total_cost_usd': 0.01}))\n"
        ))
        result = ClaudeCodeCLI(timeout=30).complete(LLMRequest("ping", "sys", "hello", SCHEMA), "claude-sonnet-5-5")
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"answer": "hello"})
        self.assertEqual((result.tokens_in, result.tokens_out), (7, 3))

    def test_claude_runner_surfaces_auth_errors(self):
        self.script("claude", "import json\nprint(json.dumps({'is_error': True, 'result': 'Failed to authenticate'}))\n")
        result = ClaudeCodeCLI(timeout=30).complete(LLMRequest("ping", "sys", "hello"), None)
        self.assertFalse(result.ok)
        self.assertIn("authenticate", result.error)

    def test_claude_research_runs_inside_the_sandbox_in_the_errands_folder(self):
        self.script("claude", (
            "import sys, json, os\n"
            "args = sys.argv[1:]\n"
            "sys.stdin.read()\n"
            "open(os.environ['ARGS_OUT'], 'w').write(json.dumps({'args': args, 'cwd': os.getcwd(),\n"
            "    'papers': sorted(os.listdir('papers'))}))\n"
            "print(json.dumps({'result': 'done', 'structured_output': {'answer': 'read'}}))\n"
        ))
        folder = Path(tempfile.mkdtemp())
        (folder / "papers").mkdir()
        (folder / "papers" / "01.pdf").write_bytes(b"%PDF-1.7")
        record = self.bin / "args.json"
        os.environ["ARGS_OUT"] = str(record)
        try:
            request = LLMRequest("dive", "sys", "hello", SCHEMA, research=True, workdir=folder)
            self.assertTrue(ClaudeCodeCLI(timeout=30).complete(request, "claude-sonnet-5-5").ok)
        finally:
            os.environ.pop("ARGS_OUT")
        seen = json.loads(record.read_text())
        args = seen["args"]
        self.assertEqual(Path(seen["cwd"]).resolve(), folder.resolve())
        self.assertEqual(seen["papers"], ["01.pdf"])
        self.assertIn("WebSearch", args[args.index("--tools") + 1])
        self.assertEqual(args[args.index("--permission-prompts") + 1], "none", "nobody is there to ask at night")
        settings = json.loads(args[args.index("--settings") + 1])
        self.assertTrue(settings["sandbox"]["enabled"])
        self.assertFalse(settings["sandbox"]["allowUnsandboxedCommands"])
        self.assertTrue(settings["sandbox"]["network"]["strictAllowlist"])
        self.assertTrue(settings["permissions"]["blockReadsOutsideWorkingDirectories"])
        fetches = [rule for rule in settings["permissions"]["allow"] if rule.startswith("WebFetch")]
        for host in ("arxiv.org", "reddit.com", "zhihu.com", "*.wikipedia.org", "douban.com"):
            self.assertIn(f"WebFetch(domain:{host})", fetches)
        from unconscious.llm.research import SHELF, off_shelf

        self.assertFalse([host for host in SHELF if off_shelf(host)], "no personal hosting on the shelf")
        self.assertFalse([d for d in settings["sandbox"]["network"]["allowedDomains"] if off_shelf(d.lstrip("*."))])
        self.assertNotIn("WebFetch", settings["permissions"]["allow"], "never every page on the web")
        self.assertTrue(all("domain:" in rule for rule in fetches))

    def test_without_research_claude_has_no_tools(self):
        self.script("claude", (
            "import sys, json\n"
            "args = sys.argv[1:]\n"
            "assert args[args.index('--tools') + 1] == ''\n"
            "assert '--settings' not in args\n"
            "sys.stdin.read()\n"
            "print(json.dumps({'result': 'plain', 'structured_output': {'answer': 'x'}}))\n"
        ))
        self.assertTrue(ClaudeCodeCLI(timeout=30).complete(LLMRequest("dream", "sys", "hi", SCHEMA), None).ok)

    def test_codex_runner_reads_the_last_message_file(self):
        self.script("codex", (
            "import sys\n"
            "args = sys.argv[1:]\n"
            "assert args[args.index('--sandbox') + 1] == 'read-only'\n"
            "out = args[args.index('--output-last-message') + 1]\n"
            "sys.stdin.read()\n"
            "open(out, 'w').write('{\"answer\": \"from codex\"}')\n"
        ))
        result = CodexCLI(timeout=30).complete(LLMRequest("ping", "sys", "hello", SCHEMA), None)
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"answer": "from codex"})


if __name__ == "__main__":
    unittest.main()
