import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from unconscious.config import Settings
from unconscious.llm.base import STR, LLMRequest, LLMResult, check, extract_json, obj
from unconscious.llm.providers import ClaudeCodeCLI, CodexCLI
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
    return Router(settings, None, providers=providers)


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
        self.assertEqual(r.chain("dream")[0], ("claude", "opus"))
        self.assertEqual(r.chain("digest")[0], ("claude", "haiku"))
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
        result = ClaudeCodeCLI(timeout=30).complete(LLMRequest("ping", "sys", "hello", SCHEMA), "haiku")
        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"answer": "hello"})
        self.assertEqual((result.tokens_in, result.tokens_out), (7, 3))

    def test_claude_runner_surfaces_auth_errors(self):
        self.script("claude", "import json\nprint(json.dumps({'is_error': True, 'result': 'Failed to authenticate'}))\n")
        result = ClaudeCodeCLI(timeout=30).complete(LLMRequest("ping", "sys", "hello"), None)
        self.assertFalse(result.ok)
        self.assertIn("authenticate", result.error)

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
