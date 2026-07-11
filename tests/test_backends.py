from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import TestCase, mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from du_research.ai_backend import ClaudeCodeBackend, CodexCLIBackend  # noqa: E402


class LocalBackendTests(TestCase):
    def test_codex_is_ephemeral_read_only_and_supports_schema(self) -> None:
        observed: dict = {}

        def run(cmd, **kwargs):
            observed.update({"cmd": cmd, **kwargs})
            output = Path(cmd[cmd.index("--output-last-message") + 1])
            output.write_text('{"status":"ready"}', encoding="utf-8")
            stdout = json.dumps({"type": "thread.started", "thread_id": "thread-1"})
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
        }
        with mock.patch("du_research.backends.local.shutil.which", return_value="/bin/codex"), mock.patch(
            "du_research.backends.local.subprocess.run", side_effect=run
        ):
            response = CodexCLIBackend().call("status", json_schema=schema)

        self.assertTrue(response.ok)
        self.assertEqual(response.structured, {"status": "ready"})
        self.assertEqual(response.session_id, "thread-1")
        self.assertIn("--ephemeral", observed["cmd"])
        self.assertEqual(observed["cmd"][observed["cmd"].index("--sandbox") + 1], "read-only")
        self.assertEqual(observed["cmd"][-1], "-")
        self.assertFalse(Path(observed["cwd"]).exists())

    def test_claude_uses_safe_mode_and_no_tools_for_text(self) -> None:
        observed: dict = {}

        def run(cmd, **kwargs):
            observed.update({"cmd": cmd, **kwargs})
            payload = {
                "result": "ready",
                "session_id": "session-1",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            }
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

        with mock.patch("du_research.backends.local.shutil.which", return_value="/bin/claude"), mock.patch(
            "du_research.backends.local.subprocess.run", side_effect=run
        ):
            response = ClaudeCodeBackend().call("status", mode="deterministic")

        self.assertTrue(response.ok)
        self.assertEqual(response.text, "ready")
        self.assertIn("--safe-mode", observed["cmd"])
        self.assertEqual(observed["cmd"][observed["cmd"].index("--permission-mode") + 1], "dontAsk")
        self.assertEqual(observed["cmd"][observed["cmd"].index("--tools") + 1], "")
        self.assertFalse(Path(observed["cwd"]).exists())

    def test_claude_exposes_only_read_for_an_image(self) -> None:
        observed: dict = {}

        def run(cmd, **kwargs):
            observed.update({"cmd": cmd, **kwargs})
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": "seen"}), stderr="")

        png = b"\x89PNG\r\n\x1a\n" + b"0" * 16
        with mock.patch("du_research.backends.local.shutil.which", return_value="/bin/claude"), mock.patch(
            "du_research.backends.local.subprocess.run", side_effect=run
        ):
            response = ClaudeCodeBackend().call("inspect", images=[png])

        self.assertTrue(response.ok)
        self.assertEqual(observed["cmd"][observed["cmd"].index("--tools") + 1], "Read")
        self.assertIn("input-1.png", observed["input"])
