import tempfile
import tomllib
import unittest
from pathlib import Path

from unconscious.config import Settings, dump_toml, load_settings


class ConfigTests(unittest.TestCase):
    def test_round_trip_through_toml(self):
        path = Path(tempfile.mkdtemp()) / "config.toml"
        settings = Settings(path=path)
        settings.update({
            "you": {"persona": 'Researcher "quoted"\nsecond line', "focus": "economics, pricing", "language": "zh"},
            "dream": {"time": "22:15", "sparks": 4},
            "providers": {"ollama": {"model": "qwen3:14b", "base_url": "http://localhost:11434/v1"}},
        })
        settings.save()
        loaded = load_settings(path)
        self.assertEqual(loaded.you.persona, 'Researcher "quoted"\nsecond line')
        self.assertEqual(loaded.you.focus, ["economics", "pricing"])
        self.assertEqual(loaded.dream.time, "22:15")
        self.assertEqual(loaded.dream.sparks, 4)
        self.assertEqual(loaded.providers["ollama"]["model"], "qwen3:14b")
        self.assertEqual(loaded.language, "zh")

    def test_values_are_coerced_and_clamped(self):
        settings = Settings()
        changed = settings.update({
            "sense": {"interval_seconds": "1", "capture_urls": "false", "quiet_apps": "A\nB, C"},
            "dream": {"time": "25:99", "sparks": 99, "candidates": 1},
            "ui": {"theme": "neon"},
            "unknown": {"x": 1},
        })
        self.assertIn("sense.interval_seconds", changed)
        self.assertEqual(settings.sense.interval_seconds, 5)
        self.assertFalse(settings.sense.capture_urls)
        self.assertEqual(settings.sense.quiet_apps, ["A", "B", "C"])
        self.assertEqual(settings.dream.time, "21:30")
        self.assertEqual(settings.dream.sparks, 6)
        self.assertGreaterEqual(settings.dream.candidates, settings.dream.sparks)
        self.assertEqual(settings.ui.theme, "system")

    def test_dump_produces_valid_toml(self):
        text = dump_toml(Settings().to_dict())
        data = tomllib.loads(text)
        self.assertIn("sense", data)
        self.assertIsInstance(data["sense"]["private_apps"], list)


if __name__ == "__main__":
    unittest.main()
