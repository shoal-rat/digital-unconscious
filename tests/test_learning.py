from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from du_research.stages.learning import update_learning_model


class LearningTests(unittest.TestCase):
    def test_learning_aggregation_builds_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            for run_name, score, domain in [
                ("run_a", 72.5, "behavior"),
                ("run_b", 81.0, "product"),
            ]:
                run_dir = workspace / "runs" / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                signal = {
                    "run_id": run_name,
                    "domain": domain,
                    "keywords": [{"keyword": "pricing", "weight": 0.4}],
                    "literature": {"top_sources": ["fake_literature"]},
                    "datasets": {"top_sources": ["fake_dataset"]},
                    "analysis": {"analysis_executed": run_name == "run_b"},
                    "review": {"overall_score": score},
                    "blockers": ["no_local_data"] if run_name == "run_a" else [],
                }
                (run_dir / "learning_signal.json").write_text(
                    json.dumps(signal, indent=2),
                    encoding="utf-8",
                )

            model = update_learning_model(workspace, min_runs_before_update=1)

            self.assertIsNotNone(model)
            assert model is not None
            self.assertEqual(model["run_count"], 2)
            self.assertEqual(model["model_version"], 1)
            self.assertTrue((workspace / "learning" / "human_idea_model.json").exists())
            self.assertTrue((workspace / "learning" / "learning_changes.md").exists())


if __name__ == "__main__":
    unittest.main()
