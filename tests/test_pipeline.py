from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from du_research.ai_backend import AIResponse  # noqa: E402
from du_research.config import AppConfig  # noqa: E402
from du_research.models import DatasetCandidate, PaperCandidate  # noqa: E402
from du_research.pipeline import ResearchPipeline  # noqa: E402


class FakeLiteratureProvider:
    name = "fake_literature"

    def search(self, query: str, max_results: int) -> list[PaperCandidate]:
        return [
            PaperCandidate(
                source=self.name,
                title="Cognitive Load and SaaS Pricing Decisions",
                summary="Observational study of pricing comprehension, churn, and attention.",
                authors=["A. Researcher", "B. Analyst"],
                year=2024,
                url="https://example.com/paper-1",
                score=0.91,
            ),
            PaperCandidate(
                source=self.name,
                title="Behavioral Signals in Subscription Retention",
                summary="Usage-event data supports descriptive and regression analysis.",
                authors=["C. Scientist"],
                year=2023,
                url="https://example.com/paper-2",
                score=0.83,
            ),
        ][:max_results]


class FakeDatasetProvider:
    name = "fake_dataset"

    def search(self, query: str, max_results: int) -> list[DatasetCandidate]:
        return [
            DatasetCandidate(
                source=self.name,
                title="Subscription Pricing Survey Dataset",
                summary="Open responses and churn labels for SaaS accounts.",
                year=2024,
                url="https://example.com/dataset-1",
                access="open",
                file_count=2,
                formats=["csv"],
                score=0.88,
            )
        ][:max_results]


class FakeAutomationRunner:
    def __init__(self):
        self.calls = []

    def execute_task_pack(self, task_path, *, credential_lookup=None):
        self.calls.append(str(task_path))
        return {
            "runner": "fake",
            "ok": True,
            "status": "completed",
            "downloaded_files": [],
        }


class FakeFeasibilityBackend:
    def call(self, prompt: str, **kwargs) -> AIResponse:
        return AIResponse(
            text=json.dumps(
                {
                    "decision": "proceed",
                    "confidence": 70,
                    "novel_angle": "A bounded content-fingerprint test",
                    "recommended_methods": ["descriptive-statistics"],
                    "required_data": ["group", "value"],
                    "estimated_effort": "low",
                    "key_risks": [],
                    "reasoning": "The local data supports the proposed smoke test.",
                }
            )
        )


class PipelineTests(unittest.TestCase):
    def test_pipeline_runs_end_to_end_with_local_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = True
            pipeline = ResearchPipeline(
                config=config,
                literature_providers=[FakeLiteratureProvider()],
                dataset_providers=[FakeDatasetProvider()],
            )
            fixture = ROOT / "tests" / "fixtures" / "sample_data.csv"
            result = pipeline.run(
                idea_text="Cognitive load as a SaaS pricing axis",
                data_file=str(fixture),
                run_id="run_test_local_csv",
            )

            self.assertEqual(result["manifest"]["status"], "completed")
            self.assertTrue(result["analysis"]["analysis_executed"])
            self.assertGreater(result["review"]["overall_score"], 0)
            self.assertTrue((Path(tmpdir) / "runs" / "run_test_local_csv" / "learning_signal.json").exists())
            self.assertTrue((Path(tmpdir) / "learning" / "human_idea_model.json").exists())
            self.assertTrue((Path(tmpdir) / "runs" / "run_test_local_csv" / "04_analysis" / "reproducibility_check.json").exists())
            self.assertTrue((Path(tmpdir) / "runs" / "run_test_local_csv" / "04_analysis" / "data" / "processed" / "provenance.json").exists())
            self.assertTrue((Path(tmpdir) / "runs" / "run_test_local_csv" / "05_drafting" / "references.bib").exists())
            self.assertTrue((Path(tmpdir) / "runs" / "run_test_local_csv" / "05_drafting" / "manuscript.pdf").exists())
            self.assertTrue((Path(tmpdir) / "runs" / "run_test_local_csv" / "06_review" / "review_history.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "runs" / "run_test_local_csv" / "06_review" / "final_manuscript.pdf").exists())
            submission_path = Path(tmpdir) / "submissions" / "run_test_local_csv_submission.json"
            self.assertTrue(submission_path.exists())
            submission_payload = json.loads(submission_path.read_text(encoding="utf-8"))
            manuscript_pdf = submission_payload["artifacts"]["manuscript_pdf"].replace("\\", "/")
            self.assertTrue(manuscript_pdf.endswith("06_review/final_manuscript.pdf"))

    def test_dry_run_generates_plan_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            pipeline = ResearchPipeline(config=config)
            result = pipeline.run(
                idea_text="Behavioral drivers of churn in B2B SaaS",
                run_id="run_test_dry",
                dry_run=True,
            )

            self.assertEqual(result["manifest"]["status"], "completed")
            self.assertFalse(result["analysis"]["analysis_executed"])
            self.assertEqual(result["datasets"]["dataset_count"], 0)
            self.assertTrue((Path(tmpdir) / "runs" / "run_test_dry" / "01_literature" / "papers.json").exists())

    def test_resume_completed_run_returns_without_replaying_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            pipeline = ResearchPipeline(config=config)
            pipeline.run(idea_text="A bounded resume test", run_id="run_resume_complete", dry_run=True)
            trace = Path(tmpdir) / "runs" / "run_resume_complete" / "execution_trace.jsonl"
            before = trace.read_text(encoding="utf-8")

            result = pipeline.run(run_id="run_resume_complete", resume=True)

            self.assertTrue(result["reused_completed_run"])
            self.assertEqual(result["manifest"]["status"], "completed")
            self.assertEqual(trace.read_text(encoding="utf-8"), before)

    def test_resume_with_new_data_reuses_upstream_and_invalidates_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            config.submission.enabled = False
            pipeline = ResearchPipeline(config=config)
            pipeline.run(idea_text="Attach data after protocol", run_id="run_resume_data", dry_run=True)

            result = pipeline.run(
                run_id="run_resume_data",
                resume=True,
                data_file=str(ROOT / "tests" / "fixtures" / "sample_data.csv"),
            )

            self.assertTrue(result["analysis"]["analysis_executed"])
            trace = (Path(tmpdir) / "runs" / "run_resume_data" / "execution_trace.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn('"event": "reused"', trace)
            self.assertIn('"data_changed": true', trace)

    def test_resume_with_changed_data_content_invalidates_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            config.submission.enabled = False
            pipeline = ResearchPipeline(config=config)
            pipeline.backend = FakeFeasibilityBackend()
            data_path = Path(tmpdir) / "mutable.csv"
            data_path.write_text("group,value\na,1\nb,2\n", encoding="utf-8")
            pipeline.run(
                idea_text="Detect an in-place dataset update",
                run_id="run_resume_content",
                data_file=str(data_path),
                dry_run=True,
            )
            trace_path = Path(tmpdir) / "runs" / "run_resume_content" / "execution_trace.jsonl"
            prior_events = trace_path.read_text(encoding="utf-8").splitlines()
            prior_manifest = pipeline.status("run_resume_content")

            data_path.write_text("group,value\na,9\nb,8\n", encoding="utf-8")
            result = pipeline.run(run_id="run_resume_content", resume=True, dry_run=True)

            self.assertEqual(result["analysis"]["numeric_summary"]["value"]["mean"], 8.5)
            self.assertNotEqual(
                result["manifest"]["data_fingerprint"],
                prior_manifest["data_fingerprint"],
            )
            resumed_events = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()[len(prior_events):]
            ]
            self.assertTrue(any(event["stage"] == "literature" and event["event"] == "reused" for event in resumed_events))
            self.assertTrue(any(event["stage"] == "feasibility" and event["event"] == "reused" for event in resumed_events))
            self.assertTrue(any(event["stage"] == "data_sources" and event["event"] == "reused" for event in resumed_events))
            for stage_name in ("analysis", "drafting", "review"):
                self.assertTrue(
                    any(event["stage"] == stage_name and event["event"] == "completed" for event in resumed_events)
                )
                self.assertFalse(
                    any(event["stage"] == stage_name and event["event"] == "reused" for event in resumed_events)
                )

    def test_resume_old_manifest_uses_processed_data_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            config.submission.enabled = False
            pipeline = ResearchPipeline(config=config)
            data_path = Path(tmpdir) / "legacy.csv"
            data_path.write_text("group,value\na,1\nb,2\n", encoding="utf-8")
            pipeline.run(
                idea_text="Resume a pre-fingerprint manifest",
                run_id="run_resume_legacy",
                data_file=str(data_path),
                dry_run=True,
            )
            run_dir = Path(tmpdir) / "runs" / "run_resume_legacy"
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("data_fingerprint")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            trace_path = run_dir / "execution_trace.jsonl"
            before = trace_path.read_text(encoding="utf-8")

            result = pipeline.run(run_id="run_resume_legacy", resume=True, dry_run=True)

            self.assertTrue(result["reused_completed_run"])
            self.assertEqual(trace_path.read_text(encoding="utf-8"), before)

    def test_daily_capture_extracts_ideas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            pipeline = ResearchPipeline(config=config)
            fixture = ROOT / "tests" / "fixtures" / "daily_log.txt"
            result = pipeline.capture_daily(str(fixture))

            self.assertGreaterEqual(result["idea_count"], 1)
            self.assertTrue((Path(tmpdir) / "ideas" / "idea_backlog.jsonl").exists())
            self.assertTrue(any("pricing" in item["idea_text"].lower() for item in result["ideas"]))

    def test_export_computer_task_creates_handoff_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            pipeline = ResearchPipeline(
                config=config,
                literature_providers=[FakeLiteratureProvider()],
                dataset_providers=[FakeDatasetProvider()],
            )
            pipeline.run(
                idea_text="Cognitive load as a SaaS pricing axis",
                run_id="run_for_task_export",
                dry_run=False,
            )
            task_path = Path(pipeline.export_computer_task("run_for_task_export"))
            self.assertTrue(task_path.exists())
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            self.assertIn("flow", payload)
            self.assertGreater(len(payload["flow"]), 0)

    def test_research_can_resolve_backlog_idea_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            pipeline = ResearchPipeline(
                config=config,
                literature_providers=[FakeLiteratureProvider()],
                dataset_providers=[FakeDatasetProvider()],
            )
            backlog = Path(tmpdir) / "ideas" / "idea_backlog.jsonl"
            backlog.parent.mkdir(parents=True, exist_ok=True)
            backlog.write_text(
                '{"idea_id":"idea_backlog_001","title":"Backlog selected idea"}\n',
                encoding="utf-8",
            )
            result = pipeline.run(
                idea_id="idea_backlog_001",
                run_id="run_from_backlog",
                dry_run=False,
            )
            self.assertEqual(result["manifest"]["idea_id"], "idea_backlog_001")
            self.assertEqual(result["manifest"]["idea_text"], "Backlog selected idea")

    def test_export_computer_task_includes_credentials_and_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            pipeline = ResearchPipeline(
                config=config,
                literature_providers=[FakeLiteratureProvider()],
                dataset_providers=[FakeDatasetProvider()],
            )
            pipeline.set_credential(
                "example.com",
                "user@example.com",
                "secret",
                login_url="https://example.com/login",
                extra_fields={
                    "username_selector": "#username",
                    "password_selector": "#password",
                    "submit_selector": "button[type=submit]",
                },
            )
            pipeline.run(
                idea_text="Cognitive load as a SaaS pricing axis",
                run_id="run_for_credential_task_export",
                dry_run=False,
            )
            task_path = Path(pipeline.export_computer_task("run_for_credential_task_export"))
            payload = json.loads(task_path.read_text(encoding="utf-8"))
            self.assertIn("example.com", payload["credential_resources"])
            actions = [step["action"] for step in payload["flow"]]
            self.assertIn("type_css", actions)
            self.assertIn("open", actions)

    def test_pipeline_can_auto_execute_automation_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.pipeline.auto_learn = False
            config.automation.auto_execute = True
            runner = FakeAutomationRunner()
            pipeline = ResearchPipeline(
                config=config,
                literature_providers=[FakeLiteratureProvider()],
                dataset_providers=[FakeDatasetProvider()],
                automation_runner=runner,
            )
            result = pipeline.run(
                idea_text="Cognitive load as a SaaS pricing axis",
                run_id="run_with_auto_automation",
                dry_run=False,
            )
            self.assertEqual(len(runner.calls), 1)
            self.assertTrue(result["automation"]["executed"])
            self.assertTrue((Path(tmpdir) / "runs" / "run_with_auto_automation" / "automation_result.json").exists())


if __name__ == "__main__":
    unittest.main()
