# ruff: noqa: E402 -- tests add the local src directory before package imports.
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from du_research.ai_backend import AIResponse
from du_research.backlog import IdeaBacklog
from du_research.config import AppConfig
from du_research.dashboard import _inline_md, _same_local_origin
from du_research.ideation import ResearchIdeationLab
from du_research.models import PaperCandidate
from du_research.stages.literature import _download_open_pdfs


class IdeationBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, prompt: str, **kwargs) -> AIResponse:
        self.calls.append(prompt)
        if "Extract atomic evidence" in prompt:
            source_id = re.search(r"Source id: (src_[a-f0-9]+)", prompt).group(1)
            payload = {
                "evidence": [
                    {
                        "kind": "finding",
                        "statement": "Shorter forms improved completion.",
                        "anchor": "The study found that shorter forms improved completion rates.",
                        "locator": "Results",
                        "confidence": 0.92,
                    }
                ]
            }
            return AIResponse(text=json.dumps(payload), model="fake", structured=payload, raw={"source": source_id})
        if "Build research opportunities" in prompt:
            evidence_ids = list(dict.fromkeys(re.findall(r'"evidence_id": "(ev_[a-f0-9_]+)"', prompt)))
            payload = {
                "opportunities": [
                    {
                        "opportunity_id": "opp_001",
                        "type": "measurement_gap",
                        "title": "Measure friction directly",
                        "rationale": "The paper reports an outcome and the dataset exposes form structure.",
                        "evidence_ids": evidence_ids[:2],
                        "research_question": "Does form length causally affect completion?",
                    }
                ],
                "ideas": [
                    {
                        "idea_id": "idea_001",
                        "title": "A pre-registered form-friction boundary test",
                        "opportunity_id": "opp_001",
                        "research_question": "When does form length reduce completion?",
                        "hypothesis": "Longer forms reduce completion most for first-time users.",
                        "null_hypothesis": "Form length does not change completion by user tenure.",
                        "population": "Product users",
                        "context": "Online onboarding",
                        "exposure": "Form length",
                        "outcome": "Completion",
                        "unit_of_analysis": "Form attempt",
                        "study_design": "Pre-registered interaction analysis",
                        "operationalization": "Map field count and completion columns before testing.",
                        "required_data": ["field_count", "completed", "user_tenure"],
                        "confounders": ["device", "traffic source"],
                        "negative_control": "Account creation weekday",
                        "smallest_useful_test": "Confirm column coverage and estimate the interaction with robust errors.",
                        "falsifier": "No interaction of practical importance in the pre-specified interval.",
                        "supporting_evidence_ids": evidence_ids[:2],
                        "opposing_evidence_ids": [],
                        "contribution": "Identifies a boundary condition instead of repeating an average effect.",
                        "novelty_uncertainty": "Not checked outside the supplied source set.",
                        "status": "confirmatory",
                        "domains": ["human-computer interaction"],
                    }
                ],
            }
            return AIResponse(text=json.dumps(payload), model="fake", structured=payload)
        if "Adversarially review" in prompt:
            idea_id = re.search(r'"idea_id": "(idea_[a-f0-9]+)"', prompt).group(1)
            payload = {
                "reviews": [
                    {
                        "idea_id": idea_id,
                        "evidence_strength": 82,
                        "testability": 88,
                        "dataset_fit": 80,
                        "information_gain": 79,
                        "user_fit": 75,
                        "novelty_risk": "A broader literature search is still required.",
                        "fatal_flaw": "User tenure may be measured after exposure.",
                        "next_validation": "Verify temporal ordering.",
                    }
                ]
            }
            return AIResponse(text=json.dumps(payload), model="fake", structured=payload)
        return AIResponse(text="", raw={"error": "unexpected prompt"})


class IdeaBacklogTests(unittest.TestCase):
    def test_response_local_ids_do_not_drop_later_distinct_ideas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            backlog = IdeaBacklog(tmpdir, max_entries=20)
            first = backlog.append_unique([{"id": "idea_001", "title": "First question"}], date="2026-01-01")
            second = backlog.append_unique([{"id": "idea_001", "title": "Different question"}], date="2026-01-02")
            self.assertEqual((first, second), (1, 1))
            values = backlog.load()
            self.assertEqual(len(values), 2)
            self.assertNotEqual(values[0]["idea_id"], values[1]["idea_id"])
            self.assertEqual(values[1]["source_model_id"], "idea_001")


class ResearchIdeationTests(unittest.TestCase):
    def test_paper_and_dataset_become_grounded_ranked_study_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paper = root / "paper.md"
            paper.write_text(
                "# Results\n\nThe study found that shorter forms improved completion rates. "
                "A limitation is that only returning users were sampled.",
                encoding="utf-8",
            )
            dataset = root / "attempts.csv"
            dataset.write_text(
                "name,field_count,completed,user_tenure\n"
                "alice,3,1,10\n"
                "bob,8,0,1\n",
                encoding="utf-8",
            )
            config = AppConfig()
            config.pipeline.workspace_dir = str(root / "workspace")
            backend = IdeationBackend()
            result = ResearchIdeationLab(config, backend).run(
                paper_paths=[paper],
                data_paths=[dataset],
                context="HCI research with a small budget",
                max_ideas=3,
                session_id="session_test",
            )

            self.assertEqual(result["source_count"], 2)
            self.assertGreaterEqual(result["evidence_count"], 2)
            self.assertEqual(result["idea_count"], 1)
            idea = result["ideas"][0]
            evidence = json.loads((Path(result["session_dir"]) / "evidence_cards.json").read_text(encoding="utf-8"))
            evidence_ids = {card["evidence_id"] for card in evidence}
            self.assertTrue(set(idea["evidence_ids"]) <= evidence_ids)
            self.assertTrue(idea["hypothesis"])
            self.assertTrue(idea["null_hypothesis"])
            self.assertTrue(idea["falsifier"])
            self.assertEqual(idea["verdict"], "ready")
            self.assertTrue(idea["speculative"])
            self.assertTrue(Path(result["report_path"]).exists())
            prompts = "\n".join(backend.calls)
            self.assertNotIn("alice", prompts)
            self.assertNotIn(str(dataset), prompts)
            self.assertNotIn(dataset.name, prompts)
            backlog = IdeaBacklog(config.pipeline.workspace_dir).load()
            self.assertEqual(backlog[0]["origin"], "paper_ideation")
            self.assertEqual(backlog[0]["source_session"], "session_test")

    def test_dry_run_builds_portable_artifacts_without_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paper = root / "paper.txt"
            paper.write_text(
                "The experiment found a measurable effect on retention. "
                "However, the limited sample leaves an important boundary unanswered.",
                encoding="utf-8",
            )
            config = AppConfig()
            config.pipeline.workspace_dir = str(root / "workspace")
            result = ResearchIdeationLab(config, None).run(
                paper_paths=[paper],
                dry_run=True,
                session_id="session_dry",
            )
            self.assertGreaterEqual(result["evidence_count"], 1)
            self.assertGreaterEqual(result["idea_count"], 1)
            self.assertEqual(result["ideas_added_to_backlog"], 0)
            self.assertTrue((Path(result["session_dir"]) / "source_manifest.json").exists())

    def test_json_wrapper_with_papers_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "papers.json"
            source.write_text(
                json.dumps({"papers": [{"title": "One", "summary": "A sufficiently long finding sentence reports a robust association in one population."}]}),
                encoding="utf-8",
            )
            config = AppConfig()
            config.pipeline.workspace_dir = str(root / "workspace")
            result = ResearchIdeationLab(config, None).run(
                paper_paths=[source], dry_run=True, session_id="session_json"
            )
            self.assertEqual(result["source_count"], 1)


class SafetyRegressionTests(unittest.TestCase):
    def test_direct_download_rejects_html_saved_as_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paper = PaperCandidate(
                source="test",
                title="Access page",
                summary="",
                authors=[],
                year=2026,
                url="https://example.test/paper",
                pdf_url="https://example.test/paper.pdf",
            )
            with patch("du_research.stages.literature.fetch_bytes", return_value=b"<html>paywall</html>"):
                downloads, errors = _download_open_pdfs([paper], Path(tmpdir), timeout=1, max_pdf_downloads=1)
            self.assertEqual(downloads, [])
            self.assertTrue(errors)
            self.assertEqual(list((Path(tmpdir) / "pdfs").glob("*.pdf")), [])

    def test_markdown_renderer_escapes_model_html(self) -> None:
        rendered = _inline_md('<script>alert("x")</script> **safe**')
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("<strong>safe</strong>", rendered)

    def test_dashboard_requires_exact_local_origin_and_port(self) -> None:
        self.assertTrue(_same_local_origin("http://127.0.0.1:8765", "127.0.0.1:8765"))
        self.assertTrue(_same_local_origin("http://localhost:8765/lab", "localhost:8765"))
        self.assertFalse(_same_local_origin("http://localhost:9999", "localhost:8765"))
        self.assertFalse(_same_local_origin("http://example.test", "example.test"))
        self.assertFalse(_same_local_origin(None, "localhost:8765"))


if __name__ == "__main__":
    unittest.main()
