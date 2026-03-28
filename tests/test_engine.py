"""Tests for the Digital Unconscious Engine and all new v0.2 components."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Any
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from du_research.ai_backend import AIResponse, ClaudeCodeBackend, AnthropicAPIBackend, create_backend
from du_research.circuit_breaker import CircuitBreaker, CircuitState
from du_research.observation import (
    BehaviorFrame,
    FileObserver,
    deduplicate_frames,
    group_into_windows,
)
from du_research.agents.compressor import CompressionAgent, _heuristic_compress
from du_research.agents.idea_generator import IdeaGeneratorAgent, _parse_ideas
from du_research.agents.judge import JudgeAgent, _heuristic_evaluate, WEIGHTS
from du_research.agents.briefing import BriefingAgent, _heuristic_briefing
from du_research.agents.learning_engine import (
    analyze_run_outcomes,
    build_human_idea_model,
    load_active_prompts,
    run_full_learning_cycle,
    save_learning_artifacts,
)
from du_research.config import AppConfig, load_config
from du_research.maintenance import WorkspaceMaintenance
from du_research.onboarding import apply_user_settings, ensure_first_run_setup
from du_research.service_manager import ServiceManager


# ---------------------------------------------------------------------------
# Fake backend for testing
# ---------------------------------------------------------------------------


class FakeBackend:
    """A configurable fake AIBackend for deterministic testing."""

    def __init__(self, responses: dict[str, str] | None = None, default: str = ""):
        self._responses = responses or {}
        self._default = default
        self.calls: list[dict[str, Any]] = []

    def call(self, prompt: str, **kwargs: Any) -> AIResponse:
        self.calls.append({"prompt": prompt[:200], **kwargs})
        # Match on any keyword in the response map
        for key, value in self._responses.items():
            if key.lower() in prompt.lower():
                return AIResponse(text=value, model="fake")
        return AIResponse(text=self._default, model="fake")


# ---------------------------------------------------------------------------
# AIBackend tests
# ---------------------------------------------------------------------------


class AIBackendTests(unittest.TestCase):
    def test_create_backend_auto_picks_claude_code_without_env(self) -> None:
        import os
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            backend = create_backend("auto")
            self.assertIsInstance(backend, ClaudeCodeBackend)
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_ai_response_ok_property(self) -> None:
        self.assertTrue(AIResponse(text="hello").ok)
        self.assertFalse(AIResponse(text="").ok)


# ---------------------------------------------------------------------------
# Circuit Breaker tests
# ---------------------------------------------------------------------------


class CircuitBreakerTests(unittest.TestCase):
    def test_successful_call_stays_closed(self) -> None:
        fake = FakeBackend(default='{"result": "ok"}')
        cb = CircuitBreaker(backend=fake, max_retries=2)
        response = cb.call("test prompt")
        self.assertTrue(response.ok)
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_repeated_failures_trip_breaker(self) -> None:
        fake = FakeBackend(default="")  # empty = failure
        cb = CircuitBreaker(
            backend=fake,
            max_retries=1,
            failure_threshold=2,
            initial_wait=0.01,
        )
        cb.call("test 1")
        cb.call("test 2")
        self.assertEqual(cb.state, CircuitState.OPEN)

    def test_open_breaker_rejects_calls(self) -> None:
        fake = FakeBackend(default="")
        cb = CircuitBreaker(
            backend=fake,
            max_retries=1,
            failure_threshold=1,
            recovery_timeout=999,
            initial_wait=0.01,
        )
        cb.call("trip")
        response = cb.call("should be rejected")
        self.assertFalse(response.ok)
        self.assertIn("circuit_breaker_open", str(response.raw))

    def test_stats_tracking(self) -> None:
        fake = FakeBackend(default="ok")
        cb = CircuitBreaker(backend=fake, max_retries=1)
        cb.call("test")
        stats = cb.stats
        self.assertEqual(stats["total_calls"], 1)
        self.assertEqual(stats["total_failures"], 0)


# ---------------------------------------------------------------------------
# Observation tests
# ---------------------------------------------------------------------------


class ObservationTests(unittest.TestCase):
    def test_file_observer_reads_text(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("I was browsing Notion pricing pages for 8 minutes\n")
            f.write("Then I searched for SaaS subscription model comparison\n")
            f.write("Short\n")  # too short, should be filtered
            f.name
        observer = FileObserver()
        frames = observer.read(Path(f.name))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].app_name, "text_log")
        Path(f.name).unlink()

    def test_file_observer_reads_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"app_name": "chrome", "title": "Pricing Page", "text": "SaaS pricing models comparison research"}) + "\n")
            f.write(json.dumps({"app_name": "vscode", "title": "main.py", "text": "def run_pipeline(): pass and more code here"}) + "\n")
        observer = FileObserver()
        frames = observer.read(Path(f.name))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].app_name, "chrome")
        Path(f.name).unlink()

    def test_deduplicate_merges_consecutive_duplicates(self) -> None:
        frames = [
            BehaviorFrame("2026-01-01T10:00:00Z", "chrome", "Page A", "content", 10),
            BehaviorFrame("2026-01-01T10:00:05Z", "chrome", "Page A", "content", 10),
            BehaviorFrame("2026-01-01T10:01:00Z", "chrome", "Page B", "different", 5),
        ]
        result = deduplicate_frames(frames)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].dwell_seconds, 20)  # merged

    def test_group_into_windows(self) -> None:
        frames = [
            BehaviorFrame("2026-01-01T10:00:00Z", "a", "t", "c", 5),
            BehaviorFrame("2026-01-01T10:10:00Z", "a", "t", "c", 5),
            BehaviorFrame("2026-01-01T11:00:00Z", "a", "t", "c", 5),  # new window
        ]
        windows = group_into_windows(frames, window_minutes=30)
        self.assertEqual(len(windows), 2)


# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------


class CompressionTests(unittest.TestCase):
    def test_heuristic_compress_works(self) -> None:
        frames = [
            BehaviorFrame("2026-01-01T10:00:00Z", "chrome", "Pricing", "SaaS pricing model analysis", 120),
            BehaviorFrame("2026-01-01T10:05:00Z", "notion", "Notes", "competitor research notes", 60),
        ]
        result = _heuristic_compress(frames)
        self.assertIn("dominant_topics", result)
        self.assertIn("app_distribution", result)
        self.assertIn("chrome", result["app_distribution"])

    def test_ai_compression_with_fake_backend(self) -> None:
        response_json = json.dumps({
            "time_range": "10:00-10:30",
            "dominant_topics": ["pricing", "SaaS"],
            "high_weight_content": [],
            "app_distribution": {"chrome": 0.7},
            "intent_signals": ["pricing research"],
            "cross_domain_hints": [],
            "search_queries": [],
        })
        fake = FakeBackend(default=response_json)
        agent = CompressionAgent(backend=fake, model="haiku")
        frames = [BehaviorFrame("2026-01-01T10:00:00Z", "chrome", "Page", "SaaS pricing", 30)]
        result = agent.compress(frames)
        self.assertEqual(result["dominant_topics"], ["pricing", "SaaS"])


# ---------------------------------------------------------------------------
# Idea Generator tests
# ---------------------------------------------------------------------------


class IdeaGeneratorTests(unittest.TestCase):
    def test_parse_ideas_from_json(self) -> None:
        raw = json.dumps({"ideas": [{"id": "1", "title": "Test idea"}]})
        ideas = _parse_ideas(raw)
        self.assertEqual(len(ideas), 1)
        self.assertEqual(ideas[0]["title"], "Test idea")

    def test_parse_ideas_from_array(self) -> None:
        raw = json.dumps([{"id": "1", "title": "Test idea"}])
        ideas = _parse_ideas(raw)
        self.assertEqual(len(ideas), 1)

    def test_generate_with_fake_backend(self) -> None:
        response = json.dumps({"ideas": [
            {"id": "idea_001", "title": "Cognitive load pricing", "description": "Study how...", "domains": ["product", "psychology"]},
            {"id": "idea_002", "title": "Screen time retention", "description": "Build a...", "domains": ["ai", "behavior"]},
        ]})
        fake = FakeBackend(default=response)
        agent = IdeaGeneratorAgent(backend=fake, model="opus")
        summary = {"dominant_topics": ["pricing", "SaaS"], "intent_signals": ["competitor analysis"]}
        ideas = agent.generate(summary)
        self.assertEqual(len(ideas), 2)
        self.assertEqual(ideas[0]["id"], "idea_001")


# ---------------------------------------------------------------------------
# Judge tests
# ---------------------------------------------------------------------------


class JudgeTests(unittest.TestCase):
    def test_heuristic_evaluate_produces_scores(self) -> None:
        ideas = [
            {"id": "1", "title": "Good idea", "description": "A detailed description of the idea with enough specificity", "data_hint": "survey data", "research_question": "How does X affect Y?", "domains": ["ai", "ux"]},
            {"id": "2", "title": "Vague", "description": "short", "domains": ["ai"]},
        ]
        evaluations = _heuristic_evaluate(ideas)
        self.assertEqual(len(evaluations), 2)
        # The more specific idea should score higher
        self.assertGreater(evaluations[0]["total_score"], evaluations[1]["total_score"])

    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0)

    def test_judge_with_fake_backend(self) -> None:
        response = json.dumps({"evaluations": [
            {"idea_id": "1", "novelty": 75, "feasibility": 70, "domain_relevance": 80, "timeliness": 65, "total_score": 72.5, "verdict": "hold", "one_line_reason": "Good but needs data"},
        ]})
        fake = FakeBackend(default=response)
        agent = JudgeAgent(backend=fake, model="sonnet")
        ideas = [{"id": "1", "title": "Test idea"}]
        evaluations = agent.evaluate(ideas)
        self.assertEqual(len(evaluations), 1)
        self.assertIn("verdict", evaluations[0])


# ---------------------------------------------------------------------------
# Briefing tests
# ---------------------------------------------------------------------------


class BriefingTests(unittest.TestCase):
    def test_heuristic_briefing_generates_markdown(self) -> None:
        summaries = [{"dominant_topics": ["pricing", "SaaS"], "intent_signals": ["competitor analysis"], "cross_domain_hints": []}]
        ideas = [{"title": "Cognitive load pricing", "total_score": 82, "description": "Study pricing psychology", "verdict": "include", "one_line_reason": "Novel angle"}]
        text = _heuristic_briefing("2026-03-28", summaries, ideas)
        self.assertIn("Daily Idea Briefing", text)
        self.assertIn("pricing", text.lower())
        self.assertIn("Cognitive load pricing", text)


# ---------------------------------------------------------------------------
# Learning Engine tests
# ---------------------------------------------------------------------------


class LearningEngineTests(unittest.TestCase):
    def _make_signal(self, domain: str = "product", score: float = 80, blockers: list[str] | None = None) -> dict[str, Any]:
        return {
            "run_id": "run_test",
            "domain": domain,
            "keywords": [{"keyword": "pricing", "weight": 0.3}, {"keyword": "saas", "weight": 0.2}],
            "literature": {"top_sources": ["crossref", "arxiv"]},
            "datasets": {"top_sources": ["zenodo"]},
            "review": {"overall_score": score, "critique_types": ["statistical_rigor"]},
            "blockers": blockers or [],
        }

    def test_analyze_run_outcomes(self) -> None:
        signals = [self._make_signal(), self._make_signal(score=85), self._make_signal(domain="ai")]
        result = analyze_run_outcomes(signals)
        self.assertEqual(result["run_count"], 3)
        self.assertGreater(result["average_quality"], 0)
        self.assertTrue(len(result["top_domains"]) > 0)

    def test_build_human_idea_model(self) -> None:
        signals = [self._make_signal(), self._make_signal(score=90)]
        model = build_human_idea_model(signals)
        self.assertIn("model_version", model)
        self.assertIn("core_obsessions", model)
        self.assertEqual(model["model_version"], 1)

    def test_save_learning_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            outcomes = analyze_run_outcomes([self._make_signal()])
            model = build_human_idea_model([self._make_signal()])
            paths = save_learning_artifacts(workspace, outcomes, model)
            self.assertTrue(Path(paths["model_path"]).exists())
            self.assertTrue(Path(paths["outcomes_path"]).exists())
            self.assertTrue(Path(paths["changes_path"]).exists())

    def test_patterns_detect_review_bottleneck(self) -> None:
        signals = [
            self._make_signal(),
            self._make_signal(),
            self._make_signal(),
        ]
        # All have statistical_rigor critique
        result = analyze_run_outcomes(signals)
        pattern_types = [p["type"] for p in result["patterns"]]
        self.assertIn("review_bottleneck", pattern_types)

    def test_full_learning_cycle_writes_active_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            runs_dir = workspace / "runs"
            for index in range(2):
                run_dir = runs_dir / f"run_{index}"
                run_dir.mkdir(parents=True, exist_ok=True)
                signal = self._make_signal(score=82 + index)
                (run_dir / "learning_signal.json").write_text(
                    json.dumps(signal, ensure_ascii=False),
                    encoding="utf-8",
                )
            fake = FakeBackend(
                responses={
                    "prompt engineer": json.dumps(
                        {
                            "edit": "Add an explicit rule to penalize missing evidence.",
                            "location": "scoring rules",
                            "reasoning": "This targets the dominant review bottleneck.",
                        }
                    )
                },
                default='{"edit":"Add a conservative fallback.","location":"end","reasoning":"Stability"}',
            )
            result = run_full_learning_cycle(
                workspace,
                backend=fake,
                current_prompts={"judge": "Base judge prompt."},
                min_runs_before_evolution=1,
            )
            self.assertTrue(result["updated"])
            prompts = load_active_prompts(workspace)
            self.assertIn("judge", prompts)
            self.assertIn("Learned refinement", prompts["judge"])


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class ConfigTests(unittest.TestCase):
    def test_default_config_has_new_sections(self) -> None:
        config = AppConfig()
        self.assertEqual(config.ai.mode, "auto")
        self.assertEqual(config.ai.creative_model, "opus")
        self.assertTrue(config.observation.enabled)
        self.assertEqual(config.idea.include_threshold, 75)
        self.assertEqual(config.circuit_breaker.max_retries, 3)

    def test_load_config_from_file(self) -> None:
        config = load_config("config/pipeline.toml")
        self.assertEqual(config.ai.mode, "auto")
        self.assertEqual(config.ai.compressor_model, "haiku")


class SetupTests(unittest.TestCase):
    def test_first_run_setup_persists_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            old_appdata = os.environ.get("APPDATA")
            os.environ["APPDATA"] = tmpdir
            try:
                state = ensure_first_run_setup(
                    config,
                    project_root=ROOT,
                    force=True,
                    interactive=False,
                )
            finally:
                if old_appdata is None:
                    os.environ.pop("APPDATA", None)
                else:
                    os.environ["APPDATA"] = old_appdata
            self.assertTrue(state["settings_path"])
            config_reloaded = AppConfig()
            config_reloaded.pipeline.workspace_dir = tmpdir
            settings = apply_user_settings(config_reloaded)
            self.assertTrue(settings)
            self.assertTrue(config_reloaded.automation.auto_execute)
            self.assertTrue(config_reloaded.idea.auto_research_enabled)


class ServiceAndMaintenanceTests(unittest.TestCase):
    def test_service_manager_start_background_writes_pid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            manager = ServiceManager(config=config, project_root=ROOT)

            class DummyProcess:
                pid = 4321

            with mock.patch("subprocess.Popen", return_value=DummyProcess()):
                result = manager.start_background(config_path=ROOT / "config" / "pipeline.toml")

            self.assertTrue(result["running"])
            self.assertEqual(result["pid"], 4321)
            self.assertTrue(Path(result["pid_path"]).exists())

    def test_workspace_maintenance_prunes_old_files_and_trims_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.retention.observation_days = 0
            config.retention.daily_cycle_days = 0
            config.retention.browser_artifact_days = 0
            config.retention.service_log_max_mb = 1
            workspace = Path(tmpdir)

            old_observation = workspace / "observation" / "2025-01-01_frames.jsonl"
            old_observation.parent.mkdir(parents=True, exist_ok=True)
            old_observation.write_text('{"x":1}\n', encoding="utf-8")

            old_cycle = workspace / "daily" / "cycle_2025-01-01"
            old_cycle.mkdir(parents=True, exist_ok=True)
            (old_cycle / "briefing.md").write_text("x", encoding="utf-8")

            screenshots = workspace / "browser_screenshots"
            screenshots.mkdir(parents=True, exist_ok=True)
            old_png = screenshots / "step.png"
            old_png.write_bytes(b"test")

            daemon_log = workspace / "service" / "daemon.log"
            daemon_log.parent.mkdir(parents=True, exist_ok=True)
            daemon_log.write_bytes(b"x" * (2 * 1024 * 1024))

            old_ts = datetime.fromisoformat("2025-01-01T00:00:00+00:00").timestamp()
            os.utime(old_observation, (old_ts, old_ts))
            os.utime(old_cycle, (old_ts, old_ts))
            os.utime(old_png, (old_ts, old_ts))

            maintenance = WorkspaceMaintenance(workspace, config)
            result = maintenance.run()

            self.assertGreaterEqual(result["removed_observation_files"], 1)
            self.assertGreaterEqual(result["removed_daily_cycles"], 1)
            self.assertGreaterEqual(result["removed_browser_artifacts"], 1)
            self.assertTrue(result["trimmed_service_log"])


# ---------------------------------------------------------------------------
# End-to-end engine test with fake backend
# ---------------------------------------------------------------------------


class EngineIntegrationTests(unittest.TestCase):
    def test_daily_cycle_with_file_fallback(self) -> None:
        """Run the full daily cycle using a log file and fake AI backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test log file
            log_file = Path(tmpdir) / "daily.txt"
            log_file.write_text(
                "Browsing Notion pricing pages for competitor analysis\n"
                "Searched for SaaS subscription model comparison freemium conversion rates\n"
                "Reading cognitive science paper about attention and decision fatigue\n"
                "Looking at Spotify Wrapped feature for engagement data patterns\n",
                encoding="utf-8",
            )

            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.observation.enabled = False  # skip screenpipe
            config.observation.fallback_log_path = str(log_file)
            config.idea.auto_research_enabled = False

            # Monkey-patch the engine to use a fake backend
            from du_research.engine import DigitalUnconsciousEngine
            engine = DigitalUnconsciousEngine(config)

            # Replace the circuit breaker's backend with a fake
            compress_response = json.dumps({
                "time_range": "10:00-10:30",
                "dominant_topics": ["pricing", "SaaS", "cognitive science"],
                "high_weight_content": [{"content": "Notion pricing", "dwell_time": "8min", "weight": 0.85}],
                "app_distribution": {"text_log": 1.0},
                "intent_signals": ["competitor research", "pricing strategy"],
                "cross_domain_hints": ["cognitive science + pricing"],
                "search_queries": ["SaaS subscription model comparison"],
            })
            idea_response = json.dumps({"ideas": [
                {"id": "idea_001", "title": "Cognitive load pricing model", "description": "Design pricing based on cognitive load reduction", "source_behaviour": "Browsing pricing pages", "domains": ["product", "cognitive science"], "research_question": "Does reducing cognitive load improve conversion?", "data_hint": "A/B test data", "novelty_signal": "No published study combining these"},
                {"id": "idea_002", "title": "Behavioural retention dashboard", "description": "Spotify Wrapped for SaaS retention", "source_behaviour": "Looking at Spotify Wrapped", "domains": ["product", "data viz"], "research_question": "Does showing usage patterns improve retention?", "data_hint": "Engagement metrics", "novelty_signal": "Novel in SaaS context"},
            ]})
            judge_response = json.dumps({"evaluations": [
                {"idea_id": "idea_001", "novelty": 82, "feasibility": 75, "domain_relevance": 85, "timeliness": 90, "total_score": 82.0, "verdict": "include", "one_line_reason": "Strong cross-domain connection"},
                {"idea_id": "idea_002", "novelty": 70, "feasibility": 80, "domain_relevance": 75, "timeliness": 85, "total_score": 77.0, "verdict": "include", "one_line_reason": "Practical and timely"},
            ]})
            briefing_md = "# Daily Idea Briefing — 2026-03-28\n\n## Today's Focus\nYou spent time on pricing research and cognitive science.\n"

            fake = FakeBackend(responses={
                "compress": compress_response,
                "generate": idea_response,
                "evaluate": judge_response,
                "briefing": briefing_md,
            }, default=compress_response)

            engine.backend = FakeCircuitBreaker(fake)
            engine.compressor.backend = engine.backend
            engine.idea_generator.backend = engine.backend
            engine.judge.backend = engine.backend
            engine.briefing_agent.backend = engine.backend

            result = engine.run_daily_cycle(
                log_file=str(log_file),
                date_str="2026-03-28",
            )

            self.assertGreater(result["frames_observed"], 0)
            self.assertGreater(result["ideas_generated"], 0)
            self.assertTrue(Path(result["briefing_path"]).exists())
            self.assertTrue(Path(result["output_dir"]).exists())

            # Verify artifacts were saved
            output_dir = Path(result["output_dir"])
            self.assertTrue((output_dir / "summaries.json").exists())
            self.assertTrue((output_dir / "ideas_all.json").exists())
            self.assertTrue((output_dir / "evaluations.json").exists())

    def test_daily_cycle_auto_promotes_top_idea_to_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "daily.txt"
            log_file.write_text(
                "Browsing pricing pages for SaaS churn and user cognition\n",
                encoding="utf-8",
            )

            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.observation.enabled = False
            config.observation.fallback_log_path = str(log_file)
            config.idea.auto_research_enabled = True
            config.idea.auto_research_top_k = 1

            from du_research.engine import DigitalUnconsciousEngine
            engine = DigitalUnconsciousEngine(config)

            compress_response = json.dumps({
                "time_range": "10:00-10:30",
                "dominant_topics": ["pricing", "churn"],
                "high_weight_content": [],
                "app_distribution": {"text_log": 1.0},
                "intent_signals": ["pricing research"],
                "cross_domain_hints": ["pricing + cognition"],
                "search_queries": ["pricing churn cognition"],
            })
            idea_response = json.dumps({"ideas": [
                {"id": "idea_001", "title": "Cognitive load pricing model", "description": "Design pricing based on cognitive load reduction", "source_behaviour": "Browsing pricing pages", "domains": ["product", "cognitive science"], "research_question": "Does reducing cognitive load improve conversion?", "data_hint": "A/B test data", "novelty_signal": "No published study combining these"},
            ]})
            judge_response = json.dumps({"evaluations": [
                {"idea_id": "idea_001", "novelty": 82, "feasibility": 75, "domain_relevance": 85, "timeliness": 90, "total_score": 82.0, "verdict": "include", "one_line_reason": "Strong cross-domain connection"},
            ]})
            briefing_md = "# Daily Idea Briefing — 2026-03-28\n"
            fake = FakeBackend(responses={
                "compress": compress_response,
                "generate": idea_response,
                "evaluate": judge_response,
                "briefing": briefing_md,
            }, default=compress_response)

            engine.backend = FakeCircuitBreaker(fake)
            engine.compressor.backend = engine.backend
            engine.idea_generator.backend = engine.backend
            engine.judge.backend = engine.backend
            engine.briefing_agent.backend = engine.backend

            class StubResearchPipeline:
                def run(self, **kwargs: Any) -> dict[str, Any]:
                    return {
                        "run_id": "auto_20260328_idea_001",
                        "run_dir": str(Path(tmpdir) / "runs" / "auto_20260328_idea_001"),
                        "review": {"overall_score": 88.0},
                    }

            engine.research_pipeline = StubResearchPipeline()
            result = engine.run_daily_cycle(date_str="2026-03-28")
            self.assertEqual(result["research_runs_started"], 1)
            self.assertEqual(result["research_runs"][0]["run_id"], "auto_20260328_idea_001")

    def test_run_observation_service_executes_iterations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            from du_research.engine import DigitalUnconsciousEngine
            engine = DigitalUnconsciousEngine(config)

            calls = []

            def fake_service_once(*, log_file: str | None = None, now: datetime | None = None) -> dict[str, Any]:
                calls.append((log_file, now))
                return {
                    "timestamp": "2026-03-28T10:00:00+01:00",
                    "ingest": {"new_frames": 3},
                    "briefing_generated": True,
                    "daily_result": {
                        "output_dir": str(Path(tmpdir) / "daily" / "cycle_2026-03-28"),
                    },
                }

            engine.run_service_once = fake_service_once  # type: ignore[assignment]
            result = engine.run_observation_service(interval_minutes=1, iterations=2, log_file="fallback.txt")
            self.assertEqual(result["completed_cycles"], 2)
            self.assertEqual(len(calls), 2)

    def test_service_once_generates_daily_briefing_only_once_per_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "daily.txt"
            log_file.write_text("Browsing pricing pages for SaaS churn and user cognition\n", encoding="utf-8")
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.observation.enabled = False
            config.observation.fallback_log_path = str(log_file)
            config.daily.briefing_time = "09:00"
            config.idea.auto_research_enabled = False

            from du_research.engine import DigitalUnconsciousEngine
            engine = DigitalUnconsciousEngine(config)

            compress_response = json.dumps({
                "time_range": "09:00-09:30",
                "dominant_topics": ["pricing", "churn"],
                "high_weight_content": [],
                "app_distribution": {"text_log": 1.0},
                "intent_signals": ["pricing research"],
                "cross_domain_hints": ["pricing + cognition"],
                "search_queries": ["pricing churn cognition"],
            })
            idea_response = json.dumps({"ideas": [
                {"id": "idea_001", "title": "Cognitive load pricing model", "description": "Design pricing based on cognitive load reduction", "domains": ["product", "cognitive science"]},
            ]})
            judge_response = json.dumps({"evaluations": [
                {"idea_id": "idea_001", "novelty": 82, "feasibility": 75, "domain_relevance": 85, "timeliness": 90, "total_score": 82.0, "verdict": "include", "one_line_reason": "Strong cross-domain connection"},
            ]})
            briefing_md = "# Daily Idea Briefing — 2026-03-28\n"
            fake = FakeBackend(responses={
                "compress": compress_response,
                "generate": idea_response,
                "evaluate": judge_response,
                "briefing": briefing_md,
            }, default=compress_response)
            engine.backend = FakeCircuitBreaker(fake)
            engine.compressor.backend = engine.backend
            engine.idea_generator.backend = engine.backend
            engine.judge.backend = engine.backend
            engine.briefing_agent.backend = engine.backend

            now = datetime.fromisoformat("2026-03-28T09:30:00+01:00")
            first = engine.run_service_once(now=now)
            second = engine.run_service_once(now=now)
            self.assertTrue(first["briefing_generated"])
            self.assertFalse(second["briefing_generated"])

    def test_daily_cycle_skips_duplicate_auto_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "daily.txt"
            log_file.write_text("Browsing pricing pages for SaaS churn and user cognition\n", encoding="utf-8")
            config = AppConfig()
            config.pipeline.workspace_dir = tmpdir
            config.observation.enabled = False
            config.observation.fallback_log_path = str(log_file)
            config.idea.auto_research_enabled = True
            config.idea.auto_research_dedupe_enabled = True

            from du_research.engine import DigitalUnconsciousEngine
            engine = DigitalUnconsciousEngine(config)

            compress_response = json.dumps({
                "time_range": "10:00-10:30",
                "dominant_topics": ["pricing", "churn"],
                "high_weight_content": [],
                "app_distribution": {"text_log": 1.0},
                "intent_signals": ["pricing research"],
                "cross_domain_hints": ["pricing + cognition"],
                "search_queries": ["pricing churn cognition"],
            })
            idea_response = json.dumps({"ideas": [
                {"id": "idea_001", "title": "Cognitive load pricing model", "description": "Design pricing based on cognitive load reduction", "domains": ["product", "cognitive science"]},
            ]})
            judge_response = json.dumps({"evaluations": [
                {"idea_id": "idea_001", "novelty": 82, "feasibility": 75, "domain_relevance": 85, "timeliness": 90, "total_score": 82.0, "verdict": "include", "one_line_reason": "Strong cross-domain connection"},
            ]})
            briefing_md = "# Daily Idea Briefing — 2026-03-28\n"
            fake = FakeBackend(responses={
                "compress": compress_response,
                "generate": idea_response,
                "evaluate": judge_response,
                "briefing": briefing_md,
            }, default=compress_response)
            engine.backend = FakeCircuitBreaker(fake)
            engine.compressor.backend = engine.backend
            engine.idea_generator.backend = engine.backend
            engine.judge.backend = engine.backend
            engine.briefing_agent.backend = engine.backend

            class DuplicateAwarePipeline:
                def find_recent_duplicate_research(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                    return {"run_id": "existing_run", "status": "completed", "similarity": 1.0}

                def run(self, **kwargs: Any) -> dict[str, Any]:
                    raise AssertionError("duplicate idea should not be re-run")

            engine.research_pipeline = DuplicateAwarePipeline()
            result = engine.run_daily_cycle(date_str="2026-03-28")
            self.assertEqual(result["research_runs_started"], 0)
            self.assertTrue(result["research_runs"][0]["skipped"])


class FakeCircuitBreaker:
    """Minimal wrapper matching CircuitBreaker interface for tests."""

    def __init__(self, backend: FakeBackend):
        self.backend = backend

    def call(self, prompt: str, **kwargs: Any) -> AIResponse:
        return self.backend.call(prompt, **kwargs)

    @property
    def stats(self) -> dict[str, Any]:
        return {"state": "closed", "total_calls": len(self.backend.calls)}


if __name__ == "__main__":
    unittest.main()
