from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass
class PipelineSection:
    workspace_dir: str = "workspace"
    quality_threshold: int = 78
    auto_learn: bool = True
    network_timeout_seconds: int = 15
    max_revisions: int = 3


@dataclass
class AISection:
    """Configuration for the AI backend (dual-mode)."""
    mode: str = "auto"  # "auto" | "claude_code" | "api"
    default_model: str = "sonnet"
    creative_model: str = "opus"
    judge_model: str = "sonnet"
    compressor_model: str = "haiku"
    briefing_model: str = "opus"
    writer_model: str = "opus"
    reviewer_model: str = "sonnet"
    revision_model: str = "sonnet"
    analysis_model: str = "sonnet"
    api_key: str = ""  # only for API mode; prefer env var


@dataclass
class ObservationSection:
    """Screen observation layer settings."""
    enabled: bool = True
    screenpipe_url: str = "http://localhost:3030"
    window_minutes: int = 30
    lookback_multiplier: int = 4
    fallback_log_path: str = ""  # path to manual daily log file
    blacklist_apps: list[str] = field(default_factory=list)
    service_interval_minutes: int = 60
    recent_frame_hash_limit: int = 5000


@dataclass
class IdeaSection:
    """Idea generation and judging settings."""
    primary_domains: list[str] = field(default_factory=lambda: ["AI tools", "product design"])
    secondary_domains: list[str] = field(default_factory=lambda: ["cognitive science", "business models"])
    max_ideas_per_cycle: int = 8
    include_threshold: int = 75
    hold_threshold: int = 60
    max_briefing_ideas: int = 5
    auto_research_enabled: bool = True
    auto_research_top_k: int = 1
    auto_research_dedupe_enabled: bool = True
    auto_research_similarity_threshold: float = 0.9
    auto_research_cooldown_days: int = 14


@dataclass
class LiteratureSection:
    max_results_per_source: int = 6
    core_papers: int = 5
    download_pdfs: bool = True
    max_pdf_downloads: int = 3


@dataclass
class DatasetsSection:
    max_results_per_source: int = 5


@dataclass
class AnalysisSection:
    max_categorical_values: int = 6
    max_numeric_columns: int = 4
    enable_ai_codegen: bool = True
    max_codegen_retries: int = 5
    timeout_seconds: int = 120
    figure_dpi: int = 300


@dataclass
class PaperSection:
    target_venue: str = "Research dossier"
    final_submission_requires_approval: bool = True


@dataclass
class LearningSection:
    min_runs_before_update: int = 1
    prompt_evolution: bool = True
    human_idea_model: bool = True
    min_runs_before_evolution: int = 3


@dataclass
class DailySection:
    max_ideas: int = 10
    min_idea_score: float = 0.35
    briefing_time: str = "22:00"


@dataclass
class CircuitBreakerSection:
    max_retries: int = 3
    initial_wait: float = 2.0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0


@dataclass
class CredentialsSection:
    vault_path: str = "workspace/credentials/credentials.enc"
    key_path: str = "workspace/credentials/master.key"


@dataclass
class AutomationSection:
    enabled: bool = True
    auto_execute: bool = False
    runner: str = "claude_code"  # "claude_code" | "selenium"
    browser: str = "chrome"
    download_dir: str = "workspace/browser_downloads"
    screenshot_dir: str = "workspace/browser_screenshots"
    headless: bool = True
    timeout_seconds: int = 60
    institutional_proxy_url: str = ""


@dataclass
class SubmissionSection:
    enabled: bool = True
    approvals_path: str = "workspace/submissions"
    pending_timeout_hours: int = 72


@dataclass
class ServiceSection:
    pid_path: str = "workspace/service/daemon.json"
    log_path: str = "workspace/service/daemon.log"
    status_path: str = "workspace/service/status.json"
    run_history_limit: int = 200
    gc_every_cycles: int = 24
    maintenance_every_cycles: int = 24


@dataclass
class RetentionSection:
    observation_days: int = 30
    daily_cycle_days: int = 180
    browser_artifact_days: int = 30
    service_log_max_mb: int = 50


@dataclass
class AppConfig:
    pipeline: PipelineSection = field(default_factory=PipelineSection)
    ai: AISection = field(default_factory=AISection)
    observation: ObservationSection = field(default_factory=ObservationSection)
    idea: IdeaSection = field(default_factory=IdeaSection)
    literature: LiteratureSection = field(default_factory=LiteratureSection)
    datasets: DatasetsSection = field(default_factory=DatasetsSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    paper: PaperSection = field(default_factory=PaperSection)
    learning: LearningSection = field(default_factory=LearningSection)
    daily: DailySection = field(default_factory=DailySection)
    circuit_breaker: CircuitBreakerSection = field(default_factory=CircuitBreakerSection)
    credentials: CredentialsSection = field(default_factory=CredentialsSection)
    automation: AutomationSection = field(default_factory=AutomationSection)
    submission: SubmissionSection = field(default_factory=SubmissionSection)
    service: ServiceSection = field(default_factory=ServiceSection)
    retention: RetentionSection = field(default_factory=RetentionSection)
    config_path: Path | None = None


def _apply(section: object, values: dict[str, object]) -> None:
    for key, value in values.items():
        if hasattr(section, key):
            setattr(section, key, value)


def load_config(path: str | Path | None = None) -> AppConfig:
    config = AppConfig()
    target = Path(path) if path else Path("config/pipeline.toml")
    if target.exists():
        parsed = tomllib.loads(target.read_text(encoding="utf-8"))
        _apply(config.pipeline, parsed.get("pipeline", {}))
        _apply(config.ai, parsed.get("ai", {}))
        _apply(config.observation, parsed.get("observation", {}))
        _apply(config.idea, parsed.get("idea", {}))
        _apply(config.literature, parsed.get("literature", {}))
        _apply(config.datasets, parsed.get("datasets", {}))
        _apply(config.analysis, parsed.get("analysis", {}))
        _apply(config.paper, parsed.get("paper", {}))
        _apply(config.learning, parsed.get("learning", {}))
        _apply(config.daily, parsed.get("daily", {}))
        _apply(config.circuit_breaker, parsed.get("circuit_breaker", {}))
        _apply(config.credentials, parsed.get("credentials", {}))
        _apply(config.automation, parsed.get("automation", {}))
        _apply(config.submission, parsed.get("submission", {}))
        _apply(config.service, parsed.get("service", {}))
        _apply(config.retention, parsed.get("retention", {}))
        config.config_path = target
    return config
