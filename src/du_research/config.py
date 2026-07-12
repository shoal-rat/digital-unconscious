from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineSection:
    workspace_dir: str = "workspace"
    quality_threshold: int = 78
    auto_learn: bool = False
    network_timeout_seconds: int = 15
    max_revisions: int = 3


@dataclass
class AISection:
    """Configuration for AI backends."""
    mode: str = "auto"
    default_model: str = "sonnet"
    # Workload-first defaults. Missing optional keys fall through to the local
    # Codex and Claude subscription CLIs, so a keyless install is fully usable.
    creative_model: str = "codex:default"
    judge_model: str = "claude_code:sonnet"
    compressor_model: str = "deepseek:deepseek-v4-flash"
    briefing_model: str = "deepseek:deepseek-v4-flash"
    writer_model: str = "claude_code:opus"
    reviewer_model: str = "codex:default"
    revision_model: str = "claude_code:sonnet"
    analysis_model: str = "glm:glm-5.1"
    evidence_model: str = "deepseek:deepseek-v4-flash"
    ideation_model: str = "codex:default"
    ideation_review_model: str = "claude_code:sonnet"
    api_key: str = ""
    openai_api_key: str = ""
    kimi_api_key: str = ""
    deepseek_api_key: str = ""
    glm_api_key: str = ""
    openai_default_model: str = "gpt-5.6-sol"
    kimi_default_model: str = "kimi-k2.6"
    deepseek_default_model: str = "deepseek-v4-flash"
    glm_default_model: str = "glm-5.1"
    fallback: bool = True
    fallback_order: list[str] = field(
        default_factory=lambda: [
            "deepseek", "glm", "codex", "claude_code", "openai", "anthropic", "kimi"
        ]
    )
    # Extended-thinking token budgets (0 disables). Wired to the two
    # reasoning-heavy steps and translated per provider: an Anthropic budget,
    # OpenAI reasoning effort, the Kimi thinking toggle, or a Claude Code keyword.
    think_idea_budget: int = 0
    think_judge_budget: int = 0


@dataclass
class ObservationSection:
    """Screen observation layer settings."""
    enabled: bool = True
    source: str = "auto"  # "auto" | "screenpipe" | "vision" | "file"
    vision_model: str = "codex:default"  # local Codex supports image attachments
    vision_max_dimension: int = 1568  # downscale long edge before sending to the model
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
    focus_fields: list[str] = field(default_factory=list)  # e.g. ["economics research", "management"]
    web_search: bool = True  # let idea generation look up unfamiliar/trending topics online
    max_ideas_per_cycle: int = 8
    include_threshold: int = 75
    hold_threshold: int = 60
    max_briefing_ideas: int = 5
    # Exploration is explicit. A daily scan must never silently grow into a
    # browser/download/drafting run.
    auto_research_enabled: bool = False
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
class IdeationSection:
    """Local paper/data-to-study-card workflow settings."""
    max_ideas: int = 5
    max_source_characters: int = 24000
    max_profile_rows: int = 5000


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
    runner: str = "claude_code"  # "claude_code" | "codex" | "selenium"
    checkpoint_policy: str = "best_effort"  # "best_effort" | "strict"
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
    # Bounded long-term stores (enforced on write and by maintenance) so months
    # of daily use cannot grow these files without limit.
    rag_max_documents: int = 2000        # cap RAG knowledge store size
    idea_backlog_max: int = 500          # cap idea-backlog entries
    ideation_sessions_max: int = 100     # keep newest portable Idea Lab sessions
    domain_knowledge_history: int = 5    # shallow domain-knowledge snapshots kept
    prompt_versions_kept: int = 10       # prompt-evolution version files per agent
    evolution_log_max_lines: int = 500   # evolution_log.jsonl lines per agent


@dataclass
class AppConfig:
    pipeline: PipelineSection = field(default_factory=PipelineSection)
    ai: AISection = field(default_factory=AISection)
    observation: ObservationSection = field(default_factory=ObservationSection)
    idea: IdeaSection = field(default_factory=IdeaSection)
    literature: LiteratureSection = field(default_factory=LiteratureSection)
    datasets: DatasetsSection = field(default_factory=DatasetsSection)
    ideation: IdeationSection = field(default_factory=IdeationSection)
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
        _apply(config.ideation, parsed.get("ideation", {}))
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
