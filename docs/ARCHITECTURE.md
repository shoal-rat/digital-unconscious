# Architecture

## System Overview

Digital Unconscious is a multi-layer AI pipeline:

```
Observation → Compression → Idea Generation → Judging → Briefing → Auto-Research → Learning
```

## Module Map

### Core Infrastructure
- `du_research.ai_backend`: Dual-mode AI abstraction (Claude Code CLI / Anthropic API SDK)
- `du_research.circuit_breaker`: Three-state circuit breaker with exponential backoff
- `du_research.config`: TOML configuration loading with typed dataclass sections
- `du_research.engine`: Main orchestrator (daily cycle, learning cycle, observation service)
- `du_research.pipeline`: 6-stage research pipeline executor
- `du_research.credential_broker`: AES-GCM encrypted credential vault
- `du_research.maintenance`: Workspace cleanup and garbage collection
- `du_research.service_manager`: Background daemon lifecycle management
- `du_research.onboarding`: First-run setup wizard and autostart registration

### Agents (AI-powered with heuristic fallbacks)
- `du_research.agents.compressor`: Behaviour frame → structured summary (Haiku, strict)
- `du_research.agents.idea_generator`: Summary → creative ideas (Opus, creative temp 0.95)
- `du_research.agents.judge`: Ideas → scored evaluations (Sonnet, strict temp 0.1)
- `du_research.agents.briefing`: Daily Markdown briefing generator (Opus, balanced)
- `du_research.agents.writer`: Research manuscript drafting (Opus, temp 0.4)
- `du_research.agents.reviewer`: Adversarial paper reviewer (Sonnet, temp 0.05)
- `du_research.agents.revision`: Revision execution from reviewer feedback
- `du_research.agents.analysis_coder`: Python code generation for data analysis
- `du_research.agents.learning_engine`: Run outcomes, human model, prompt evolution

### Observation Layer
- `du_research.observation`: Screenpipe HTTP client, file fallback, deduplication, privacy filtering

### Research Pipeline Stages
- `du_research.stages.literature`: Open literature discovery (arXiv, PubMed, Semantic Scholar)
- `du_research.stages.feasibility`: Heuristic feasibility scoring
- `du_research.stages.data_sources`: Open dataset discovery (OSF, Zenodo, Kaggle)
- `du_research.stages.analysis`: Descriptive analysis and figure generation
- `du_research.stages.drafting`: Markdown manuscript generation
- `du_research.stages.review`: Rubric scoring and revision guidance

### Support
- `du_research.storage`: Run artifact persistence and trace logging
- `du_research.models`: Shared dataclasses (PaperCandidate, RunManifest, etc.)
- `du_research.net`: HTTP request utilities
- `du_research.utils`: Timestamps, keyword extraction, text utilities
- `du_research.cli`: Command-line interface

## AI Backend Architecture

```python
class AIBackend(Protocol):
    def call(self, prompt, mode, system, model, max_tokens, session_id) -> AIResponse

class ClaudeCodeBackend(AIBackend):
    # subprocess: claude --bare -p prompt --output-format json
    # Temperature simulated via --append-system-prompt mode hints

class AnthropicAPIBackend(AIBackend):
    # anthropic.Anthropic().messages.create()
    # Direct temperature control per call

class CircuitBreaker(AIBackend):
    # Wraps any backend with retry + backoff + circuit breaking
    # States: CLOSED → OPEN → HALF_OPEN
```

Mode aliases: `{"opus": "claude-opus-4-6", "sonnet": "claude-sonnet-4-6", "haiku": "claude-haiku-4-5"}`

Temperature-as-mode: `creative=0.95, balanced=0.7, strict=0.1, deterministic=0.0`

## Data Flow

### Daily Cycle
1. **Observe**: Screenpipe API or file fallback → BehaviorFrames
2. **Deduplicate**: Merge consecutive identical frames, accumulate dwell time
3. **Window**: Group into 30-minute time windows
4. **Compress**: Each window → structured JSON summary (Haiku)
5. **Generate**: Each summary → 5-8 ideas (Opus, with RAG + Human Model context)
6. **Judge**: All ideas → scored evaluations (Sonnet, with personalized rubrics)
7. **Brief**: Summaries + top ideas → daily Markdown briefing (Opus)
8. **Save**: All artifacts to `workspace/daily/cycle_YYYY-MM-DD/`
9. **Auto-Research**: Top included ideas → full research pipeline (optional)

### Learning Cycle
1. Load all `learning_signal.json` from completed research runs
2. **Run Outcome Analyzer**: Detect patterns (bottlenecks, effective sources, blockers)
3. **Human Idea Model Builder**: Update intellectual fingerprint (obsessions, blind spots, crossings)
4. **Meta-Learning Scheduler**: Decide if prompt evolution is safe
5. **Prompt Evolution Engine**: Propose surgical prompt edits, shadow-test, version
6. **Domain Knowledge Expander**: Enrich file-based knowledge store
7. Save all artifacts to `workspace/learning/`

## Artifact Contract

Every stage writes:
- One structured JSON artifact
- One human-readable Markdown artifact
- One manifest entry with status and summary
- Trace events to `execution_trace.jsonl`

## Privacy Model

- Raw screenshots never leave the device
- Only compressed text summaries are sent to Claude
- Password fields, bank pages, incognito browsers auto-filtered
- Configurable app blacklist (`config.observation.blacklist_apps`)
- All data in local workspace directory
