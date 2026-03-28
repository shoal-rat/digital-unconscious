# Digital Unconscious — Product Requirements Document v0.3.0

## 1. Summary

Digital Unconscious is a local-first, open-source AI system that passively observes
your daily screen behaviour, distils creative research and product ideas, scores
them with an adversarial judge, and optionally turns the best ones into full research
dossiers — all with zero human intervention beyond initial setup.

The system operates in two modes: **Claude Code subscription** (headless CLI) or
**Anthropic API** (direct SDK calls). Both modes share identical pipeline logic
through a unified `AIBackend` abstraction.

## 2. Problem

Knowledge workers consume vast information daily — papers, products, conversations,
code — but these "unconscious attention signals" are discarded. The best ideas often
emerge from unexpected cross-domain connections that no one captures systematically.

Existing tools (Rewind AI, screenpipe) solve "I forgot" — memory recall.
Digital Unconscious solves "I didn't realise I was thinking about this" — idea
externalisation from passive behaviour.

## 3. Product Goal

Given a day of screen behaviour, the system should produce:

- Compressed behaviour summaries (30-minute windows)
- 5-8 creative, cross-domain research/product ideas per window
- Adversarial scoring across novelty, feasibility, relevance, timeliness
- A daily Markdown briefing with the top ideas and overlooked signals
- Automatic research pipeline execution for ideas scoring above threshold
- A persistent Human Idea Model that personalises output over time
- Self-improving prompts via the Prompt Evolution Engine

## 4. Architecture

### 4.1 Dual-Mode AI Backend

| Dimension | Claude Code Mode | API Mode |
|-----------|-----------------|----------|
| Prerequisite | Pro/Max subscription ($20-200/mo) | ANTHROPIC_API_KEY |
| Temperature | Prompt-engineered (system prompt hints) | Direct parameter control |
| Session mgmt | `--session-id` for multi-turn | Custom message history |
| Rate limits | Subscription tier limits | API tier limits |

Selection: `config.ai.mode = "auto"` picks API if `ANTHROPIC_API_KEY` is set, else Claude Code.

### 4.2 Full Pipeline

```
LAYER 1: Observation
  screenpipe (localhost:3030) or file fallback
  OCR text · dwell-time weighting · privacy filtering
       ↓
LAYER 2: Compression (Claude Haiku, strict mode)
  Sliding windows · deduplication · structured summaries
       ↓
LAYER 3: Idea Generation (Claude Opus, creative mode, temp 0.95)
  + RAG knowledge context · Human Idea Model · domain configuration
       ↓
LAYER 4: Judge (Claude Sonnet, strict mode, temp 0.1)
  4 dimensions · weighted scoring · personalized rubrics
       ↓
LAYER 5: Briefing (Claude Opus, balanced mode)
  Focus · Hidden Problem · Top Ideas · Overlooked Signals · Persona Insight
       ↓
LAYER 6: Auto-Research (optional)
  Top ideas → full 6-stage research pipeline → dossier
       ↓
LAYER 7: Learning Engine (nightly)
  Run Outcome Analyzer · Human Idea Model · Prompt Evolution
  Domain Knowledge Expander · Meta-Learning Scheduler
```

### 4.3 Circuit Breaker

Three-state resilience: CLOSED → OPEN → HALF_OPEN.
Exponential backoff on rate limits (429). Automatic recovery testing.

## 5. Functional Requirements

### FR1. Observation Layer
- Screenpipe HTTP API integration (localhost:3030, OCR frames)
- File-based fallback (plain text or JSONL daily logs)
- Privacy filtering: password fields, bank pages, incognito browsers
- Configurable app blacklist via `config.observation.blacklist_apps`
- Deduplication with dwell-time accumulation
- Sliding-window grouping (default 30 minutes)

### FR2. Compression Layer
- Claude Haiku in strict mode for fast, cheap summarisation
- Output schema: time_range, dominant_topics, high_weight_content,
  app_distribution, intent_signals, cross_domain_hints, search_queries
- Heuristic fallback (Counter-based frequency analysis) when AI unavailable
- Cap: 80 frames per compression call

### FR3. Idea Generation
- Claude Opus in creative mode (temp 0.95 via prompt engineering)
- 5-8 ideas per window, each with: title, description, source_behaviour,
  domains, research_question, data_hint, novelty_signal
- Accepts RAG knowledge context from domain knowledge store
- Accepts Human Idea Model for personalised generation
- Domain configuration: primary (weight 0.6), secondary (0.3), open (0.1)

### FR4. Judge Agent
- Claude Sonnet in strict mode (temp 0.1) for consistent scoring
- 4 dimensions: Novelty (25%), Feasibility (30%), Domain Relevance (25%), Timeliness (20%)
- Verdicts: discard (<60), hold (60-75), include (>75)
- Personalized evaluation using Human Idea Model context
- Heuristic fallback scoring when AI unavailable

### FR5. Daily Briefing
- Today's Focus, Hidden Problem, Top Ideas, Overlooked Signals, Work Persona Insight
- Saved as `briefing_YYYY-MM-DD.md` in daily cycle directory
- Configurable briefing time (default 22:00)

### FR6. Auto-Research Trigger
- Ideas scoring above `include_threshold` automatically enter the 6-stage research pipeline
- Configurable top-K selection (`auto_research_top_k`, default 1)
- Smart deduplication: similarity threshold + cooldown period
- Results saved to `research_runs.json`

### FR7. Research Pipeline (6 stages)
1. Literature Scout — open API search (arXiv, PubMed, Semantic Scholar) + PDF download
2. Feasibility Assessment — go/no-go with 0-100 confidence score
3. Data Acquisition — open dataset registries (OSF, Zenodo, Kaggle)
4. Analysis & Figures — AI-powered code generation, auto-debug, 300 DPI figures
5. Paper Drafting — Methods → Results → Discussion → Introduction → Abstract
6. AI Peer Review Loop — adversarial reviewer + revision agent, quality gate ≥ 82/100

### FR8. Learning Engine
- **Run Outcome Analyzer**: patterns across runs (review bottlenecks, database effectiveness, recurring blockers)
- **Human Idea Model Builder**: intellectual fingerprint (core obsessions, preferred analogies, blind spots, research taste trajectory)
- **Prompt Evolution Engine**: incremental prompt edits, versioned, shadow-tested, auto-rollback
- **Domain Knowledge Expander**: enriches file-based knowledge store from successful runs
- **Meta-Learning Scheduler**: conservative update rules (min 3 runs, 1 evolution/week, quality regression detection)

### FR9. Credential Broker
- AES-GCM encrypted vault for institutional credentials
- One-time human provision per resource, stored locally forever
- CLI management: `du credential add/list`

### FR10. Service Daemon
- Background observation loop with configurable interval
- Automatic daily cycle trigger at briefing time
- Workspace maintenance and garbage collection
- Platform-specific autostart (login-item registration)

## 6. CLI Commands

| Command | Description |
|---------|-------------|
| `du start` | Start foreground observation service |
| `du daily` | Run full daily cycle manually |
| `du research --idea "..."` | Run research pipeline on an idea |
| `du research --auto` | Auto-select top backlog idea for research |
| `du research --resume --run-id ID` | Resume an existing research run |
| `du learn` | Run learning engine cycle |
| `du init` | Initialise workspace and config |
| `du config` | Show/update domain configuration |
| `du service start/stop/status` | Background daemon management |
| `du credential add/list` | Credential vault management |
| `du status --run-id ID` | Inspect research run status |
| `du logs --run-id ID` | View execution trace |

## 7. Data & Privacy

- All data stored locally. Screenshots never leave the device.
- Only compressed text summaries sent to Claude (Code or API).
- Configurable app/domain blacklists.
- One-click data deletion support.
- No cross-user learning, no telemetry, no phone-home.

## 8. Configuration

All settings in `config/pipeline.toml`:
- `[ai]`: mode, model selection per agent, API key
- `[observation]`: screenpipe URL, window minutes, blacklist apps
- `[idea]`: domains, thresholds, auto-research settings
- `[learning]`: prompt evolution, human model, min runs
- `[circuit_breaker]`: retries, backoff, failure threshold
- `[credentials]`: vault path, key path
- `[automation]`: browser settings, runner selection
- `[service]`: daemon paths, maintenance intervals
- `[retention]`: data retention periods

## 9. Non-Goals for v0.3

- ChromaDB vector database (file-based RAG context used instead)
- Journal-specific PDF formatting
- Autonomous paper submission
- Unsupervised browser login flows
- Federated/cross-user learning

## 10. Success Criteria

- Full daily cycle completes from observation to briefing
- Research pipeline produces inspectable dossiers from ideas
- Learning engine improves prompts after ≥3 completed runs
- Human Idea Model reflects user's evolving research interests
- All operations work offline (heuristic fallbacks) or online (AI-powered)
- 31+ tests passing covering all components

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Claude Code rate limits | Circuit breaker with exponential backoff; batch processing |
| API Key/subscription confusion | Auto-detection in `create_backend("auto")` |
| Privacy sensitivity | Local-first; text-only to AI; configurable blacklists |
| Prompt evolution degrades quality | Versioned prompts; shadow testing; auto-rollback on regression |
| Compression loses key signals | 30-day raw log retention; community-maintained prompts |
| Idea quality cold start | 2-4 week learning period; "Today's Summary" as interim value |

## 12. Future Phases

- ChromaDB semantic vector retrieval (upgrade from file-based RAG)
- Supervised browser automation in sandboxed environment
- Journal formatting and submission workflow
- Multi-user team deployment
- Community prompt marketplace
