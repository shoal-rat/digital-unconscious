# Competitive Landscape & Architecture Research

Research date: 2026-03-28

## 1. GPT-Researcher (assafelovic/gpt-researcher)

**What it is:** An autonomous agent that conducts deep research on any topic and produces detailed reports with citations. Outperformed Perplexity, OpenAI Deep Research, and others in Carnegie Mellon's DeepResearchGym benchmark (May 2025).

**Architecture:**
- **Planner-Executor-Publisher pattern** with LangGraph-based state machine orchestration
- **7 specialized agents:** ChiefEditor (coordinator), Editor (outline), Researcher (per-section deep research), Reviewer (quality validation), Reviser (improvement), Writer (compilation), Publisher (format conversion)
- **Review-revision loop** per section before final publication -- sections processed in parallel, each passing through quality gates
- Agents communicate through **LangGraph state objects** maintaining research context

**Key design patterns:**
- **Three-tier LLM strategy:** FAST_LLM (gpt-4o-mini, summaries), SMART_LLM (gpt-4o, report generation), STRATEGIC_LLM (o1/o3, planning). Optimizes cost vs. quality per task complexity.
- **Embedding-based context compression:** Uses vector similarity (not token counting) to compress gathered content to relevant chunks
- **Modular skill architecture:** Components (ResearchConductor, BrowserManager, ContextManager, ReportGenerator) are discrete, testable "skills"
- **Provider abstraction:** Unified `"provider:model"` string format supports 25+ LLM providers without code changes

**Rate limits / concurrency:**
- Sub-queries executed in parallel
- URL scraping controlled by `MAX_SCRAPER_WORKERS` env var
- MCP strategy options: `fast` (single execution + caching), `deep` (per-query), `disabled`

**What works well:** Reliable report quality, extensive retriever support (Tavily, Google, DuckDuckGo, Brave, SearXNG), MCP server for Claude Desktop integration.

**What doesn't:** Heavy dependency on LangGraph/LangChain ecosystem. Sequential quality gates can slow end-to-end time.

**Relevance to Digital Unconscious:** The three-tier LLM strategy and embedding-based context compression are directly applicable. The planner-executor-publisher pattern maps well to our daily_capture -> literature -> drafting pipeline.

---

## 2. Screenpipe (screenpipe/screenpipe)

**What it is:** A Rust application that continuously captures screen and audio, creating a searchable AI-powered memory of everything you do. All data stored locally. 100% local, privacy-first.

**Architecture:**
- **Event-driven capture** (NOT continuous polling): Triggers on app focus changes, clicks, scrolling, typing pauses, clipboard copy, or idle fallback (~5s timer)
- Each capture pairs a **screenshot + accessibility tree data** at same timestamp
- Falls back to **OCR** when accessibility data unavailable (remote desktops, games)
- **Audio processing:** 30-second chunks through Whisper (local) or Deepgram (cloud)
- **SQLite database** (`~/.screenpipe/db.sqlite`) for metadata, text, transcriptions
- **Media files** in `~/.screenpipe/data/` (JPEG screenshots, audio chunks)

**Rust workspace crates:**
- `screenpipe-vision`: Screen capture + OCR
- `screenpipe-audio`: Audio capture + speech-to-text
- `screenpipe-accessibility`: UI event detection (macOS, Windows)
- `screenpipe-db`: SQLite schema
- `screenpipe-server`: REST API (localhost:3030)
- `screenpipe-app-tauri`: Desktop UI (Tauri + Next.js)

**Pipes (AI Agent) system:**
- Pipes are `.md` prompt files in `~/.screenpipe/pipes/{name}/`
- Execute on cron-like schedules
- An AI agent (Claude Code or similar) reads the prompt, queries screenpipe API, executes actions
- REST API provides content search, keyword search, UI element queries, frame access, direct SQL

**Resource usage:** ~600 MB RAM, 5-10% CPU, 5-10 GB storage/month (thanks to event-driven capture reducing redundancy).

**What works well:** Event-driven capture is very efficient. Accessibility tree + OCR fallback is clever. Local-first architecture is privacy-respecting. MCP server integration for Claude Desktop.

**What doesn't:** OCR accuracy varies with screen complexity. No built-in semantic analysis of captured content. Pipes system is simple (cron + prompt) with limited inter-pipe coordination.

**Relevance to Digital Unconscious:** This is the closest analog to our daily_capture stage. Key lessons:
1. **Event-driven capture beats polling** -- dramatically reduces resource usage
2. **Accessibility tree > OCR** for structured text extraction
3. **SQLite + filesystem** is sufficient for local artifact storage
4. **Pipe-as-prompt** pattern (`.md` files on cron) is simple but effective for scheduling AI analysis passes

---

## 3. Observer AI (Roy3838/Observer)

**What it is:** Local open-source micro-agents that observe, log, and react to your digital environment. Privacy-first, runs in browser.

**Architecture:**
- **Sensor-Model-Tools paradigm:** Sensors (screen OCR, screenshots, audio, clipboard, camera) -> Small LLMs -> Tool functions
- Agents run on **configurable loop intervals** in browser sandbox
- Sensors inject data as variables: `$SCREEN_OCR`, `$SCREEN_64`, `$MEMORY`, `$IMEMORY`
- **Cross-agent communication** via `getMemory(agentId)` calls
- **Lazy evaluation:** Sensors only capture when agents request them

**Key design patterns:**
- **Browser sandbox execution** limits agent capabilities for safety
- **Memory persistence** (text and image) across agent loops
- **Reactive tooling:** Agents conditionally trigger notifications (email, Discord, Telegram)

**What works well:** Very lightweight, easy agent creation, visual agent builder with drag-and-drop blocks.

**What doesn't:** Browser sandbox limits capabilities. No deep analysis pipeline. OCR accuracy issues. No inter-process communication with external apps.

**Relevance to Digital Unconscious:** The sensor variable injection pattern (`$SCREEN_OCR`, `$MEMORY`) is a clean interface design. Cross-agent memory sharing via `getMemory(agentId)` is a simple coordination mechanism we could adopt for inter-stage communication.

---

## 4. AutoResearchClaw (aiming-lab/AutoResearchClaw)

**What it is:** Fully autonomous 23-stage pipeline that turns a research idea into a conference-ready paper (NeurIPS/ICML/ICLR format). No human intervention required.

**Architecture -- 8 Phases:**
- **Phase A (Scoping):** Topic initialization, problem decomposition
- **Phase B (Literature):** Multi-source discovery (OpenAlex, Semantic Scholar, arXiv), relevance screening
- **Phase C (Synthesis):** Knowledge clustering, gap identification, hypothesis generation via multi-agent debate
- **Phase D (Design):** Experiment planning with hardware detection (CUDA/MPS/CPU)
- **Phase E (Execution):** Sandbox-based experiments with self-healing code repair (up to 10 iterations)
- **Phase F (Analysis & Decision):** Multi-agent result interpretation with PROCEED/REFINE/PIVOT logic
- **Phase G (Writing):** Section-by-section drafting (5,000-6,500 words), peer review
- **Phase H (Finalization):** Quality gates, LaTeX export, citation verification

**Multi-agent debate system:**
- **3 personas:** Innovator (novel approaches), Pragmatist (feasibility), Contrarian (challenge assumptions)
- Independent reasoning then synthesis -- addresses "groupthink" in single-model reasoning

**Self-healing executor:**
- AST validation before execution
- Immutable harness preventing side effects
- NaN/Inf fast-fail detection
- Repair loop with error context fed to LLM
- Partial result capture from incomplete runs
- 300-second default time budget

**Rate limit handling:**
- **Circuit breaker pattern** for literature API degradation
- Query expansion + deduplication to reduce redundant searches
- Fallback chain across literature sources
- Pipeline continues with available data, flags gaps

**Citation verification (4-layer):**
1. arXiv ID validation
2. CrossRef/DataCite DOI matching
3. Semantic Scholar title similarity
4. LLM-based relevance scoring
- Hallucinated references automatically pruned

**Decision loops:** Stage 15 analysis can jump backward: REFINE -> Stage 13 (retune params), PIVOT -> Stage 8 (restart hypothesis generation). Artifact versioning prevents data loss on loops.

**MetaClaw integration:** Cross-run learning captures failure lessons as reusable skills injected into all 23 stages (+18.3% robustness in controlled tests).

**Relevance to Digital Unconscious:** This is the most architecturally similar project. Key takeaways:
1. **Circuit breaker pattern** for API rate limits -- our pipeline should degrade gracefully
2. **PROCEED/REFINE/PIVOT decision logic** at checkpoints -- we need similar adaptive flow control
3. **Multi-agent debate** (Innovator/Pragmatist/Contrarian) for hypothesis quality -- applicable to our feasibility scoring
4. **Cross-run learning (MetaClaw)** is exactly what our `learning` stage aims to do
5. **Self-healing execution** with retry budgets -- practical for autonomous operation
6. **Artifact versioning per iteration** -- prevents data loss during refinement loops

---

## 5. Stanford STORM (stanford-oval/storm)

**What it is:** LLM-powered knowledge curation system that researches topics and generates full-length reports with citations. Published at NAACL 2024.

**Architecture (two-stage):**
- **Pre-writing stage:** Internet research + outline generation
  - **Perspective-guided question asking:** Surveys existing articles on similar topics to discover diverse viewpoints
  - **Simulated conversation:** LLM simulates dialogue between a Wikipedia writer and topic expert, grounded in internet sources. Enables iterative deepening through follow-up questions.
- **Writing stage:** Uses outline + references to populate cited content

**Key design patterns:**
- **Cost-optimized multi-LM:** Cheaper models (GPT-3.5) for conversation simulation/query splitting, powerful models (GPT-4) for outline generation and final writing
- **DSPy framework** for modular components
- **Four customizable modules:** curation, outline generation, article generation, polishing
- **Interface-based design:** Clean separation between module specs and implementations
- **Retriever abstraction:** Supports YouRM, BingSearch, VectorRM, SerperRM, Brave, SearXNG, DuckDuckGo, Tavily, GoogleSearch, Azure AI Search

**Relevance to Digital Unconscious:** The perspective-guided question asking and simulated conversation patterns are directly applicable to our literature discovery stage. Having an LLM "interview" a topic from multiple angles generates better research questions than single-shot prompting.

---

## 6. Deer-Flow 2.0 (bytedance/deer-flow)

**What it is:** ByteDance's open-source SuperAgent framework for research, coding, and creative tasks. Ground-up rewrite in v2.0, currently #1 trending on GitHub (March 2026).

**Architecture:**
- Built on **LangGraph + LangChain**
- **SuperAgent pattern:** Lead agent decomposes objectives into sub-tasks, spawns multiple sub-agents executing in parallel
- **Three sandbox modes:** Local execution, Docker isolation, Kubernetes via Provisioner (horizontal scaling)
- **Long-term memory system** with multi-session continuity, queryable via commands
- **Gateway pattern:** Unified message routing across Telegram, Slack, Feishu

**Key design patterns:**
- **Config-as-Code:** YAML-driven model/agent/sandbox definitions with env var substitution
- **Recursion limits:** Configurable per-user/channel (e.g., `recursion_limit: 150`)
- **Context engineering:** Global/per-user config injection, thinking modes, plan-vs-execution toggles
- **Extensible skills** via MCP server integration with OAuth token flows
- **LangSmith integration** for tracing and observability

**Tech stack:** Python 3.12+, LangChain, LangGraph, Node.js 22+, Docker, Kubernetes (optional), nginx

**Relevance to Digital Unconscious:** The config-as-code pattern and recursion limits are directly applicable safety mechanisms. The sandbox architecture (local -> Docker -> K8s progression) is a good scaling roadmap. The IM gateway pattern could be useful if we want to surface insights to messaging platforms.

---

## 7. MassGen (massgen/MassGen)

**What it is:** Multi-agent scaling system where an orchestrator coordinates multiple AI agents working in parallel on the same problem, achieving consensus through collaborative refinement.

**Architecture:**
- **Orchestrator** manages task distribution across 3-5 concurrent agents
- Each agent uses a **different frontier model** (Claude, GPT, Gemini, Grok)
- **Real-time intelligence sharing** via collaboration hub
- **Consensus mechanism:** Agents vote when confidence threshold reached, best collectively validated answer wins
- **Adaptive restarts:** Agents can restart upon receiving new peer insights

**What works well:** Provider agnosticism, rich TUI with real-time visualization, quickstart wizard.

**What doesn't:** Cost multiplication (N agents = Nx API cost), convergence uncertainty, tool integration complexity.

**Relevance to Digital Unconscious:** The consensus-through-diversity pattern is interesting for our review stage -- having multiple model perspectives validate a research idea could improve quality. However, the cost multiplication makes this impractical for daily automated runs.

---

## 8. Claude Code Automation Projects

### Anthropic's Official Autonomous Coding Demo (anthropics/claude-quickstarts)

**Two-agent pattern:**
1. **Initializer Agent (Session 1):** Reads spec, generates `feature_list.json` (200 test cases), sets up project
2. **Coding Agent (Sessions 2+):** Reads feature list, implements sequentially, marks as "passing", commits to git

**Key patterns:**
- **`feature_list.json` as source of truth** -- decouples agent sessions from state, enables arbitrary resumption
- **Fresh context window per session** -- prevents context degradation
- **Explicit state transfer via JSON** -- not context memory
- **Sequential feature processing** -- avoids parallelization complexity
- **Security: allowlisted bash commands + filesystem bounds + sandbox**

### Ralph (frankbria/ralph-claude-code)

**Continuous autonomous development loop:**
- Reads instructions from `.ralph/PROMPT.md` -> invokes Claude Code -> tracks progress -> evaluates completion -> repeats

**Key innovations:**
- **Dual-condition exit gate:** Requires both (a) >= 2 completion indicators AND (b) explicit `EXIT_SIGNAL: true` from Claude
- **Three-layer circuit breaker:** Opens after 3 loops with no progress or 5 loops with identical errors. OPEN -> HALF_OPEN -> CLOSED state transitions with 30-minute cooldown.
- **Rate limit handling:** 100 API calls/hour (configurable), 1-120 minute timeouts per invocation, 24-hour session expiration. Three-layer detection: timeout guards, JSON parsing for `rate_limit_event`, filtered text fallback.

**Relevance to Digital Unconscious:** Ralph's circuit breaker and exit detection patterns are directly applicable to our pipeline orchestration. The dual-condition exit gate (heuristic + explicit signal) is a robust pattern for knowing when a stage is truly complete.

---

## Cross-Cutting Patterns & Recommendations

### 1. Multi-Tier LLM Strategy (adopt)
Used by GPT-Researcher, STORM, and AutoResearchClaw. Map task complexity to model tier:
- **Fast/cheap model:** Daily capture parsing, keyword extraction, basic classification
- **Smart model:** Literature analysis, feasibility scoring, draft generation
- **Strategic model:** Research planning, hypothesis generation, review synthesis

### 2. Circuit Breaker for API Resilience (adopt)
Used by AutoResearchClaw and Ralph. Our pipeline should:
- Track consecutive failures per API endpoint
- Degrade gracefully (continue with partial data)
- Auto-recover after cooldown period
- Log gaps for user awareness

### 3. Event-Driven Capture over Polling (adopt)
Screenpipe's approach: trigger on meaningful OS events, not fixed intervals. For our daily_capture stage, this means processing new notes/logs only when they change rather than scanning everything on a schedule.

### 4. Artifact Versioning at Checkpoints (adopt)
AutoResearchClaw and Anthropic's demo both persist artifacts (JSON, git commits) at each stage. Our `run_manifest.json` and `execution_trace.jsonl` already do this -- good alignment.

### 5. Adaptive Decision Loops (consider)
AutoResearchClaw's PROCEED/REFINE/PIVOT pattern at stage boundaries. Our pipeline is currently linear; adding backward jumps (e.g., if feasibility score is too low, regenerate ideas) would improve output quality.

### 6. Cross-Run Learning (adopt)
AutoResearchClaw's MetaClaw captures failure lessons as reusable skills. Our `learning` stage already aims for this. Key insight: inject learned patterns into ALL stages, not just the learning stage.

### 7. Multi-Agent Debate for Quality (consider)
AutoResearchClaw's Innovator/Pragmatist/Contrarian pattern. Applicable to our feasibility and review stages, but increases cost. Could be optional/configurable.

### 8. Persistent State Pattern (adopt)
Anthropic's `feature_list.json` and Ralph's checkpoint system. Our pipeline already uses `run_manifest.json` -- ensure each stage can resume from checkpoint without re-running earlier stages.

### 9. Dual-Condition Completion Detection (adopt)
Ralph's pattern: require both heuristic indicators AND explicit signal before declaring a stage complete. Prevents premature termination in autonomous mode.

### 10. Provider Abstraction (adopt)
GPT-Researcher's `"provider:model"` format. Our pipeline should support swapping LLM providers without code changes to avoid vendor lock-in and enable the multi-tier strategy.

---

## Positioning: What Makes Digital Unconscious Unique

None of the surveyed projects combine ALL of these capabilities:

| Capability | GPT-Res | Screenpipe | Observer | AutoResearch | STORM | Deer-Flow | **Ours** |
|---|---|---|---|---|---|---|---|
| Passive screen/context observation | - | Yes | Yes | - | - | - | **Yes** |
| Autonomous idea generation | - | - | - | Partial | - | - | **Yes** |
| Literature discovery | Yes | - | - | Yes | Yes | Yes | **Yes** |
| Feasibility scoring | - | - | - | Partial | - | - | **Yes** |
| Dataset discovery | - | - | - | - | - | - | **Yes** |
| End-to-end paper drafting | Yes | - | - | Yes | Yes | - | **Yes** |
| Cross-run learning | - | - | - | Yes | - | Yes | **Yes** |
| Claude Code integration | Partial | Partial | - | Partial | - | Partial | **Yes** |
| Local-first / privacy-respecting | - | Yes | Yes | - | - | - | **Yes** |

The "Digital Unconscious" concept -- where passive observation of a researcher's daily digital activity feeds into an autonomous research pipeline -- remains a genuinely novel combination. The closest analogs are Screenpipe (observation without research) and AutoResearchClaw (research without observation).
