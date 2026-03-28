# Implementation Status — v0.3.0

## Fully Implemented

### Digital Unconscious Core (PRD Features)
- [x] Dual-mode AI backend (Claude Code headless + Anthropic API SDK)
- [x] Circuit breaker with 3-state resilience and exponential backoff
- [x] Screenpipe observation layer with HTTP API integration
- [x] File-based fallback observation (text + JSONL)
- [x] Privacy filtering (hardcoded + configurable app blacklist)
- [x] Sliding-window compression with Claude Haiku
- [x] Creative idea generation with Claude Opus (high temp)
- [x] Adversarial judge with Claude Sonnet (low temp, 4 dimensions)
- [x] Personalized judge rubrics from Human Idea Model
- [x] Daily Markdown briefing generation
- [x] Auto-research trigger for high-scoring ideas
- [x] Smart deduplication for auto-research (similarity + cooldown)
- [x] RAG context from domain knowledge store
- [x] Human Idea Model (intellectual fingerprint)
- [x] Prompt Evolution Engine (versioned, shadow-tested)
- [x] Domain Knowledge Expander
- [x] Meta-Learning Scheduler (conservative update rules, auto-rollback)
- [x] Full learning cycle orchestration
- [x] Observation service daemon with configurable intervals
- [x] Platform autostart registration
- [x] Workspace maintenance and garbage collection

### Research Pipeline (6 Stages)
- [x] Literature Scout (arXiv, PubMed, Semantic Scholar, open PDF download)
- [x] Feasibility Assessment (0-100 score, go/review/archive)
- [x] Dataset Discovery (OSF, Zenodo, open registries)
- [x] Analysis & Figures (AI code generation, auto-debug, 300 DPI)
- [x] Paper Drafting (Markdown manuscript with BibTeX)
- [x] AI Peer Review Loop (adversarial reviewer + revision agent)
- [x] AI-powered writer, reviewer, revision, analysis-coder agents
- [x] Supervised computer-use task export

### Infrastructure
- [x] AES-GCM encrypted credential vault
- [x] CLI: daily, research (--idea, --auto, --resume), learn, init, config, start, service, credential, logs, status
- [x] Onboarding wizard with first-run setup
- [x] Background service daemon management
- [x] 31+ passing tests

## Not Yet Implemented (Deferred to Future Phases)

- [ ] ChromaDB vector database for semantic RAG retrieval
- [ ] Supervised browser automation in sandboxed environment
- [ ] Institutional portal navigation (library login flows)
- [ ] Journal-specific camera-ready typesetting
- [ ] Final submission workflow and approval queue
- [ ] Multi-user/team deployment
- [ ] Community prompt marketplace

## Why Those Items Are Deferred

The deferred items require external infrastructure (ChromaDB, browser sandbox),
institution-specific handling (portal UIs, credential flows), or represent
long-term platform features beyond the single-user local-first scope. The current
file-based RAG context provides the same information flow as ChromaDB would, using
the domain knowledge JSON store built by the learning engine.
