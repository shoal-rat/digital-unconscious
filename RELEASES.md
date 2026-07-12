# Releases

## v2.1.0 — Evidence-first Idea Lab

Digital Unconscious now closes the missing reverse loop: papers and local dataset structure can generate traceable, testable research ideas.

### New

- `du ideate` / `du explore` for PDF, text, Markdown, JSON, JSONL, CSV, and TSV inputs.
- A local source ledger with stable IDs and SHA-256 content identity.
- Exact-anchor evidence cards; invented model quotes fail deterministic validation.
- Privacy-preserving dataset profiles that exclude raw rows and example values from model prompts.
- Named opportunity operators for contradictions, limitations, boundary extensions, measurement gaps, method transfer, replication, robustness, and dataset reuse.
- Falsifiable study cards with hypothesis, null, design, required variables, confounders, smallest useful test, and falsifier.
- DeepSeek extraction, Codex synthesis, Claude methods review, deterministic final ranking, and normal provider fallback.
- An Idea Lab dashboard screen and portable `workspace/ideation/session_*` reports.
- Optional `papers` extra for PDF text extraction.

### Reconstructed

- One `IdeaBacklog` repository now owns identity, deduplication, ranking, and retention across the engine, CLI, tray, and Idea Lab, with locked atomic writes for concurrent processes.
- Response-local IDs such as `idea_001` are replaced by content-derived app IDs and reconciled without cross-window collisions.
- The idea budget is enforced once per daily cycle, not once per compressed time window.
- Full structured idea cards survive promotion into research through `research_seed.json` and the run manifest.
- Real resume reloads completed artifacts; a new path or changed file content invalidates analysis, drafting, and review while preserving upstream work.
- Model routing diagnostics include the three Idea Lab workloads.

### Safer defaults

- Daily scans no longer launch research, browser automation, or autostart by default.
- Research no longer evolves prompts from its own review scores by default; learning is explicit.
- Setup saves preferences without unexpectedly starting a background service.
- Automatic browser acquisition was removed from normal literature and dataset stages.
- Direct PDF downloads require a PDF signature and record their checksum.
- Acquisition guidance permits lawful open access only and stops at paywalls, login, CAPTCHA, MFA, terms, and payment.
- Dataset prompts omit local paths, filenames, raw rows, and example values.
- Dashboard-rendered model content is HTML-escaped and state changes require an exact same-origin local host and port.

### Verification

- Evidence graph, source parser, privacy, backlog identity, resume, and acquisition regression tests.
- Full unit and integration suite on the core install.
- Clean package build and wheel smoke installation.
- Browser inspection of the dashboard and Idea Lab at desktop and mobile widths.

## v2.0.0 — Subscription-first reconstruction

Digital Unconscious is now a smaller, safer, genuinely zero-key application.

### New

- Real local Codex support through isolated `codex exec` calls.
- Rebuilt Claude Code runner using safe mode, subscription auth, native effort, and structured output.
- Workload-first routing for DeepSeek V4, GLM-5.1, Codex, Claude Code, GPT-5.6, Claude API, and Kimi.
- `du doctor` for secret-free runtime readiness diagnostics.
- Zero-key vision through temporary Codex image attachments with Claude fallback.
- Modular backend kernel: base contract, local runners, hosted adapters, and router.
- New visual identity, diagrams, README, architecture, model-routing, security, and reconstruction docs.

### Fixed

- `codex` no longer incorrectly aliases the OpenAI API.
- Local model calls no longer inherit repository or home-directory access.
- Setup no longer stores an unclassified API key in plaintext JSON.
- Privacy language now states when text and screenshots cross the local boundary.
- Automatic startup no longer mutates `~/.claude/agents`.
- Auto mode uses both signed-in local CLIs and falls back across available providers.

### Removed

- Stale binary PRD exports and overlapping planning/status documents. Git history remains the archive.
- Startup-time Claude agent installation.
- The claim that screenshots never leave the device.

### Verification

- Unit and integration suite on the lightweight install.
- Real Codex subscription smoke call.
- Real Claude Code subscription smoke call.
- Package build and clean-wheel installation.
- Browser inspection of the local dashboard and setup flow.

## Earlier line

| Version | Theme |
| --- | --- |
| 1.7 | Dashboard and product interaction redesign |
| 1.6 | Vision observation and web search |
| 1.5 | Bounded long-term memory |
| 1.4 | Failover, reasoning controls, and usage tracking |
| 1.3 | Initial multi-provider API layer |
| 1.2 | Claude research-agent experiments |
| 1.0 | One-command desktop workflow |
| 0.9 and earlier | Research pipeline and daily loop foundations |

The complete historical notes remain available in Git before tag `v2.0.0`.
