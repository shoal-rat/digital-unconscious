# Releases

## v1.5.0 - Bounded Long-Term Memory

Keeps months of daily use from bloating prompts or disk. The prompt context was
already well-bounded; these changes bound the on-disk/in-memory stores it draws on.

- Fixed an accelerating bug: `domain_knowledge.json` nested the entire previous
  file under `previous_version` every learning run (unbounded JSON depth). It now
  keeps a flat, bounded history of shallow snapshots; old nested files self-heal.
- RAG file store upserts by document id (re-ingesting a paper no longer duplicates
  it) and keeps only the newest `[retention].rag_max_documents`.
- Prompt evolution retains only the newest N version files per agent, rotates
  `evolution_log.jsonl`, and bounds the stacked "Learned refinement" blocks so an
  agent's active prompt can't grow forever.
- Idea backlog is capped to `[retention].idea_backlog_max`.
- `WorkspaceMaintenance` enforces every cap as a safety net (also reclaiming files
  that grew before the caps existed).
- New `[retention]` knobs: `rag_max_documents`, `idea_backlog_max`,
  `domain_knowledge_history`, `prompt_versions_kept`, `evolution_log_max_lines`.

Approach informed by long-running-agent memory research (Generative Agents,
MemGPT, recursive summarization): dedup, consolidate to a fixed size, and cap —
rather than bolt on a heavy memory subsystem, since the prompt path was already
bounded.

Tests: 70 -> 74 passing.

---

## v1.4.0 - Failover, Thinking, Usage Tracking, Import Fix

Reliability and capability:
- Provider failover: in `multi` mode (and `auto` when a hosted key is set) a
  failing provider transparently fails over to the next available one, ending at
  the local Claude Code CLI. Configurable via `[ai].fallback` and
  `[ai].fallback_order`.
- Extended thinking / reasoning effort: a `think` budget on every backend call,
  wired to idea generation and judging via `[ai].think_idea_budget` and
  `[ai].think_judge_budget`. Translated to Anthropic thinking budgets, OpenAI
  reasoning effort, the Kimi thinking toggle, or a Claude Code keyword.
- Per-cycle token/cost usage tracking, written to `usage.json`, appended as a
  briefing footer, and surfaced on the dashboard.

Fixes:
- The research pipeline crashed on import because `stages/analysis.py` imported
  reportlab (an undeclared dependency) and Pillow at module scope. Both are now
  lazy with graceful SVG-only degradation, and a `figures` extra declares them.

Tests: 60 -> 70 passing.

---

## v1.3.0 - Multi-Provider Agent Backends

New backend routing:
- Claude Code, Anthropic API, OpenAI/Codex, and Kimi/Moonshot now share one `AIBackend` interface.
- `mode = "multi"` can route per agent with model prefixes such as `openai:gpt-5.5`, `kimi:kimi-k2.6`, `anthropic:claude-opus-4-8`, or `claude_code:opus`.
- `mode = "auto"` now checks Anthropic, OpenAI, Kimi/Moonshot, then local Claude Code.
- Kimi support uses the OpenAI-compatible API endpoint and defaults to `kimi-k2.6`.

Reliability:
- Selenium is now a true optional browser extra instead of an import-time requirement.
- The encrypted credential vault now declares `cryptography` as a runtime dependency and lazy-loads it for clearer lightweight behavior.
- Added `AGENTS.md` with repo-specific guidance for future coding-agent runs.

---

## v1.2.0 — Research Skills (Latest)

**11 AI research skills auto-installed as Claude Code agents.**

New academic skills:
- **lit-search**: Multi-database literature search (Google Scholar, Semantic Scholar, arXiv, PubMed, OpenAlex)
- **citation-network**: Trace citation chains, identify hub papers, rising stars, cutting-edge work
- **research-gap**: Find under-explored areas, methodological gaps, contradictions in literature
- **abstract**: Generate 3 abstract versions (structured, narrative, concise)
- **journal-match**: Recommend journals ranked by scope fit, impact factor, open access
- **peer-review**: Adversarial 8-dimension review with actionable revision suggestions
- **cite-verify**: Verify DOIs, detect retractions and miscitations
- **report-template**: Generate formatted Markdown + HTML research reports

All skills auto-install to `~/.claude/agents/` on first run and update on upgrade.

```powershell
pip install "digital-unconscious[full] @ git+https://github.com/shoal-rat/digital-unconscious.git"
du
```

---

## v1.1.0 — Claude Code Latest Features

Upgraded AI backend to use the latest Claude Code capabilities:
- `--permission-mode auto` for fully autonomous operation (no approval prompts)
- `--json-schema` for guaranteed structured JSON output
- `--chrome` for native Chrome browser control
- `--allowedTools` with proper tool names (WebSearch, WebFetch, Bash, Read, Write)
- `--agent` flag to invoke specific research skills

---

## v1.0.0 — One-Command Desktop Experience

**Type `du` and everything runs automatically.**

- Tray icon appears in taskbar (right-click for all actions)
- Dashboard server starts at localhost:9830
- Observation service runs silently in background
- First run opens browser setup wizard
- No terminal knowledge needed — just `du` once, then interact through tray icon

---

## v0.9.0 — Full LLM-Only

Removed all traditional algorithms. Everything is AI-powered:
- Literature ranking by Claude Sonnet (not token overlap formulas)
- Paper enrichment by Claude Haiku (not keyword matching)
- Dataset ranking by Claude Haiku (not heuristic scoring)
- Feasibility assessment by Claude Opus (not weighted formulas)
- Review scoring by Claude reviewer agent (not hardcoded dimensions)
- Daily capture by Claude Sonnet (not regex scoring)
- Task queue: when LLM unavailable, work is saved and retried via `du drain`

---

## v0.6.0 — Pure AI, No Heuristics

Philosophical shift: if the LLM can't be reached, the work waits.
- Removed `_heuristic_compress`, `_heuristic_evaluate`, `_heuristic_briefing`
- All agents return None on failure (caller queues for later)
- New `TaskQueue` with persistent JSONL storage

---

## v0.5.0 — Focus Fields & System Tray

- **Focus field filtering**: `du config --focus "economics,management"` — ideas must land in your fields
- Cross-domain inspiration preserved (filter outputs, not inputs)
- **System tray app**: lightbulb icon with quick actions (dashboard, briefing, research)
- Conservative heuristic judge (~1-in-200 selectivity)

---

## v0.4.0 — ChromaDB RAG & Browser Automation

- **ChromaDB vector store** for semantic knowledge retrieval (with file fallback)
- **Claude Code Chrome browsing** for paper downloads and dataset acquisition
- **AI-powered feasibility assessment** via Claude Opus
- Configurable `checkpoint_policy`: "best_effort" (autonomous) vs "strict" (human gates)

---

## v0.3.0 — Close PRD Gaps

- Personalized judge rubrics from Human Idea Model
- RAG context from domain knowledge store passed to idea generator
- Configurable `blacklist_apps` wired into observation layer
- `du research --auto` and `--resume` CLI flags
- `daily_ideas` passed to learning cycle for complete model training
