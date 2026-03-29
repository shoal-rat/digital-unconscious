# Releases

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
