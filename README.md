<p align="center">
  <h1 align="center">Digital Unconscious</h1>
  <p align="center"><strong>Your screen behaviour is an unread research journal.<br>This system reads it for you.</strong></p>
</p>

<p align="center">
  <code>Passive Screen Observation</code> &times; <code>AI Idea Generation</code> &times; <code>Autonomous Research</code>
</p>

---

**Digital Unconscious** watches what you browse, read, and search throughout your day. It compresses those behavioural signals into structured summaries, generates cross-domain research ideas, scores them with an adversarial judge, and delivers a daily briefing of your best ideas — all running locally, all automatic.

The best ideas get promoted into a full autonomous research pipeline: literature review, feasibility assessment, data acquisition, analysis, paper drafting, and AI peer review.

## How it works

```
You browse the web, read papers, write code, chat on Slack...
         |
    Screenpipe captures screen text (local, private)
         |
    Compression Agent (Claude Haiku) distils 30-min windows
         |
    Idea Generator (Claude Opus, high temperature)
    draws cross-domain connections from your behaviour
         |
    Judge Agent (Claude Sonnet, low temperature)
    scores on novelty, feasibility, relevance, timeliness
         |
    Daily Briefing — your "hidden research agenda" revealed
         |
    Auto-Research Pipeline (optional, for top ideas)
    Literature → Feasibility → Data → Analysis → Paper → Review
         |
    Learning Engine — the system gets smarter about YOU over time
```

## Quickstart (one line)

```bash
pip install "digital-unconscious[full] @ git+https://github.com/shoal-rat/digital-unconscious.git" && du
```

That's it. On first run, a setup wizard opens in your browser. Configure your research fields, click "Start", and the system runs silently in the background from then on. You'll receive daily briefings automatically.

### What happens after setup

1. The system registers itself in Windows Startup (runs on login)
2. A background service observes your screen via screenpipe
3. Every day at your configured time, it generates a briefing
4. Open `du dashboard` anytime to view briefings and ideas
5. You never need to touch it again

### Alternative install

```bash
# Clone and install locally
git clone https://github.com/shoal-rat/digital-unconscious.git
cd digital-unconscious
pip install -e ".[full]"

# One-time setup (opens browser wizard)
du

# Set your focus (ideas will be filtered to these fields)
du config --focus "economics research,behavioral finance"
du config --primary "pricing psychology,decision science"

# Run your first daily cycle (with a log file or screenpipe)
du daily --log-file your_activity.jsonl

# Open the dashboard
du dashboard

# Or launch the system tray icon
du tray
```

## Features

### Passive observation
- Integrates with [screenpipe](https://github.com/mediar-ai/screenpipe) for 24/7 screen capture
- Falls back to JSONL or plain-text activity logs
- Privacy-first: passwords, bank pages, incognito auto-filtered
- Configurable app blacklist

### AI-powered idea generation
- Claude Opus at high temperature for divergent, cross-domain ideas
- Focus field filtering: ideas *inspired by* any domain but *applicable to* your field
- RAG knowledge base (ChromaDB or file fallback) enriches context
- Human Idea Model personalises output based on your intellectual fingerprint

### Adversarial judging
- Claude Sonnet scores each idea on 4 weighted dimensions
- Conservative by design: ~1 in 200 ideas reaches the "include" threshold
- Focus field alignment: off-topic ideas are heavily penalised
- Heuristic fallback when AI is unavailable

### Autonomous research pipeline
- 6-stage pipeline: Literature → Feasibility → Data → Analysis → Drafting → Review
- Claude Code computer-use for browsing papers and downloading datasets
- AI-powered feasibility assessment, code generation, paper writing
- Adversarial peer review loop with auto-revision

### Self-improving
- Prompt Evolution Engine proposes and shadow-tests incremental prompt improvements
- Human Idea Model tracks your obsessions, blind spots, and productive crossings
- Domain Knowledge Expander enriches the RAG base from successful runs
- Meta-Learning Scheduler prevents over-fitting with conservative update rules

### Desktop integration
- System tray icon (right-click for quick actions)
- Web dashboard at `localhost:9830`
- Background service daemon with configurable intervals
- Platform autostart (launch on login)

## Commands

| Command | What it does |
|---------|-------------|
| `du tray` | System tray icon with quick actions |
| `du dashboard` | Web UI for briefings, ideas, learning |
| `du start` | Foreground observation service |
| `du daily` | Run one daily cycle now |
| `du research --idea "..."` | Research a specific idea |
| `du research --auto` | Auto-pick and research the best backlog idea |
| `du learn` | Run the learning engine |
| `du config --focus "field1,field2"` | Set focus fields for idea filtering |
| `du service start/stop/status` | Background daemon management |
| `du credential add/list` | Manage encrypted credentials |

## Dual-mode AI backend

| | Claude Code (subscription) | Anthropic API |
|---|---|---|
| Cost | Included in Pro/Max ($20-200/mo) | ~$7-13/mo (pay per token) |
| Setup | `claude /login` | `export ANTHROPIC_API_KEY=sk-...` |
| Temperature | Simulated via prompt engineering | Direct parameter control |
| Computer use | Native Chrome browsing | Not available |

The system auto-detects which mode to use. Set `mode = "claude_code"` or `mode = "api"` in `config/pipeline.toml` to force one.

## Configuration

All settings live in `config/pipeline.toml`:

```toml
[idea]
primary_domains = ["AI tools", "product design"]
secondary_domains = ["cognitive science", "business models"]
focus_fields = ["economics research", "management"]  # ideas must land here
include_threshold = 75
max_ideas_per_cycle = 8

[observation]
enabled = true
screenpipe_url = "http://localhost:3030"
blacklist_apps = ["game", "video_player"]

[automation]
auto_execute = true
checkpoint_policy = "best_effort"  # autonomous, no human gates
```

## Architecture

```
src/du_research/
  agents/           AI agents (compressor, idea generator, judge, briefing, writer, reviewer)
  stages/           Research pipeline stages (literature, feasibility, data, analysis, drafting, review)
  ai_backend.py     Dual-mode abstraction (Claude Code CLI / Anthropic SDK)
  circuit_breaker.py Three-state resilience with exponential backoff
  engine.py         Main orchestrator (daily cycle, learning, observation service)
  pipeline.py       6-stage research pipeline
  rag.py            ChromaDB vector store for RAG knowledge base
  dashboard.py      Web UI server
  tray.py           System tray application
  observation.py    Screenpipe integration + file fallback
```

## Privacy

- All data stays on your device. Screenshots never leave.
- Only compressed text summaries go to Claude (Code or API).
- Configurable app and domain blacklists.
- No telemetry, no cross-user learning, no phone-home.

## Testing

```bash
PYTHONPATH=src python -m pytest tests/ -v  # 57 tests
```

## License

MIT

---

<p align="center"><em>"The system that learns from you becomes, over time, more you than any tool you have ever used."</em></p>
