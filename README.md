# Digital Unconscious

Passive observation to research ideas, daily briefs, and reproducible dossier pipelines.

`Digital Unconscious` is a local-first autonomous research engine that watches daily screen activity, compresses behavioral signals into ideas, ranks them, and can automatically promote strong ideas into a literature-to-manuscript workflow.

## What It Does

- passively captures behavior from Screenpipe or a fallback activity log
- accumulates observations in a long-running local daemon
- generates a daily idea briefing at a configured time
- deduplicates repeated ideas and avoids rerunning near-identical research
- scouts literature and datasets from open providers
- runs optional CSV analysis with figures and reproducibility artifacts
- drafts, reviews, revises, and packages a research dossier
- exports or runs Claude Code browser automation tasks for gated portals
- learns across runs with prompt evolution and a persistent human idea model

## System Flow

```text
Screen Activity
  -> Observation Journal
  -> Compression Agent
  -> Idea Generator
  -> Judge Agent
  -> Daily Briefing
  -> Auto-Research Gate
  -> Literature / Data / Analysis / Draft / Review / Submission Package
```

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m du_research.cli
```

On first launch, the app stores default setup choices locally. After that, launching without arguments defaults to the long-running observation service.

## Main Commands

```powershell
python -m du_research.cli
python -m du_research.cli start
python -m du_research.cli service start
python -m du_research.cli service status
python -m du_research.cli service stop
python -m du_research.cli daily
python -m du_research.cli research --idea "Cognitive load as a SaaS pricing axis"
python -m du_research.cli research --idea-id idea_001
python -m du_research.cli research --idea "Behavioral drivers of churn" --data-file tests/fixtures/sample_data.csv
python -m du_research.cli learn
```

## Background Operation

- first-run setup persists defaults under `workspace/setup/`
- the foreground service polls on a schedule and writes status into `workspace/service/`
- the background daemon can be started with `service start`
- Windows autostart can be enabled once and then left alone
- retention and maintenance settings prune old observation, browser, and daily artifacts over time

## Claude Code Automation

This repository supports browser automation through Claude Code task packs and direct execution via the configured automation runner. It does not silently take over your personal Chrome profile. Credential handling and supervised boundaries are documented in [docs/CLAUDE_CODE_INTEGRATION.md](docs/CLAUDE_CODE_INTEGRATION.md).

## Repository Layout

```text
src/du_research/
  agents/          multi-agent prompting and learning
  stages/          literature, data, analysis, drafting, review
  automation.py    browser automation runner
  engine.py        long-running observation and daily idea cycle
  pipeline.py      end-to-end research orchestration
  service_manager.py
  maintenance.py
docs/
  PRD.md
  ARCHITECTURE.md
  IMPLEMENTATION_STATUS.md
tests/
workspace/         runtime artifacts, ignored by git
```

## Documentation

- Product scope: [docs/PRD.md](docs/PRD.md)
- System design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Claude Code integration boundary: [docs/CLAUDE_CODE_INTEGRATION.md](docs/CLAUDE_CODE_INTEGRATION.md)
- Implementation tracking: [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md)

## Testing

```powershell
python -m unittest discover -s tests -v
```

## License

MIT. See [LICENSE](LICENSE).
