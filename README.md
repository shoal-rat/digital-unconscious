# DU Research MVP

`DU Research` is a local-first research pipeline that turns an idea into a reproducible research dossier.

This repository deliberately implements the buildable part of the original concept:

- daily idea capture from exported notes or activity logs
- passive observation via Screenpipe-driven daily cycles
- literature discovery from open APIs
- open PDF downloading when a direct PDF link is available
- feasibility scoring
- dataset source discovery
- optional CSV-based descriptive analysis
- manuscript starter generation
- reviewer-style scoring
- learning summaries across runs

It does **not** claim to autonomously produce submission-ready academic papers from arbitrary ideas without human involvement. The improved PRD in [docs/PRD.md](docs/PRD.md) narrows the scope to a realistic MVP and keeps the larger vision as future work.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
python -m du_research research --idea "Cognitive load as a SaaS pricing axis"
python -m du_research research --idea "Behavioral drivers of churn" --data-file tests/fixtures/sample_data.csv
python -m du_research learn
```

Artifacts are written to `workspace/`.

## Commands

```powershell
python -m du_research research --idea "Your idea"
python -m du_research research --idea-id idea_001
python -m du_research daily
python -m du_research start --iterations 1
python -m du_research research --idea "Your idea" --data-file path\to\data.csv
python -m du_research research --idea "Your idea" --dry-run
python -m du_research daily-capture --input path\to\daily_log.txt
python -m du_research status --run-id run_20260328_193000_your_idea
python -m du_research export-computer-task --run-id run_20260328_193000_your_idea
python -m du_research learn
```

## Claude Code / Computer Use

This repo can export a supervised browser task pack for a Claude Code or other computer-use runner, but it does not store or automate your primary Chrome profile or account credentials itself. The boundary is documented in `docs/CLAUDE_CODE_INTEGRATION.md`.

## Passive Observation

`du daily` now prefers passive observation from the local Screenpipe service at the configured `screenpipe_url`. The cycle is:

`observe -> compress -> generate ideas -> judge -> briefing -> auto-promote top ideas into research`

If Screenpipe is unavailable, the system falls back to `--log-file` or `observation.fallback_log_path`.

## Output Layout

Each run creates a folder like:

```text
workspace/
  runs/
    run_20260328_193000_example/
      run_manifest.json
      execution_trace.jsonl
      01_literature/
      02_feasibility/
      03_data_sources/
      04_analysis/
      05_drafting/
      06_review/
      learning_signal.json
  learning/
    human_idea_model.json
    learning_changes.md
```

## Testing

```powershell
python -m unittest discover -s tests -v
```
