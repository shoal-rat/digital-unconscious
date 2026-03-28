# Architecture

## Module Map

- `du_research.cli`: command-line entrypoints
- `du_research.config`: TOML config loading
- `du_research.integrations.claude_code`: supervised computer-use task export
- `du_research.pipeline`: end-to-end orchestration
- `du_research.storage`: artifact persistence and trace logging
- `du_research.models`: dataclasses for normalized stage data
- `du_research.stages.daily_capture`: idea extraction from daily logs and notes
- `du_research.stages.literature`: open literature discovery and ranking
- `du_research.stages.feasibility`: heuristic feasibility scoring
- `du_research.stages.data_sources`: open dataset discovery
- `du_research.stages.analysis`: descriptive analysis and figures for local CSV input
- `du_research.stages.drafting`: manuscript starter generation
- `du_research.stages.review`: rubric scoring and revision guidance
- `du_research.stages.learning`: cross-run aggregation

## Execution Model

The pipeline is synchronous and artifact-driven. Each stage receives Python objects plus prior artifacts, then emits:

- stage JSON
- stage Markdown
- a `StageResult` summary stored in `run_manifest.json`
- one or more trace events in `execution_trace.jsonl`

## Ranking Heuristics

Daily ideas, literature, and dataset ranking use explainable heuristics:

- token overlap with the idea
- recency
- source-specific metadata such as citation count or access status
- summary/abstract availability

## Browser Automation Boundary

Browser automation is not executed by this Python CLI. Instead, the system exports a supervised task pack for an external computer-use runner so login steps, consent walls, and other sensitive actions remain explicitly gated.


## Analysis Model

The analysis stage is intentionally narrow:

- CSV only
- descriptive statistics only
- simple SVG output for one figure
- no inferential statistics claims unless data and logic explicitly support them

## Learning Model

The MVP learning model tracks:

- recurring keywords
- repeated domains
- average review score
- preferred literature sources
- common blockers such as “no data” or “low evidence”

This is a local summary model, not a behavioral surveillance system.
