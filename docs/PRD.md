# DU Research MVP PRD

## 1. Summary

This document replaces the original concept-heavy specification with a buildable product definition.

The shipped product is a **local-first research dossier generator**. It takes an idea, gathers open literature and dataset candidates, scores feasibility, optionally analyzes a local CSV, drafts a manuscript starter, runs a reviewer-style quality pass, stores a structured learning signal for future runs, and can extract candidate ideas from daily exported notes or activity files.

## 2. Problem

The original document assumes a fully autonomous research worker that can browse, log into institutional systems, run arbitrary analyses, and generate publication-ready papers with zero human intervention. That is a long-term vision, not an MVP.

The immediate user need is narrower:

- turn a rough idea into an evidence-backed research package
- extract candidate research ideas from daily exported notes or activity logs
- keep every step inspectable and reproducible
- reduce blank-page work for early-stage research
- accumulate local knowledge about which ideas and sources tend to work

## 3. Product Goal

Given a short idea statement, the system should produce within minutes:

- a ranked literature brief from open sources
- candidate ideas extracted from a daily activity file
- a feasibility memo with an explicit confidence score
- dataset candidates and a data acquisition plan
- an optional analysis artifact if the user supplies a CSV
- a manuscript starter in Markdown
- a reviewer report with dimension scores and revision guidance
- a machine-readable learning signal

## 4. Non-Goals For This Release

- no browser automation
- no credential vault
- no paywalled source access
- no journal-specific PDF formatting
- no autonomous submission
- no hidden LLM dependency
- no claims of statistical validity beyond the executed analysis

## 5. Target User

One technically capable individual researcher who wants a local workflow and values transparent artifacts over “agent magic”.

## 6. Success Criteria

- A run can be executed from the command line with only Python 3.11.
- Every stage writes inspectable artifacts to disk.
- The pipeline completes without API keys when network access is available.
- The system still completes in `--dry-run` mode without network access.
- Learning summaries aggregate completed runs without external services.

## 7. User Stories

1. As a researcher, I can give the system an idea and get a structured brief instead of starting from scratch.
2. As a researcher, I can attach a local CSV and receive a reproducible descriptive analysis.
3. As a researcher, I can inspect exactly which papers and datasets were found and why they were ranked.
4. As a researcher, I can compare runs over time and see which themes and sources are most productive.

## 8. Functional Requirements

### FR1. Run Creation

- Accept `research --idea "<text>"`.
- Generate a stable run id and artifact directory.
- Persist `run_manifest.json` and `execution_trace.jsonl`.

### FR1a. Daily Idea Capture

- Accept `daily-capture --input <path>`.
- Parse plain text or `jsonl` daily exports.
- Rank candidate research ideas with transparent heuristics.
- Append accepted candidates to a local backlog.

### FR2. Literature Discovery

- Query open literature providers with no credentials.
- Normalize results into a common schema.
- Rank results using transparent heuristics.
- Download direct open PDF links when available.
- Save raw paper metadata and a Markdown summary.

### FR3. Feasibility Assessment

- Produce a 0-100 confidence score.
- Classify the idea as `proceed`, `review`, or `archive`.
- Recommend methods, likely data shapes, and the most distinctive angle seen in the evidence.

### FR4. Dataset Discovery

- Query open dataset registries.
- Save ranked dataset candidates and a concrete acquisition plan.
- Distinguish between `open`, `restricted`, and `unknown` access.

### FR5. Analysis

- If a CSV is provided, profile it, generate descriptive statistics, and emit at least one figure.
- If no CSV is provided, generate an analysis plan instead of failing.
- Save machine-readable results and a reproducibility script.

### FR6. Drafting

- Generate a Markdown manuscript starter with sections for abstract, research question, literature, data plan, methods, results or planned analysis, limitations, and references.

### FR7. Review

- Score the draft across multiple dimensions.
- Emit actionable suggestions.
- Persist `quality_score.json`, `review_report.json`, and `review_report.md`.

### FR8. Learning

- Emit `learning_signal.json` after each run.
- Aggregate all runs into `human_idea_model.json`.
- Summarize changes in `learning_changes.md`.

### FR9. Supervised Computer-Use Handoff

- Export a `computer_use_task.json` artifact for an external Claude Code or computer-use runner.
- Explicitly mark login, CAPTCHA, consent, and payment moments as human-approval checkpoints.

## 9. Non-Functional Requirements

- Local-first storage only
- Standard-library-only Python implementation
- Deterministic heuristics where possible
- Clear failure states instead of silent retries
- Safe defaults for network timeouts

## 10. Architecture

### Pipeline Stages

1. `LiteratureScout`
2. `FeasibilityAssessor`
3. `DatasetScout`
4. `AnalysisStage`
5. `DraftingStage`
6. `ReviewStage`
7. `LearningAggregator`

### Data Flow

`idea -> literature -> feasibility -> datasets -> analysis -> draft -> review -> learning signal`

### Artifact Contract

Every stage writes:

- one structured JSON artifact
- one human-readable Markdown artifact
- one manifest entry with status and summary

## 11. Risks

- Public APIs may rate limit or change shape.
- Literature relevance scoring is heuristic, not semantic.
- Without user-supplied data, the analysis stage can only plan, not validate.
- Reviewer scoring is rubric-based and should not be treated as peer review.

## 12. Future Phases

The following remain valid long-term directions, but are intentionally deferred:

- authenticated data access
- supervised browser automation in a sandboxed environment
- LLM-backed synthesis and revision loops
- prompt evolution
- journal formatting and submission
- richer personalized idea modeling from passive behavior
