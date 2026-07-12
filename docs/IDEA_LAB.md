# Research Idea Lab

Idea Lab is the reverse half of Digital Unconscious: instead of starting with an idea and searching for papers, it starts with papers and local data structure and asks what small, useful study could follow.

```text
paper/data -> source ledger -> evidence cards -> opportunities -> study cards -> backlog
```

## Run it

Text, Markdown, and JSON paper records work with the core install. PDF extraction is an optional edge:

```bash
python -m pip install -e ".[papers]"

du ideate \
  --paper notes/paper-a.pdf \
  --paper workspace/runs/example/01_literature/papers.json \
  --data data/observations.csv \
  --context "Behavioral economics; one-month project; observational methods only"
```

`du explore` is an alias. The same workflow is available from **Idea Lab** in the local dashboard.

Use `--dry-run` to verify parsing and build deterministic fallback artifacts without calling a model. Use `--no-backlog` when you want an isolated exploration.

## What is grounded

Every accepted evidence card has:

- a stable source ID and content SHA-256;
- a type such as finding, method, limitation, boundary condition, or dataset field;
- a short exact anchor that deterministic code verifies against the source text;
- a locator, extraction method, and confidence.

An invented or altered anchor is discarded. A paper summary is evidence of what that summary says; it is not silently upgraded to full-text support.

Dataset inputs are profiled locally. The model receives field names, inferred types, row/column counts, sample missingness, sample cardinality, and numeric ranges. It does **not** receive local paths, filenames, raw rows, or example values. This makes the profile useful for research design without turning a basic ideation call into a data upload.

## How opportunities are found

The synthesis model must use a named operator:

- contradiction or replication;
- explicit limitation or boundary extension;
- method transfer;
- measurement or temporal gap;
- dataset reuse;
- robustness test.

These are hypotheses about useful next work, not proof of novelty. The report therefore records novelty as a risk or uncertainty. Establishing novelty requires a separate, documented search with adequate coverage.

## Study-card contract

A card is rejected unless it has source evidence plus a question, hypothesis, null hypothesis, study design, smallest useful test, and falsifier. Valid cards also ask for population, exposure, outcome, unit of analysis, operationalization, required variables, confounders, negative controls, and opposing evidence when available.

The reviewer scores five visible dimensions:

| Dimension | Weight |
| --- | ---: |
| Evidence strength | 24% |
| Testability | 24% |
| Dataset fit | 20% |
| Expected information gain | 18% |
| User fit | 14% |

Code—not the model—computes the final weighted score and verdict. `ready` and `develop` cards may enter the shared backlog; `fragile` cards stay in the session report.

## Artifact contract

Each `workspace/ideation/session_*` directory is a portable linked graph:

```text
source_manifest.json  source identity, hash, metadata, parse warnings
evidence_cards.json   exact anchors and source links
opportunities.json    named gap operators and evidence IDs
ideas.json            validated, ranked study cards
reviews.json          adversarial score inputs and risks
report.md             readable research memo
session.json          compact result and artifact locations
```

The IDs form the graph directly:

```text
source_id -> evidence_id -> opportunity_id -> idea_id
```

There is no graph server, hidden cloud database, or required vector store. JSON remains authoritative and can be inspected, versioned, or analyzed with ordinary tools.

## Model policy

The defaults match work to cost and difficulty:

- DeepSeek V4 Flash for bounded evidence extraction;
- Codex subscription for cross-source opportunity and study design;
- Claude Code subscription for adversarial methods review.

All three roles use the normal fallback router. Codex or Claude Code can therefore run the complete flow without provider API keys. Inspect the effective route with `du models`.
