<p align="center">
  <img src="https://raw.githubusercontent.com/shoal-rat/digital-unconscious/main/docs/assets/logo.svg" alt="Digital Unconscious" width="520">
</p>

<p align="center"><strong>A quiet research loop for the ideas hidden inside your workday.</strong></p>

<p align="center">
  <a href="https://github.com/shoal-rat/digital-unconscious/actions"><img src="https://img.shields.io/github/actions/workflow/status/shoal-rat/digital-unconscious/ci.yml?branch=main&style=flat-square" alt="Build"></a>
  <a href="https://github.com/shoal-rat/digital-unconscious/releases"><img src="https://img.shields.io/github/v/release/shoal-rat/digital-unconscious?style=flat-square" alt="Release"></a>
  <a href="https://github.com/shoal-rat/digital-unconscious/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-6ee7b7?style=flat-square" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-8b9cff?style=flat-square" alt="Python 3.11 plus">
  <img src="https://img.shields.io/badge/API_key-not_required-f7c873?style=flat-square" alt="No API key required">
</p>

Digital Unconscious turns the weak signals around your screen activity into a short daily briefing and a searchable idea backlog. Its Research Idea Lab can also turn your papers and the structure of a local dataset into cited evidence, research opportunities, and falsifiable study cards.

It is deliberately small. There is no cloud account, team workspace, social feed, or chat shell. Your signed-in **Codex** or **Claude Code** subscription can run the complete loop, so an API key is optional.

## Start in two minutes

```bash
git clone https://github.com/shoal-rat/digital-unconscious.git
cd digital-unconscious
python -m pip install -e ".[vision,papers]"

du doctor
du daily --log-file tests/fixtures/daily_log.txt
du dashboard
```

`du doctor` reports which local subscription CLIs and optional providers are ready without printing secrets. If either Codex or Claude Code is installed and signed in, the app can run without an API key.

Install the richer desktop and research extras only if you need them:

```bash
python -m pip install -e ".[full]"
```

## The loop

<p align="center">
  <img src="https://raw.githubusercontent.com/shoal-rat/digital-unconscious/main/docs/assets/readme-daily-loop.svg" alt="The daily research loop">
</p>

1. **Observe** a screenshot, screenpipe stream, or plain text activity log.
2. **Compress** noisy activity into bounded 30-minute working notes.
3. **Connect** distant topics into concrete research questions.
4. **Challenge** each idea for novelty, feasibility, fit, and timing.
5. **Brief** only the strongest ideas and keep the rest in a bounded backlog.
6. **Research** selected ideas through literature, feasibility, data, analysis, drafting, and review.

Every research stage writes JSON, readable Markdown, and trace events. Runs can be inspected and resumed without replaying completed work.

A daily scan does **not** start literature downloads, browser automation, or manuscript drafting by default. Exploration is an explicit action.

## From papers and data to a useful research idea

<p align="center">
  <img src="https://raw.githubusercontent.com/shoal-rat/digital-unconscious/main/docs/assets/readme-idea-lab.svg" alt="Evidence-first Research Idea Lab">
</p>

```bash
du ideate \
  --paper reading/paper.pdf \
  --paper workspace/runs/example/01_literature/papers.json \
  --data data/observations.csv \
  --context "One-month behavioral economics project; observational methods only"
```

Idea Lab verifies exact source anchors, profiles dataset structure without sending raw rows, detects named opportunity patterns, and rejects cards without a hypothesis, null, design, smallest useful test, falsifier, and valid evidence IDs. It writes a readable report plus linked JSON under `workspace/ideation/session_*`.

Novelty is deliberately reported as uncertainty—not as “nobody has done this.” The supplied corpus cannot establish that claim. Read the [Idea Lab contract](https://github.com/shoal-rat/digital-unconscious/blob/main/docs/IDEA_LAB.md) for input formats, scoring, privacy, and artifacts.

## Models fit the work—not the other way around

<p align="center">
  <img src="https://raw.githubusercontent.com/shoal-rat/digital-unconscious/main/docs/assets/readme-provider-routing.svg" alt="Workload-first model routing">
</p>

The default policy is opinionated but replaceable:

| Work | Preferred route | Why |
| --- | --- | --- |
| Compression and briefing | DeepSeek V4 Flash, then local CLIs | Economical high-volume synthesis |
| Paper evidence extraction | DeepSeek V4 Flash, then local CLIs | Bounded structured extraction |
| Cross-source study design | Codex subscription | Strong linked reasoning with no API key |
| Adversarial methods review | Claude Code subscription | Careful challenge with no API key |
| Idea generation and adversarial review | Codex subscription | Strong structured reasoning with no API key |
| Judging, drafting, and revision | Claude Code subscription | Careful long-form work with no API key |
| Long-horizon analysis | GLM-5.1, then local CLIs | Optional agentic engineering path |

Missing credentials are normal. The router skips unavailable providers and falls through to Codex, Claude Code, or another configured backend. Inspect the effective plan with:

```bash
du models
```

Optional hosted providers use environment variables:

```bash
export DEEPSEEK_API_KEY="..."   # DeepSeek V4 Flash / Pro
export ZAI_API_KEY="..."        # GLM-5.1
export OPENAI_API_KEY="..."     # GPT-5.6 family
export ANTHROPIC_API_KEY="..."  # Claude API
```

See [Model routing](https://github.com/shoal-rat/digital-unconscious/blob/main/docs/MULTI_PROVIDER_BACKENDS.md) for modes, prefixes, and fallback behavior.

## Research pipeline

<p align="center">
  <img src="https://raw.githubusercontent.com/shoal-rat/digital-unconscious/main/docs/assets/readme-research-pipeline.svg" alt="Six-stage research pipeline">
</p>

Promote a specific question or let the app choose the strongest backlog item:

```bash
du research --idea "How can sparse attention change long-running personal research agents?"
du research --auto
```

The pipeline searches open scholarly sources, checks feasibility, discovers datasets, creates analysis artifacts, drafts a dossier, and runs a review/revision loop. Final submission remains a human decision.

`--resume --run-id …` reloads completed artifacts instead of replaying their model or network work. Supplying a new `--data-file` preserves upstream discovery and rebuilds analysis plus downstream artifacts.

## Commands

| Command | Purpose |
| --- | --- |
| `du doctor` | Check zero-key local runtimes and optional providers |
| `du daily [--log-file PATH]` | Run one observation-to-briefing cycle |
| `du dashboard` | Open the local dashboard |
| `du ideate --paper PATH [--data PATH]` | Build evidence-backed research study cards |
| `du explore …` | Alias for `du ideate` |
| `du models` | Explain per-agent routing and fallback |
| `du usage` | Summarize recorded token and cost metadata |
| `du research --idea "…"` | Start a six-stage research run |
| `du research --auto` | Promote the best backlog idea |
| `du learn` | Update the bounded personal idea model |
| `du service start|stop|status` | Manage the background cycle |
| `du config --focus "a,b"` | Narrow idea output to chosen fields |

## A small configuration surface

The checked-in defaults live in [`config/pipeline.toml`](https://github.com/shoal-rat/digital-unconscious/blob/main/config/pipeline.toml). The useful knobs are intentionally few:

```toml
[ai]
mode = "auto"
fallback = true
compressor_model = "deepseek:deepseek-v4-flash"
creative_model = "codex:default"
judge_model = "claude_code:sonnet"
evidence_model = "deepseek:deepseek-v4-flash"
ideation_model = "codex:default"
ideation_review_model = "claude_code:sonnet"

[observation]
source = "auto" # vision -> screenpipe -> file fallback
blacklist_apps = ["password manager", "bank"]

[idea]
focus_fields = ["economics", "management"]
include_threshold = 75
max_ideas_per_cycle = 8
auto_research_enabled = false
```

Provider prefixes are `codex:`, `claude_code:`, `deepseek:`, `glm:`, `openai:`, `anthropic:`, and `kimi:`.

## Privacy, precisely

“Local-first” here means storage, orchestration, retention, and the dashboard are local. It does **not** mean the selected language model runs on-device.

- Activity artifacts live under `workspace/` and are ignored by Git.
- Text summaries are sent to the selected model when a model-backed step runs.
- Vision sends the captured image to the selected model service, including when invoked through a subscription CLI.
- Temporary images used by Codex or Claude Code are isolated and deleted after each call.
- The setup page never stores API keys; optional hosted keys come from environment variables.
- An app blacklist can prevent capture from named applications.
- Browser automation never approves payments, CAPTCHA, MFA, or terms on your behalf.
- Idea Lab sends paper excerpts, but dataset inputs are reduced locally to schema and aggregate profile statistics; raw rows and example values stay local.

Read the complete [security and data boundary](https://github.com/shoal-rat/digital-unconscious/blob/main/docs/SECURITY.md) before enabling passive vision.

## Architecture

The application keeps a stable `AIBackend.call(...)` contract while splitting implementation into four narrow pieces:

```text
backends/base.py    response contract and shared normalization
backends/local.py   isolated Codex and Claude Code subscription runners
backends/hosted.py  optional API adapters
backends/router.py  availability, workload routing, and fallback
backlog.py          bounded identity and retention shared by every surface
ideation.py         source ledger, evidence graph, study cards, and ranking
```

The rest of the product remains provider-blind. See [Architecture](https://github.com/shoal-rat/digital-unconscious/blob/main/docs/ARCHITECTURE.md) and the [v2 reconstruction audit](https://github.com/shoal-rat/digital-unconscious/blob/main/docs/RECONSTRUCTION.md).

## Test and build

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
python -m build
```

The core test suite does not require network access or provider credentials. Local Codex and Claude Code have separate smoke paths exposed through `du doctor` and the backend integration tests.

## Scope

Digital Unconscious is for one person, on one machine, with bounded local memory. Multi-user deployment, institutional login automation, payments, and autonomous publication are intentionally outside the product.

MIT licensed. Contributions should preserve the small surface, honest privacy language, lazy optional dependencies, and human approval boundary described in [`AGENTS.md`](https://github.com/shoal-rat/digital-unconscious/blob/main/AGENTS.md).
