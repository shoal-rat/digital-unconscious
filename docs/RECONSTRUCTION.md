# v2 reconstruction audit

This document records what the structural review found, what was replaced, and what remains intentionally outside the app.

## Findings

| Finding | Why it mattered | v2 decision |
| --- | --- | --- |
| The 998-line backend module mixed contracts, images, five providers, routing, and diagnostics | Every provider change risked the whole model layer | Split into base, local, hosted, and router modules; keep a compatibility facade |
| “Codex” meant OpenAI API rather than Codex CLI | The advertised zero-key path did not exist | Add isolated `codex exec`; reserve `openai:` for the API |
| Auto mode selected only Claude Code when no key existed | Codex subscriptions were ignored | Detect and route across both local CLIs |
| Provider routing ignored the shape and cost of work | Cheap summaries and deep review used the same preference | Introduce explicit workload-first defaults |
| Claude CLI calls inherited the current project and enabled broad agent behavior | A summary task had more authority than it needed | Use a temporary root, safe mode, and no tools by default |
| Setup stored an unclassified API key in plaintext JSON | It contradicted the encrypted-credential claim | Remove API keys from the setup form; use environment variables |
| Vision was disabled without an API key | Subscription CLIs can now process temporary image inputs | Enable zero-key vision through Codex, with Claude fallback |
| Documentation claimed screenshots never left the device | This was false whenever vision used a remote model | Replace with an explicit local-first versus local-model boundary |
| Launch copied missing Claude agents into the user's home directory | Hidden global mutation added noise and trust risk | Remove automatic skill installation from startup |
| Release tags and the only GitHub Release had diverged | Users could not tell what was current | Re-establish a coherent v2 release and changelog |
| Binary PRD artifacts and overlapping status documents dominated a tiny repository | Product history obscured the runnable product | Keep the durable architecture, routing, security, and audit docs; history remains in Git |

## Preserved

- the daily observation-to-briefing loop
- the six-stage research pipeline and artifact contract
- bounded memory and maintenance behavior
- circuit breaker, task queue, resume, and learning loop
- local dashboard, tray, and service management
- optional file, screenpipe, vision, vector, figure, and browser edges

## Deliberately not added

- a second chat interface
- accounts, teams, syncing, or hosted storage
- a generic workflow graph framework
- autonomous payments, terms acceptance, or publication
- a model-pricing database that would immediately become stale
- a heavyweight dependency-injection container

The result is more capable at the model boundary while smaller in its public story: observe, connect, challenge, and preserve.

## v2.1 structural follow-up

The second audit found that the product was still idea-first: downloaded paper extraction was unlinked, dataset discovery could not shape ideas, model-local IDs caused cross-day data loss, the per-cycle idea cap multiplied across windows, and resume was only a parsed flag. v2.1 adds a linked source/evidence/opportunity/study-card graph, a single backlog repository, a global daily budget, complete research-seed preservation, and artifact-backed resume. It also makes exploration explicit instead of silently starting a dossier from a daily scan.

## Remaining engineering debt

- `dashboard.py` and `learning_engine.py` are still large orchestration modules. The backlog and research ideation contracts are now separate; further extraction should follow a concrete second implementation rather than a framework-first rewrite.
- The test suite is comprehensive but concentrated in `test_engine.py`; new backend tests should continue moving toward provider-focused files.
- Browser automation remains the highest-risk optional edge and should eventually run in an operating-system sandbox rather than only a dedicated browser profile.
- ChromaDB remains optional because the bounded file store is adequate for a small single-user app.
