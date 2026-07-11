# Architecture

Digital Unconscious is a local orchestrator around two bounded loops:

```text
daily:    observe -> compress -> connect -> judge -> brief -> promote
research: literature -> feasibility -> data -> analysis -> draft -> review
```

## Design rules

1. **One-person product.** Runtime state belongs to one local workspace.
2. **Provider-blind domain logic.** Agents know the `AIBackend` contract, not SDKs or CLIs.
3. **Subscription-first execution.** Codex and Claude Code can complete the loop without API keys.
4. **Artifacts over hidden state.** Each research stage writes JSON, Markdown, and trace events.
5. **Bounded memory.** Backlogs, prompt versions, RAG documents, and logs have explicit caps.
6. **Lazy edges.** Vision, tray, browser, vector, and document libraries load only when used.
7. **Human publication boundary.** Research can be drafted and reviewed automatically; submission cannot.

## Runtime map

```text
CLI / local dashboard
        |
        v
DigitalUnconsciousEngine -------- ResearchPipeline
        |                                |
        +-- observers                    +-- six artifact stages
        +-- daily agents                 +-- review/revision loop
        +-- task queue                    +-- approval record
        +-- bounded memory
        |
        v
CircuitBreaker -> ModelRouter -> provider adapter
                                  |-- Codex CLI (subscription)
                                  |-- Claude Code CLI (subscription)
                                  |-- DeepSeek API (optional)
                                  |-- GLM API (optional)
                                  |-- OpenAI Responses API (optional)
                                  |-- Anthropic API (optional)
                                  `-- Kimi API (compatibility)
```

## Model kernel

The v2 model seam replaces the former 998-line `ai_backend.py` monolith.

| Module | Responsibility |
| --- | --- |
| `backends/base.py` | `AIBackend`, `AIResponse`, image and reasoning normalization |
| `backends/local.py` | isolated `codex exec` and `claude -p` runners |
| `backends/hosted.py` | optional hosted adapters and current model aliases |
| `backends/router.py` | prefix parsing, readiness, workload routing, and fallback |
| `ai_backend.py` | small compatibility facade for external callers |

Local calls execute in a fresh temporary directory. Codex uses a read-only sandbox; Claude Code starts in safe mode with no tools unless a call explicitly needs image reading or web search. This prevents a summarization request from inheriting access to the repository or the user's home directory.

## Artifact contract

Each research stage updates `run_manifest.json`, appends to `execution_trace.jsonl`, and writes a structured and readable artifact. A stage is complete only when its artifact can be reloaded. Resume skips completed stages and re-enters at the first incomplete one.

Typical workspace:

```text
workspace/
  daily/cycle_YYYY-MM-DD/
  ideas/idea_backlog.jsonl
  knowledge/
  learning/
  queue/
  runs/<run-id>/
  service/
  setup/
```

## Failure behavior

- Provider failure advances through the configured fallback chain.
- A circuit breaker bounds retries and opens after repeated failure.
- Model-dependent work that still cannot complete is persisted to the task queue.
- Optional dependency failure degrades the relevant edge, not import of the whole package.
- Research stage completion is recorded only after its artifact is written.

## Data boundary

Storage and orchestration are local, but model inference may be remote. Text prompts and vision images cross the boundary to the provider selected for that call. See [SECURITY.md](SECURITY.md) for the precise policy.
