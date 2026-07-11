# Releases

## v2.0.0 — Subscription-first reconstruction

Digital Unconscious is now a smaller, safer, genuinely zero-key application.

### New

- Real local Codex support through isolated `codex exec` calls.
- Rebuilt Claude Code runner using safe mode, subscription auth, native effort, and structured output.
- Workload-first routing for DeepSeek V4, GLM-5.1, Codex, Claude Code, GPT-5.6, Claude API, and Kimi.
- `du doctor` for secret-free runtime readiness diagnostics.
- Zero-key vision through temporary Codex image attachments with Claude fallback.
- Modular backend kernel: base contract, local runners, hosted adapters, and router.
- New visual identity, diagrams, README, architecture, model-routing, security, and reconstruction docs.

### Fixed

- `codex` no longer incorrectly aliases the OpenAI API.
- Local model calls no longer inherit repository or home-directory access.
- Setup no longer stores an unclassified API key in plaintext JSON.
- Privacy language now states when text and screenshots cross the local boundary.
- Automatic startup no longer mutates `~/.claude/agents`.
- Auto mode uses both signed-in local CLIs and falls back across available providers.

### Removed

- Stale binary PRD exports and overlapping planning/status documents. Git history remains the archive.
- Startup-time Claude agent installation.
- The claim that screenshots never leave the device.

### Verification

- Unit and integration suite on the lightweight install.
- Real Codex subscription smoke call.
- Real Claude Code subscription smoke call.
- Package build and clean-wheel installation.
- Browser inspection of the local dashboard and setup flow.

## Earlier line

| Version | Theme |
| --- | --- |
| 1.7 | Dashboard and product interaction redesign |
| 1.6 | Vision observation and web search |
| 1.5 | Bounded long-term memory |
| 1.4 | Failover, reasoning controls, and usage tracking |
| 1.3 | Initial multi-provider API layer |
| 1.2 | Claude research-agent experiments |
| 1.0 | One-command desktop workflow |
| 0.9 and earlier | Research pipeline and daily loop foundations |

The complete historical notes remain available in Git before tag `v2.0.0`.
