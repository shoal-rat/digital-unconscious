# Digital Unconscious agent guide

## Product boundary

Keep this a small, single-user, local-first research loop. Do not add accounts, team sync, a generic chat surface, autonomous publication, payments, or hidden global configuration.

Local-first describes storage and orchestration. Model inference may be remote. Never claim that screenshots stay on-device when vision is enabled.

## Project shape

- Python package: `src/du_research`
- Tests: `tests`, using `unittest`
- Default config: `config/pipeline.toml`
- Runtime state: `workspace/` (ignored by Git)
- Optional integrations must stay lazy so a core install can import, test, and run from a text log.

## Model boundary

- All model access uses `AIBackend.call(...)`.
- `backends/base.py` owns the contract and normalization.
- `backends/local.py` owns Codex and Claude Code subscription runners.
- `backends/hosted.py` owns optional API adapters.
- `backends/router.py` owns provider selection and fallback.
- `ai_backend.py` is a compatibility facade; do not rebuild logic there.
- Provider prefixes: `codex:`, `claude_code:`, `deepseek:`, `glm:`, `openai:`, `anthropic:`, `kimi:`.
- A fallback provider must use its own default model, never an incompatible model name from the failed provider.

Local model calls must remain isolated: temporary working directory, read-only/safe execution, and no tools unless the call explicitly needs image reading or web search.

## Secrets and safety

- Do not commit or persist provider keys in setup JSON. Use environment variables.
- Keep the encrypted credential vault for supervised browser tasks only.
- Do not copy browser profiles, cookies, or credentials.
- Never automate CAPTCHA, MFA, terms acceptance, payments, subscriptions, or final submission.
- Treat passive screenshots as sensitive outbound model input.

## Verification

Before publishing:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
python -m build
```

For backend changes, add mocked CLI/SDK tests and run one real local smoke call when the relevant signed-in CLI is available.

For dashboard changes, start `du dashboard --no-open` on an unused port and inspect setup, dashboard, models, and status in a browser.

## Style

- Prefer explicit dataclasses and small functions over frameworks.
- Preserve artifact compatibility when refactoring orchestration.
- Comments should explain a security boundary, data contract, or non-obvious provider quirk.
- Add a dependency only when its feature cannot remain optional.
