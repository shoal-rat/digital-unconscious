# Digital Unconscious Agent Guide

## Project Shape
- Python package under `src/du_research`.
- Tests live in `tests` and use `unittest`; run them with `python -m unittest discover -s tests -v`.
- Default config is `config/pipeline.toml`; user runtime state is written under `workspace/`, which is ignored by git.
- Browser automation and external API SDKs are optional at import time. Keep optional integrations lazy so lightweight installs can still run tests, docs, dashboard, and dry runs.

## Backend Conventions
- Use the `AIBackend.call(...)` protocol for all model access.
- Provider routing belongs in `du_research.ai_backend`, not in individual agents.
- Supported backend modes are `auto`, `multi`, `claude_code`, `api`/`anthropic`, `openai`/`codex`, and `kimi`/`moonshot`.
- Per-call model prefixes are supported: `openai:gpt-5.5`, `kimi:kimi-k2.6`, `anthropic:claude-opus-4-8`, and `claude_code:opus`.
- Do not commit real API keys. Prefer `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `MOONSHOT_API_KEY`, or `KIMI_API_KEY`.

## Safety And Privacy
- Raw screenshots should remain local. Only compressed text summaries should be sent to model APIs.
- Do not add browser profile reuse, cookie export, password export, unsupervised payments, or automatic terms acceptance.
- Credential storage must remain encrypted; keep `cryptography` as a runtime dependency for vault writes.
- Browser automation should use allowlisted task packs and manual checkpoints for CAPTCHA, MFA, consent, or payment walls.

## Verification
- Before finishing code changes, run:
  `python -m unittest discover -s tests -v`
- For dashboard or setup UI changes, start `du dashboard --no-open` on an unused local port and inspect with the in-app browser.
- Keep generated runtime artifacts under `workspace/`; do not commit them.

## Style
- Prefer small, explicit dataclasses and module-level helpers over broad framework abstractions.
- Keep docs and comments practical. Add comments only when they explain a non-obvious boundary or safety rule.
- Preserve the local-first posture: integrations may connect to external models or tools only through explicit configuration.
