# Multi-Provider AI Backends

Updated: 2026-06-04

Digital Unconscious now supports Claude Code, Anthropic API, OpenAI API, and Kimi/Moonshot API through the same `AIBackend.call(...)` interface.

## Why This Shape

Recent coding-agent systems are strongest when the main workflow stays focused and delegates specific work to the right model/tool surface:

- Claude Code Opus 4.8 adds dynamic workflows for large codebase-scale tasks and stronger self-verification.
- Codex has goal mode, richer browser context, Appshots, in-app browser annotations, and Windows computer use for longer local work.
- Kimi K2.6 provides a 256K context window with text, image, and video input, long-horizon coding stability, and OpenAI-compatible API access.

The project uses those lessons conservatively:

- one backend protocol for every agent
- provider prefixes for targeted routing
- lazy optional integrations
- browser automation via explicit task packs and checkpoints
- tests and local artifacts as the completion gate

## Modes

Set `[ai].mode` in `config/pipeline.toml`:

| Mode | Behavior |
| --- | --- |
| `auto` | Uses `ANTHROPIC_API_KEY`, then `OPENAI_API_KEY`, then `MOONSHOT_API_KEY`/`KIMI_API_KEY`, then local Claude Code. |
| `multi` | Routes each call by provider prefix, or falls back in the same order as `auto`. |
| `claude_code` | Uses local `claude -p`. |
| `api` / `anthropic` | Uses the Anthropic Python SDK. |
| `openai` / `codex` | Uses the OpenAI Python SDK. |
| `kimi` / `moonshot` | Uses Kimi/Moonshot through the OpenAI-compatible chat API. |

## Provider Prefixes

Agent model fields can target a provider directly:

```toml
[ai]
mode = "multi"
creative_model = "openai:gpt-5.5"
judge_model = "anthropic:claude-sonnet-4-6"
compressor_model = "kimi:kimi-k2.6"
briefing_model = "claude_code:opus"
```

Unprefixed role aliases still work:

- `opus`
- `sonnet`
- `haiku`
- `codex`
- `kimi`

Each backend resolves those aliases to provider-appropriate defaults.

## Keys

Prefer environment variables:

```powershell
$env:ANTHROPIC_API_KEY = "..."
$env:OPENAI_API_KEY = "..."
$env:MOONSHOT_API_KEY = "..."
```

`KIMI_API_KEY` is also accepted for Kimi. Do not commit real keys to `config/pipeline.toml`.

## Optional Extras

```bash
pip install "digital-unconscious[api]"
pip install "digital-unconscious[openai]"
pip install "digital-unconscious[kimi]"
pip install "digital-unconscious[browser]"
pip install "digital-unconscious[full]"
```

`browser` installs Selenium for the non-Claude browser runner. Claude Code browser/computer-use task packs remain available without Selenium.

## References

- Anthropic Claude Opus 4.8: https://www.anthropic.com/news/claude-opus-4-8
- OpenAI ChatGPT/Codex release notes: https://help.openai.com/en/articles/6825453-chatgpt-release-notes
- Kimi K2.6 API docs: https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart
