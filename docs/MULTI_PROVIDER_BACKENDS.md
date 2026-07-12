# Model routing

The router chooses by workload first and provider second. Local subscription CLIs are real providers—not aliases for hosted APIs.

## Zero-key providers

| Prefix | Command | Authentication | Isolation |
| --- | --- | --- | --- |
| `codex:` | `codex exec` | existing ChatGPT/Codex login | temporary workspace, read-only sandbox |
| `claude_code:` | `claude -p` | existing Claude subscription login | temporary workspace, safe mode, tools off by default |

Both support structured output. Codex receives images with `--image`; Claude Code receives a temporary image and only the `Read` tool for that call. Temporary files are removed automatically.

## Optional hosted providers

| Prefix | Environment variable | Default |
| --- | --- | --- |
| `deepseek:` | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |
| `glm:` / `zai:` | `ZAI_API_KEY` or `GLM_API_KEY` | `glm-5.1` |
| `openai:` | `OPENAI_API_KEY` | `gpt-5.6-sol` via Responses API |
| `anthropic:` | `ANTHROPIC_API_KEY` | `claude-sonnet-5` |
| `kimi:` | `MOONSHOT_API_KEY` or `KIMI_API_KEY` | `kimi-k2.6` |

DeepSeek uses V4 Flash for inexpensive synthesis and V4 Pro when explicitly selected. GLM-5.1 is available for long-horizon agentic analysis. Hosted adapters remain optional; the core package imports and tests without their SDKs.

## Defaults

```toml
[ai]
mode = "auto"
fallback = true
fallback_order = ["deepseek", "glm", "codex", "claude_code", "openai", "anthropic", "kimi"]

compressor_model = "deepseek:deepseek-v4-flash"
briefing_model = "deepseek:deepseek-v4-flash"
creative_model = "codex:default"
judge_model = "claude_code:sonnet"
writer_model = "claude_code:opus"
reviewer_model = "codex:default"
analysis_model = "glm:glm-5.1"
evidence_model = "deepseek:deepseek-v4-flash"
ideation_model = "codex:default"
ideation_review_model = "claude_code:sonnet"
```

Idea Lab uses a deliberate cascade: economical evidence extraction, strong cross-source synthesis, then an independent methods challenge. The final evidence-ID validation and weighted score are deterministic Python, so a model cannot override provenance gates or silently change the rubric.

An explicit prefix expresses preference, not fragility. If that provider is unavailable and fallback is enabled, the router selects the first ready provider from `fallback_order`. A fallback uses its own default model rather than receiving an incompatible model name.

## Modes

| Mode | Behavior |
| --- | --- |
| `auto` / `multi` | route by prefix and fall back across ready providers |
| `codex` | pin the Codex subscription CLI |
| `claude_code` | pin the Claude Code subscription CLI |
| `deepseek`, `glm`, `openai`, `anthropic`, `kimi` | pin one optional hosted adapter |

`codex` now means the local Codex CLI. `openai` means the OpenAI API. This corrects the ambiguous pre-v2 behavior where “Codex” was only an alias for OpenAI Chat Completions.

## Reasoning control

Agents pass one provider-neutral `think` value. It may be an integer budget or `low`, `medium`, `high`, `xhigh`, or `max`.

- Codex and Claude Code receive their native effort controls where supported.
- OpenAI receives Responses API reasoning effort.
- Anthropic receives an extended-thinking budget.
- DeepSeek receives thinking mode plus its supported effort mapping.

No chain-of-thought is persisted by Digital Unconscious. Only final response text, structured output, usage metadata, and provider trace fields enter artifacts.

## Inspect and debug

```bash
du doctor        # installed CLIs and configured optional providers
du models        # every agent's preferred route and fallback
du doctor --json
du models --json
```

Provider responses include `router_provider`, `router_chain`, and—after failover—`router_fallback_from` in internal metadata.
