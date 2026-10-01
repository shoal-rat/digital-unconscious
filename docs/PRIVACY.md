# What stays in the harbour

Digital Unconscious watches the tide of one person's attention, so the boundary has to be exact.

## What the tide watcher sees

Every few seconds the tide watcher reads:

- the name of the foreground application;
- its window title (if *Read window titles* is on and the OS permits it);
- the active tab's address in Chromium browsers and Safari on macOS (if *Read browser addresses* is on);
- seconds since the last keyboard or mouse input.

It never takes screenshots, never reads window contents, and never records keystrokes.

Before a glance becomes part of the driftline:

| Rule | Effect |
| --- | --- |
| Fog: quiet apps (password managers by default) | dropped entirely |
| Fog: quiet sites (`bank`, `paypal.com`, `accounts.google.com`, sign-in hosts…) | dropped entirely |
| Private or incognito windows | dropped entirely |
| Chat and mail apps, and webmail | recorded as time only: no titles, no addresses |
| E-mail addresses, card-like and long numbers, token-like strings in titles | replaced with `[email]`, `[number]`, `[id]` |
| URLs | credentials, query strings and fragments removed; id-like path segments replaced; search engines keep only the query as the subject |

Slack water (no input for `sense.idle_seconds`) is not credited.

## Where it settles

One SQLite file, `memory.db`, in the harbour folder (`~/.digital-unconscious/` or `$DUN_HOME`), plus `config.toml`.
After two weeks the visits to one subject on one day merge into a single row, and the tide washes the raw driftline
away after `sense.retention_days` (90 by default), together with the digest that cached its sorting. Currents, their
daily tides, dives, fish and seabed searches stay until you let the sharks have them: *Harbour → The sea* can let the
shark eat a day or feed everything to the sharks, and so can `dun forget`. When the shark eats, the space is given
back to the disk. A login item's log is kept under 1 MB.

## What crosses the water

Only the crew's errands and seabed soundings:

| When | What is sent | To |
| --- | --- | --- |
| Sorting the catch (each dive, and any past day the night watch missed while night diving is on) | subject labels (cleaned titles, search queries, domains), times, bottle and washed-in excerpts, current names | the *digest* crew member |
| Diving | the day's topics with evidence labels, measured undercurrents, current names, your persona and waters, your net's counts, recent fish titles | the *dream* crew member |
| The lighthouse | the candidate fish and what they cite | the *critique* crew member |
| The seabed | the fish's sounding lines (search terms) | OpenAlex, then arXiv if needed |
| The seabed | the fish and the abstracts that came up | the *dive* crew member |

Claude Code and Codex use your subscription; the errand still goes to Anthropic or OpenAI. To keep every dive at
home, bring an Ollama model aboard:

```toml
[providers.ollama]
model = "qwen3:14b"

[models]
digest = "ollama"
dream = "ollama"
critique = "ollama"
dive = "ollama"
fallback = false
```

Crew members that run as CLIs work in an empty temporary folder with tools disabled (Claude Code) or a read-only
sandbox (Codex), so an errand cannot read your files.

## What never comes aboard

- API keys: hosted providers read them from environment variables only.
- Browser profiles, cookies or credentials of any kind.
- Model reasoning: only final answers, structured output and usage counts.
