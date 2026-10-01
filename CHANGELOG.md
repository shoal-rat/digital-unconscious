# Changelog

## 3.0.0 — rebuilt from the idea up

A new codebase. Earlier versions remain in Git history.

### The idea, done properly

- Continuous attention sampling (foreground app, window title, browser address, idle time) with real dwell time
  replaces hourly screenshots. macOS, Windows and X11 samplers; ActivityWatch import for everything else.
- Searches, jots and fed documents are treated as the strongest evidence of intent.
- Long-running **threads** and computed **signals** (orbit, return, surge, seed, collision, steady, fade) give the
  nightly dream memory across weeks.
- The **dream** writes a headline, a reflection, an undercurrent question and candidate sparks that must cite their
  evidence; invented references are rejected by code.
- An independent **critique** scores every card; code applies the rubric, your taste and a diversity rule.
- **Dive**: a literature check through OpenAlex and arXiv with validated citations and honest novelty language.
- **Taste**: keep, follow or let go of sparks with reasons; pin or mute threads.

### A native app

- PySide6 desktop app with a tray presence, single-instance behaviour and notifications.
- Visual language from a slow morning in Nice: linen and sand, the bay as the dream panel (azure by day, deep blue by
  night), the day's threads as pebbles on the shore, a light touch of Y2K gloss.
- The whole interface speaks as a small sea: the shore, currents, the shoal, the logbook, the harbour; a tide
  watcher, bottles, the lighthouse keeper, the crew, and a shark for forgetting. Code keeps plain names.
- Slow, meaningful motion: a breathing waterline, a shoal with one fish per idea, a bottle on the swell, a fin.
- English and Simplified Chinese interface and dreams (Chinese headings fall back to Songti beside Fraunces).
- `dun demo` opens the app on three weeks of sample memory.

### Light on the battery

- On macOS the tide watcher asks the window server directly through `ctypes` (about 0.1 ms a glance) instead of
  spawning AppleScript and `ioreg` three times every 15 seconds.
- Adaptive pacing: glances stretch from 15 s to 60 s while attention rests on one thing, only the idle clock is read
  at slack water, and on battery everything stretches further. Time is still counted from the real clock.
- One shared animation clock that stops when the window is hidden or behind other apps and goes calm on battery;
  the dream's waterline repaints only its own strip from cached layers. *Harbour → the water's motion* can make it
  calm or still.
- The window polls a small `api.pulse()` instead of the full state, every 30 s when it lives only in the menu bar.
- The crew is Claude Sonnet 5.5 (sorting, the lighthouse, the seabed) and Opus 5.5 (diving).

### Smaller and stricter

- Engine on the standard library: SQLite, urllib, subprocess. One memory file, one settings file.
- Subscription-first models: Claude Code and Codex without API keys; DeepSeek, GLM, Kimi, OpenAI, Anthropic and any
  OpenAI-compatible server (including Ollama) optional.
- Privacy at capture time: quiet apps and sites, private windows, time-only chat apps, redaction, URL cleaning,
  90-day raw-trace retention.
- The CLI is `dun` (it was `du`, which shadowed the Unix disk-usage command).

### Removed

The six-stage research pipeline (paper downloads, generated analysis code, manuscript drafting, review loops),
browser automation and the encrypted credential vault, prompt self-evolution, the RAG store, the circuit breaker
and task queue, and the server-rendered web dashboard.
