# The ecosystem, in code

Digital Unconscious is imagined as a small sea, and the code is laid out along the same shoreline. People read the
ocean names; the code keeps plain ones, so a contributor always knows both.

<p align="center"><img src="assets/architecture.svg" alt="The shore (ui), the tide watcher (sense, ingest), the sea floor (store), the deep (mind), the crew (llm router) and the night watch (jobs)" width="880"></p>

Everything runs in one process on one machine. Nothing lives on a server.

## Who lives where

| In the sea | In the code | What it does |
| --- | --- | --- |
| **The shore** | `ui/` | The PySide6 window, the menu-bar mark, the bottle and shark dialogs, the painted sea, pebbles and beads |
| what the shore asks of the sea | `api.py` | View models: plain dicts the shore and the terminal render |
| **The tide watcher** | `sense/sensors.py`, `sense/watcher.py` | Glances at the front window every few seconds (macOS osascript and ioreg, Win32 via ctypes, X11 xdotool); turns glances into a driftline with real time; stops counting at slack water; rolls over at midnight; lets the tide wash old driftlines away |
| fog and time-only waters | `sense/privacy.py` | What is never recorded (quiet apps and sites, private windows) and what is blurred (addresses, numbers, tokens) |
| reading the driftline | `sense/subjects.py` | Raw glance → subject: tidy titles, find searches, recognise papers and code projects |
| things that wash in | `ingest.py` | Bottles (jots), papers and notes, plain text logs, JSON records, ActivityWatch |
| **The sea floor** | `store.py` | One SQLite file (WAL). Short-lived connections, so the watcher, the shore and the terminal never block each other |
| the harbour settings | `config.py` | One TOML file, edited by the Harbour page and by hand |
| **The deep** | `mind/` | Everything that happens below the surface at night |
| sorting the catch | `mind/digest.py` | A day's subjects → topics → currents; checks every reference, joins existing currents instead of redrawing them, writes the day atomically |
| reading the currents | `mind/signals.py` | Eddy, return tide, swell, driftwood, confluence, main current, ebb: arithmetic over weeks of tides |
| the dive | `mind/dream.py` | Context → dream → checking where each fish rose → the lighthouse → weighing → keeping a few → bringing it ashore |
| the lighthouse | inside `mind/dream.py` | A second crew member scores each fish and writes a warning; code applies the weights |
| your net | `mind/taste.py` | Counts of what you kept and threw back; a bounded nudge to ranking and a paragraph for the next dive |
| the seabed | `mind/dive.py`, `scholar.py` | OpenAlex (then arXiv) soundings plus one crew call; citations must point at something that came up |
| what the crew is told | `mind/prompts.py` | Every prompt and answer shape, in one place |
| **The crew** | `llm/router.py`, `llm/providers.py` | Each job names the hand it prefers; the first one aboard takes it, the next takes over if they falter |
| a paper crew | `llm/fake.py` | A deterministic stand-in so the tests and `DUN_FAKE_LLM=1` sail offline |
| **The night watch** | `jobs.py` | One diver at a time (subscription CLIs are never sent down together); a watch keeper sends the night's dive down after dream time, and yesterday's if the machine was asleep |
| a borrowed sea | `demo.py` | Three weeks of generated tides with hand-written dives, for `dun demo` |

## What settles on the sea floor

<p align="center"><img src="assets/data-model.svg" alt="Driftline, tides, currents, undercurrents, dives, fish, seabed searches, your net and taste" width="880"></p>

The **driftline** (`traces`) is the only minute-by-minute record of a person, and the tide washes it away after
`sense.retention_days`. Everything else is a shape, and settles:

| In the sea | Table | Notes |
| --- | --- | --- |
| driftline | `traces` | one row per stretch of attention on one subject |
| tides | `thread_days`, `subject_threads` | how much time each current carried each day, and which subjects belong to it |
| currents | `threads` | buoyed (`pinned`), becalmed (`muted`) or joined to another (`merged`) |
| undercurrents | — | measured on demand from the tides, never stored |
| dives | `dreams`, `digests` | one per night: headline, reflection, undercurrent, what was measured |
| fish | `sparks` | each keeps a copy of where it rose, so it outlives the driftline |
| seabed | `dives` | the report and what came up |
| your net | `events` | keep, swim after, throw back (with a reason) |
| the crew's log | `llm_calls` | who took each job, how long it took, what it cost |
| the night watch's log | `jobs` | queued, running, done or failed, and the current step |

A *subject* is a day's driftline gathered by `subject_key`. Keys are made to survive trivial title changes (unread
counters, unsaved markers, page numbers) and to gather an editor's windows by project.

## Keeping the fish honest

- The crew is told about the day with short tags: `S7` for something that washed up today, `T3` for a current.
  Every answer must use them, and code throws back anything that does not.
- Sorting drops tags it never offered, gives each subject to one topic, follows joined currents, and refuses to draw
  a current that already exists under a slightly different name.
- A fish is thrown back if it rose from an unknown movement of the water, cites nothing it was given, or is a near
  copy of one caught in the last three weeks.
- The lighthouse only scores. Weights, the *common fish* penalty, your net (at most ±15%) and the rule against
  catching the same fish twice are code.
- A seabed citation survives only if it points at a work that actually came up.

## The crew's rota

| Job | Who takes it first (the first one aboard wins) |
| --- | --- |
| sorting the catch (`digest`) | DeepSeek → Claude Code (Haiku) → Codex → Anthropic API (Haiku 4.5) → … |
| diving (`dream`) | Claude Code (Opus) → Codex → Anthropic API (Opus 5.5) → … |
| the lighthouse (`critique`) | Codex → Claude Code (Sonnet) → …, reordered so the keeper is not the diver when possible |
| the seabed (`dive`) | Claude Code (Sonnet) → Codex → Anthropic API (Opus 5.5) → … |

Claude Code runs each errand in an empty temporary folder with `--safe-mode`, `--tools ""`, its own system prompt and
no MCP servers; Codex runs in a read-only sandbox. An answer in the wrong shape is sent back once with the problems
listed; then the next crew member takes over. Every attempt goes in the crew's log.

## Life on the shore

`dun` runs the window, the menu-bar mark, the tide watcher (a daemon thread), the night watch (one worker thread) and
the watch keeper in one process. Opening it a second time brings the first window forward (`QLocalServer`).
`dun watch` runs only the tide watcher; the shore notices a living watcher and does not start a second.

Pages are redrawn from the sea floor on `refresh()`. The window checks `api.state()` every two seconds: a dive's
progress moves in place, and when it surfaces the page is redrawn and a notification says the dream has washed
ashore. Everything the crew writes is shown as plain text, never as markup.

The sea, the pebbles, the beads, the waterline, the shoal, the bottle and the fin are painted with `QPainter`.
There are no `QGraphicsEffect`s: they re-composite the linen behind them and leave seams.
