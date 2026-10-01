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
| sorting the catch (`digest`) | Claude Code (Sonnet 5.5) → Codex → Anthropic API (Sonnet 5.5) → DeepSeek → … |
| diving (`dream`) | Claude Code (Opus 5.5) → Codex → Anthropic API (Opus 5.5) → … |
| the lighthouse (`critique`) | Codex → Claude Code (Sonnet 5.5) → …, reordered so the keeper is not the diver when possible |
| the seabed (`dive`) | Claude Code (Sonnet 5.5) → Codex → Anthropic API (Sonnet 5.5) → … |

Claude Code runs each errand in an empty temporary folder with `--safe-mode`, `--tools ""`, its own system prompt and
no MCP servers; Codex runs in a read-only sandbox. An answer in the wrong shape is sent back once with the problems
listed; then the next crew member takes over. Every attempt goes in the crew's log.

## Life on the shore

`dun` runs the window, the menu-bar mark, the tide watcher (a daemon thread), the night watch (one worker thread) and
the watch keeper in one process. Opening it a second time brings the first window forward (`QLocalServer`).
`dun watch` runs only the tide watcher; the shore notices a living watcher and does not start a second.

Pages are redrawn from the sea floor on `refresh()`, which reads the full `api.state()`. Between redraws the window
polls `api.pulse()`, a handful of counts: a dive's progress moves in place, and when it surfaces the page is redrawn
and a notification says the dream has washed ashore. Everything the crew writes is shown as plain text, never as
markup.

The sea, the pebbles, the beads, the waterline, the shoal, the bottle and the fin are painted with `QPainter`.
There are no `QGraphicsEffect`s: they re-composite the linen behind them and leave seams.

## Sailing light

The app is meant to stay open on a laptop all day, so every wake-up has to earn its place. Measurements are in the
README.

- **Glances, not stares.** On macOS `sense/macnative.py` asks the window server and the accessibility API through
  `ctypes` (about 0.1 ms) instead of spawning `osascript` and `ioreg` for every sample; a browser's address is asked
  for only when the window title changes. Windows and X11 keep their samplers.
- **Pacing.** `Watcher.next_delay()` glances every `interval_seconds` while attention moves, doubles after four
  glances on the same subject and doubles again after eight (at most 60 s), checks only the idle clock at slack
  water, and on battery (`sense/power.py`, cached for two minutes) starts from 20 s and stretches to 80 s. Credit
  comes from real elapsed time, so pacing changes how fast a switch is noticed, never how much time is counted. The
  status row in the sea floor is rewritten only when what it says changes, or once a minute.
- **One clock** (`ui/motion.py`). Everything that moves subscribes to a single coarse timer with the pace it needs:
  the waterline 10 frames a second, the sidebar's tide line about 6, the bottle and the fin 12. The clock stops when
  no subscriber is visible, the window is minimised or the app is not in front; on battery (`motion = "auto"`) or
  in `calm` nothing moves more than 5 times a second; `off` leaves a still frame.
- **Repaint as little as possible.** The dream's sea and sand are painted once per size and theme into cached
  layers. Only the waterline band moves, and it is its own opaque child widget, so a frame repaints that strip
  and nothing above, below or behind it. Inside the strip the lines are hairline strokes and the fills are
  unsmoothed, because Qt's antialiased rasteriser costs in proportion to the area a shape spans.
- **Polling follows attention.** `MainWindow._pace()` polls every 5 s while the window is open, every 1.5 s during
  a dive and every 30 s when only the menu-bar mark needs news.
