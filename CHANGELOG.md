# Changelog

## 3.2.0 — read it all, and a day may dive twice

### Two dives in one day

- **Nothing of the first dive is lost.** Diving again used to write over the day's dive: its headline, reflection and
  undercurrent were gone, and the fish you had not touched were deleted (and their numbers handed to new fish, so a
  seabed report could turn up on the wrong one). Now every dive is kept. The Logbook still has one entry a day, led
  by the latest dive, and says how many dives the day held; on that day's shore the earlier dives fold away under
  *Open this dive*, whole, with their fish. Fish nobody touched *drift out*: they stay with their own dive and leave
  the shoal's "new"; you can still keep or follow them, and a fish's page links to the dive it really came from.
- **No fish twice.** A second dive is shown everything the day already caught, kept or threw back, so it brings back
  none of it.
- **The evening is not lost.** A dive by hand in the afternoon used to cancel the night's dive, and whatever came in
  after a day's dive never reached its currents. Now the night dives again when the day grew a good deal since (half
  an hour more, or a new bottle), and otherwise sorts the evening into its currents; the next morning catches up any
  day sorted before the rest of it came in.
- *Dive again* on the Shore no longer re-sorts a day that has nothing new. Past midnight the Shore turns to the new
  day. "Dive № n" counts dives in the order they were made, so a number never moves. The seabed's second search is
  now *Back to the seabed*, no longer a second *Dive again*.
- A dive made by hand in the daytime is headed *today's dream*, not *tonight's*. A fish's page leads back to the dive
  it came from, opening an earlier dive in place (that link, and a paper's title, had stopped responding to clicks).
- Memories from before open unchanged: the first start rebuilds one table and keeps every dive. Days dived before
  the upgrade keep their sorting; from then on, the hours after a dive are caught up.

### Nothing cut off

- **Nothing stops halfway.** A long dream no longer ends mid-sentence on the Shore. Qt measured a narrowed title or
  undercurrent at the panel's full width, came out short, and took the difference from the reflection; every narrowed
  text is now measured at the width it really has.
- **A pill to read it all.** A reflection longer than seven lines shows its first six, the last one fading, and a
  translucent *Read it all* pill that unfolds it in place (*Fold it back* folds it again). A fold you open stays open
  while the app runs. The same pill sits under a fish's long insight on the Shore, and under lists that used to stop
  without saying so: what washed ashore today, a current's logbook, the crew's usage. A card on the Shore shows four
  pieces of evidence and says how many more are on the fish's page.
- **Small windows.** Cards re-flow into fewer columns instead of pushing the page past the window's edge; a fish's
  side column, a current's buttons and the dream's facts wrap; long names end in "…" with the whole name on hover
  (shown as plain text, never as markup).
  The Strata keeps its words off the dots and lists them all on hover.
- A dive that failed when you asked for another one is explained under the dream, in full.
- `tests/test_layout.py` renders every page at four window sizes in English and Chinese, with texts two and three
  times as long and a day dived twice, and fails on anything cut off, overlapping or wider than the window.

## 3.1.1 — the mark holds, and sits like its neighbours

- Clicking the menu-bar mark no longer closes the app on newer macOS. Qt read the click count of the event that
  opened the mark's menu, and macOS aborts when that event is not a mouse event; on macOS the menu is now popped by
  the app itself.
- The Dock icon is a Liquid Glass icon (an Icon Composer `.icon`, compiled by `actool` into `Assets.car`, with an
  `.icns` for macOS before 26), the same size as Apple's own. The app also stopped replacing its Dock icon at launch
  with an edge-to-edge drawing, which is what made it look a size too big.

## 3.1.0 — a real Mac app, and a crew that reads

- **Digital Unconscious.app.** A drag-to-install disk image with its own icon: the app is called Digital Unconscious
  in the Dock and the menu bar (no longer "python"), asks for Accessibility itself, reopens from the Dock, quits with
  ⌘Q, and finds `claude` and your proxies even when opened from Finder. Built for Apple Silicon by
  `scripts/build_mac.py`.
- **Looking things up.** Claude may search the web while it dreams (concepts, how two ideas connect, how a topic
  lives in popular culture) and reads the open-access full texts at the seabed. Pages and downloads come only from
  big sites: encyclopedias, Reddit, Zhihu, Douban, Bilibili, Weibo, Chinese and English news, film and book sites,
  scholarly indexes and publishers. Everything happens in one sandboxed folder per errand, deleted afterwards.
  *Harbour → The crew* turns it off.
- Full texts download side by side within a minute; a seabed dive that reads four papers takes about a minute and a
  half.

## 3.0.2 — when the crew can't sail

- **Region guard.** While the connection is in mainland China, Claude, Codex, the Anthropic API and OpenAI stay
  ashore: nothing is sent to them. The connection is looked up right before every errand, through the same exit the
  CLIs would take, so a VPN counts and switching it off mid-dive is caught. Unsure (no network) means wait. DeepSeek,
  GLM, Kimi and local models still sail. *Harbour → The crew* can turn it off; `models.hold_regions` sets the regions.
- **Signed out, limited, offline.** Errors are recognised; the errand passes to the next crew member and the one in
  trouble pauses (5, 10, 20 minutes… at most two hours). A sign-in is asked for once in the menu bar, and noticed by
  itself afterwards through `claude auth status` / `codex login status`.
- **Waiting is not failing.** A crew that cannot sail no longer spends the night's attempts; the night watch queues
  the dive once someone can sail. The shore says in plain words why it is waiting.
- `dun doctor` shows where the connection is and whether Claude and Codex are signed in.
- The screenshot scripts no longer leave their borrowed seas in the temp folder.

## 3.0.1 — small, and it never forgets the main line

- **The main line is fenced off.** Currents and their day-by-day history, dreams, fish, seabed searches and your net
  are never touched by housekeeping; only the shark removes them. The policy lives in one place
  (`housekeeping.py`) and a test checks that a year of tidying leaves every main-line row and every undercurrent
  unchanged.
- **No day slips through.** The night watch now sorts any past day it missed (the laptop slept through the night)
  into its currents, so the history undercurrents are measured from has no holes before raw driftlines expire.
- **Details fade, quietly.** After two weeks the visits to one subject on one day merge into one row (same subjects,
  same totals for every dive); digests leave with their raw driftlines; old job records and the crew's call log are
  trimmed; the login log stays under 1 MB.
- **The shark really eats.** Forgetting gives the space back to the disk. Two years of heavy use now take about 24 MB
  instead of 36 MB, and feeding everything to the sharks leaves 0.1 MB instead of 36 MB.
- *Harbour → The sea* shows how much the harbour holds.
- A sea that starts in the menu bar builds no page until the window opens (54 MB instead of 65 MB at login), and Qt no
  longer searches for Chinese fonts the machine does not have.

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
