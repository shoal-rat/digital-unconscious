# Digital Unconscious — notes for contributors and coding agents

## Product boundary

A native desktop app for one person on one machine. It records attention, sorts it into threads, computes patterns,
and dreams up grounded ideas. Keep it that way:

- No accounts, sync, servers, web dashboards or HTML-in-a-wrapper UIs. The interface is PySide6.
- No browser automation, no stored credentials, no autonomous publishing.
- No screenshots or keystroke capture. Window metadata only.
- Model inference may be remote; never claim otherwise in UI or docs.

## Layout

- `src/unconscious/`: engine (standard library only) and `ui/` (PySide6).
- `tests/`: `unittest`; run offline with the fake model. `tests/helpers.py` has `TempApp`.
- `scripts/snapshots.py`: renders every page to PNG offscreen. Use it to check UI changes.
- `scripts/build_mac.py`: builds `Digital Unconscious.app` and its `.dmg` (`pip install -e ".[mac]"`), around
  `packaging/macos/launcher.py`.
- `scripts/readme_art.py`: repaints the README pictures (`docs/assets/en`, `docs/assets/zh`, the sea loops).
  README.md (English) and README.zh-CN.md (Chinese) are kept in step; change both.

## Rules that keep the output honest

- The model never invents provenance: inputs carry `S`/`T` references and code rejects anything else.
- Patterns are computed in `mind/signals.py`, not asked of a model.
- The critique scores; code applies weights, taste and diversity.
- All model text is rendered as plain text (`QLabel` with `PlainText`, or the `Para` widget).
- The main line (`housekeeping.MAIN_LINE`: currents, their daily history, dreams, fish, seabed searches, the net) is
  never deleted except by the shark (`forget_day`, `forget_everything`). Anything that fades is decided in
  `housekeeping.tidy()` and must leave what a dive reads unchanged; a new table goes on one side or the other, and
  `tests/test_housekeeping.py` gets a line for it.
- What a research errand may reach is decided only in `llm/research.py`: `SHELF` holds big platforms, never
  personal hosting (`OFF_SHELF`), and the sandbox settings stay strict (no unsandboxed retries, no reads outside the
  errand's folder). Full texts live in the errand's temporary folder and leave with it.
- Every errand goes through `Router.call`, which applies the region guard and crew pauses. Nothing calls a
  provider's `complete` directly outside tests, and nothing about the network or a CLI is asked on the UI thread.
- Prompts and schemas live only in `mind/prompts.py`. Schemas are strict (every property required, no extras),
  because Codex and the Anthropic structured-output API require it.

## The ocean vocabulary

People read the app as a small sea; the code keeps plain names. Keep both consistent:
Shore (today), Currents (threads), Shoal (sparks, each one a fish), Logbook (journal), Harbour (settings),
tide watcher (sensor), driftline (traces), slack water / fog / at anchor (idle / quiet app / paused), bottle (jot),
washed in (fed document), eddy / return tide / swell / driftwood / confluence / main current / ebb / channel
(orbit / return / surge / seed / collision / steady / fade / gap), the lighthouse (critique), the crew (models),
the seabed (literature dive), the shark (forgetting). New strings follow the same voice in English and Chinese;
avoid computer words like "filter", "purge" or "sync" in anything a person reads.

## Interface conventions

- Colours, fonts and the stylesheet come from `ui/theme.py`. Never hard-code colours in pages; add a token.
- The dream panel is the sea: azure in the morning theme, deep blue in the evening theme. Motion is slow and
  meaningful: the waterline, a shoal with one fish per idea, a bobbing bottle, a passing fin.
- Paint with `QPainter`; avoid `QGraphicsEffect` (it re-composites the textured page and leaves seams).
- Qt ignores a `border-radius` larger than half a widget's height: size pills with `min-height`.
- Strings go through `ui/i18n.py` in both English and Chinese.

## Verify before publishing

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
ruff check src tests scripts
python scripts/snapshots.py snapshots/ && python scripts/snapshots.py snapshots/ --dark --zh
python -m build
```

For provider changes, add a test with a stand-in executable (see `tests/test_llm.py`) and, when the CLI is signed
in, run `dun doctor --ping`.
