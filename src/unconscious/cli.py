"""`dun` — the command line, from the harbour.

  dun             open the app: it watches the tide by day and dives at night
  dun dream       dive into a day now, in the terminal
  dun jot TEXT    throw a thought into the sea in a bottle
  dun feed FILE   let a paper, notes or an ActivityWatch export wash ashore
  dun watch       only the tide watcher, for headless machines
  dun doctor      who is aboard, and what the watcher can see
  dun demo        open the app on a borrowed sea: three weeks of sample memory
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from datetime import date
from pathlib import Path

from unconscious import __version__

TTY = sys.stdout.isatty()
OCEAN = {
    "collision": "confluence", "orbit": "eddy", "return": "return tide", "surge": "swell", "seed": "driftwood",
    "gap": "channel", "steady": "main current", "fade": "ebb",
}


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if TTY else text


def dim(text: str) -> str:
    return _c("2", text)


def bold(text: str) -> str:
    return _c("1", text)


def accent(text: str) -> str:
    return _c("38;5;167", text)


def _app():
    from unconscious.app import App

    return App()


def _pid_alive(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False
    return True


def _other_watcher(app) -> int | None:
    from unconscious.api import sensor_status

    status = sensor_status(app)
    pid = int(status.get("pid") or 0)
    if status.get("state") not in {"stopped", "never"} and _pid_alive(pid):
        return pid
    return None


# ----------------------------------------------------------------- commands


def cmd_app(args: argparse.Namespace) -> int:
    try:
        from unconscious.ui.app import run
    except ImportError as exc:
        print(f"The desktop app needs PySide6 ({exc}). Install it with: pip install PySide6-Essentials")
        print("Everything else works from the command line: dun dream, dun jot, dun watch …")
        return 1
    return run(
        hidden=getattr(args, "hidden", False),
        watch=not getattr(args, "no_watch", False),
        auto_dream=not getattr(args, "no_dream", False),
    )


def cmd_watch(args: argparse.Namespace) -> int:
    from unconscious.sense.watcher import Watcher

    app = _app()
    other = _other_watcher(app)
    if other:
        print(f"A sensor is already running (pid {other}).")
        return 1
    stop = threading.Event()
    print(dim(f"observing every {app.settings.sense.interval_seconds}s · memory {app.home} · Ctrl+C to stop"))
    try:
        Watcher(app).run(stop)
    except KeyboardInterrupt:
        stop.set()
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    from unconscious.mind.digest import DigestError
    from unconscious.mind.dream import STEPS, DreamError, run_dream

    app = _app()
    day = args.day or date.today().isoformat()
    labels = {
        "gather": "wading in", "digest": "sorting the catch into currents", "signals": "reading the currents",
        "dream": "diving", "critique": "holding it up to the lighthouse", "save": "bringing it ashore",
    }

    def progress(step: str) -> None:
        index = STEPS.index(step) + 1 if step in STEPS else 0
        print(dim(f"  [{index}/{len(STEPS)}] {labels.get(step, step)}…"), flush=True)

    try:
        result = run_dream(app, day, progress, force_digest=args.redigest)
    except (DreamError, DigestError) as exc:
        from unconscious.ui.i18n import crew_message

        print(accent("  ✕ ") + crew_message(str(exc)))
        return 1
    _print_dream(app, day)
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


def _print_dream(app, day: str) -> None:
    from unconscious import api
    from unconscious.text import duration

    view = api.dream_view(app, day)
    if not view:
        print("No dream for that day.")
        return
    print()
    print("  " + dim(f"DIVE № {view['number']} · {day}"))
    print("  " + bold(view["title"]))
    print()
    print(_wrap(view["reflection"], 2))
    print()
    print("  " + accent("“") + view["undercurrent"] + accent("”"))
    for spark in view["sparks"]:
        print()
        print("  " + accent(OCEAN.get(spark["mechanism"], spark["mechanism"]).upper()) + dim(f"  #{spark['id']} · {spark['score']:.0f}"))
        print("  " + bold(spark["title"]))
        print(_wrap(spark["question"], 2))
        if spark.get("first_step"):
            print(dim("  first step: ") + spark["first_step"])
        if spark.get("evidence"):
            refs = "; ".join(f"{e['label'][:48]} ({duration(e['seconds'])})" if e.get("seconds") else e["label"][:48]
                             for e in spark["evidence"][:3])
            print(dim("  from: " + refs))
    print()


def _wrap(text: str, indent: int) -> str:
    import textwrap

    width = min(shutil.get_terminal_size((96, 20)).columns, 96) - indent
    return textwrap.fill(text or "", width=width, initial_indent=" " * indent, subsequent_indent=" " * indent)


def cmd_jot(args: argparse.Namespace) -> int:
    from unconscious.ingest import jot

    text = " ".join(args.text).strip() or sys.stdin.read().strip()
    jot(_app(), text)
    print(dim("  ✎ bottled. The tide will bring it to tonight's dream."))
    return 0


def cmd_feed(args: argparse.Namespace) -> int:
    from unconscious.ingest import IngestError, feed_path

    app = _app()
    status = 0
    for path in args.paths:
        try:
            result = feed_path(app, path)
        except IngestError as exc:
            print(accent("  ✕ ") + f"{path}: {exc}")
            status = 1
            continue
        count = result["traces"] if isinstance(result["traces"], int) else len(result["traces"])
        print(dim(f"  ❡ {Path(path).name}: {result['kind']} · {count} trace(s)"))
    return status


def cmd_import_aw(args: argparse.Namespace) -> int:
    from unconscious.ingest import IngestError, import_activitywatch

    app = _app()
    day = args.day or date.today().isoformat()
    try:
        result = import_activitywatch(app, day, args.url)
    except IngestError as exc:
        print(accent("  ✕ ") + str(exc))
        return 1
    print(dim(f"  imported {result['traces']} traces from ActivityWatch for {day}"))
    return 0


def cmd_today(args: argparse.Namespace) -> int:
    from unconscious import api
    from unconscious.text import duration

    app = _app()
    view = api.day_view(app, args.day or date.today().isoformat())
    print()
    print("  " + bold(view["day"]) + dim(f"  {duration(view['seconds'])} observed"))
    for subject in view["subjects"][:20]:
        tag = {"search": "⌕", "jot": "✎", "reading": "❡", "note": "·"}.get(subject["kind"], " ")
        when = duration(subject["seconds"]) if subject["seconds"] else ""
        thread = dim(f"  [{subject['thread']}]") if subject.get("thread") else ""
        print(f"  {tag} {when:>7}  {subject['label'][:70]}{thread}")
    print()
    return 0


def cmd_threads(args: argparse.Namespace) -> int:
    from unconscious import api
    from unconscious.text import duration

    app = _app()
    view = api.threads_view(app)
    blocks = " ▁▂▃▄▅▆▇█"
    print()
    for thread in view["threads"][:30]:
        series = thread["series"][-21:]
        peak = max(series) or 1
        spark = "".join(blocks[min(8, int(8 * value / peak + 0.99))] if value else "·" for value in series)
        flags = " ".join(OCEAN.get(k, k) for k in thread["signals"])
        print(f"  {dim(spark)}  {thread['name'][:40]:<40} {duration(thread['total']):>7}  {accent(flags)}")
    if view["signals"]:
        print()
        for signal in view["signals"]:
            print("  " + signal["text"])
    print()
    return 0


def cmd_sparks(args: argparse.Namespace) -> int:
    app = _app()
    for spark in app.store.sparks(status=args.status, limit=args.limit):
        print(f"  #{spark['id']:<4} {dim(spark['day'])} {accent(OCEAN.get(spark['mechanism'], spark['mechanism'])[:11].ljust(11))} "
              f"{spark['title'][:72]} {dim(spark['status'])}")
    return 0


def cmd_dive(args: argparse.Namespace) -> int:
    from unconscious.mind.dive import DiveError, run_dive

    app = _app()
    try:
        result = run_dive(app, args.spark, lambda s: print(dim(f"  {s}…"), flush=True))
    except DiveError as exc:
        print(accent("  ✕ ") + str(exc))
        return 1
    dive = app.store.dives(args.spark)[0]
    report, papers = dive["report"], dive["papers"]
    print()
    print("  " + accent(str(report.get("verdict", "")).upper()) + "  " + bold(report.get("sharpened_question", "")))
    print(_wrap(report.get("summary", ""), 2))
    for item in report.get("known", []):
        refs = "".join(f"[{r}]" for r in item["refs"])
        print(_wrap(f"• {item['point']} {refs}", 2))
    print(dim("  gap: ") + report.get("gap", ""))
    for index, paper in enumerate(papers, 1):
        if index in report.get("cited", []):
            print(dim(f"  [{index}] {paper['title'][:80]} ({paper.get('year') or 'n.d.'}) {paper.get('url', '')}"))
    print()
    _ = result
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from unconscious.llm.base import LLMRequest
    from unconscious.sense.sensors import SensorError, make_sensor

    app = _app()
    print()
    print("  " + bold("Digital Unconscious") + dim(f" v{__version__} · Python {platform.python_version()} · {platform.system()}"))
    print(dim(f"  home {app.home}"))
    print()
    print("  " + bold("Sensor"))
    sensor = make_sensor(app.settings.sense.capture_urls)
    try:
        sample = sensor.sample()
        print(f"    ✓ foreground app: {sample.app or '—'}")
        print(f"    {'✓' if sample.title else '·'} window titles {'readable' if sample.title else 'not visible (needs Accessibility permission?)'}")
        print(f"    ✓ idle time: {sample.idle_seconds:.0f}s" if sensor.capabilities.idle else "    · idle time unavailable")
    except SensorError as exc:
        print(accent("    ✕ ") + str(exc))
    if sensor.capabilities.note:
        print(dim("    " + sensor.capabilities.note))
    print()
    print("  " + bold("Models"))
    described = app.router.describe()
    for name, ok in described["available"].items():
        if name == "ollama" and not ok:
            continue
        print(f"    {'✓' if ok else '·'} {name}")
    print()
    for role, route in described["routes"].items():
        chain = " → ".join(route["chain"]) or accent("nothing available")
        print(f"    {role:<9} {chain}")
    print()
    if app.settings.models.region_guard:
        country = app.router.region.current(max_age=0)
        held = country == "?" or country in app.settings.models.hold_regions
        if country == "?":
            print(accent("    ✕ ") + "could not tell where the connection is: Claude and Codex wait ashore")
        else:
            print(f"    {accent('✕') if held else '✓'} connection in {country}: "
                  + ("Claude, Codex, Anthropic and OpenAI stay ashore" if held else "the whole crew may sail"))
    for name in ("claude", "codex"):
        provider = app.router.providers.get(name)
        if described["available"].get(name) and hasattr(provider, "signed_in"):
            signed = provider.signed_in()
            if signed is False:
                print(accent("    ✕ ") + f"{name} is not signed in: run `{'claude auth login' if name == 'claude' else 'codex login'}`")
            elif signed:
                print(f"    ✓ {name} signed in")
    for name, trouble in (described.get("troubles") or {}).items():
        print(accent("    ! ") + f"{name} is resting after {trouble['kind']} ({trouble['strikes']}×): {trouble['detail'][:120]}")
    if args.ping:
        print()
        print("  " + bold("Ping"))
        for role in ("digest", "dream"):
            result = app.router.call(LLMRequest(role=role, system="Reply with the single word: pong", prompt="ping", max_tokens=20, effort="low"))
            mark = "✓" if result.ok else accent("✕")
            detail = f"{result.label} · {result.ms} ms" if result.ok else result.error[:240]
            print(f"    {mark} {role:<9} {detail}")
    print()
    if not any(route["chain"] for route in described["routes"].values()):
        print("  No model is available. Install and sign in to Claude Code (`claude`) or Codex (`codex`),")
        print("  or export an API key such as DEEPSEEK_API_KEY.")
        print()
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    app = _app()
    if args.action == "path":
        print(app.settings_path)
        return 0
    if args.action == "edit":
        editor = os.environ.get("EDITOR") or ("open" if platform.system() == "Darwin" else "notepad" if platform.system() == "Windows" else "nano")
        return subprocess.call([editor, str(app.settings_path)])
    if args.action == "set":
        if not args.key or args.value is None:
            print("usage: dun config set section.key value")
            return 2
        section, _, key = args.key.partition(".")
        changed = app.update_settings({section: {key: args.value}})
        print(dim("  updated " + ", ".join(changed)) if changed else "  nothing changed (unknown key or same value)")
        return 0
    print(app.settings_path.read_text(encoding="utf-8"))
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    app = _app()
    if args.everything:
        answer = input("The sharks will eat every trace, current, dive and fish. Type 'feed the sharks' to let them: ")
        if answer.strip() != "feed the sharks":
            print("The sharks went hungry. Nothing was eaten.")
            return 1
        app.store.forget_everything()
        print("The sharks have eaten everything.")
        return 0
    if args.day:
        print(f"The shark ate {app.store.forget_day(args.day)} traces from {args.day}.")
        return 0
    print("usage: dun forget --day YYYY-MM-DD | --everything")
    return 2


def cmd_demo(args: argparse.Namespace) -> int:
    from unconscious.demo import seed_demo

    home = Path(args.home or Path.home() / ".digital-unconscious-demo").expanduser()
    os.environ["DUN_HOME"] = str(home)
    summary = seed_demo(home, language=args.language, reset=True)
    print(dim(f"  seeded {summary['days']} days, {summary['traces']} traces, {summary['threads']} threads, "
              f"{summary['dreams']} dreams into {home}"))
    if args.no_open:
        print(f"  Open it with: dun --home {home}")
        return 0
    try:
        from unconscious.ui.app import run
    except ImportError as exc:
        print(f"The desktop app needs PySide6 ({exc}).")
        return 1
    return run(demo=True, watch=False, auto_dream=False)


def cmd_service(args: argparse.Namespace) -> int:
    from unconscious import service

    return service.handle(args.action)


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dun", description="Digital Unconscious — it watches the tide by day and dives at night.")
    parser.add_argument("--home", help="harbour folder (default ~/.digital-unconscious, or $DUN_HOME)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"dun {__version__}")
    sub = parser.add_subparsers(dest="command")

    for name, helptext in (("app", "open the app on the shore (the default)"), ("up", "same as `app`")):
        app_cmd = sub.add_parser(name, help=helptext)
        app_cmd.add_argument("--hidden", action="store_true", help="start in the menu bar without opening the window")
        app_cmd.add_argument("--no-watch", action="store_true", help="leave the tide watcher ashore")
        app_cmd.add_argument("--no-dream", action="store_true", help="do not dive on its own at night")
        app_cmd.set_defaults(func=cmd_app)

    sub.add_parser("watch", help="only the tide watcher, no window").set_defaults(func=cmd_watch)

    dream = sub.add_parser("dream", help="dive into a day now")
    dream.add_argument("--day", help="YYYY-MM-DD (default today)")
    dream.add_argument("--redigest", action="store_true", help="sort the catch into currents again first")
    dream.add_argument("--json", action="store_true")
    dream.set_defaults(func=cmd_dream)

    jot = sub.add_parser("jot", help="throw a thought into the sea in a bottle")
    jot.add_argument("text", nargs="*")
    jot.set_defaults(func=cmd_jot)

    feed = sub.add_parser("feed", help="let documents or exports wash ashore")
    feed.add_argument("paths", nargs="+")
    feed.set_defaults(func=cmd_feed)

    aw = sub.add_parser("import-aw", help="bring in a day from a running ActivityWatch")
    aw.add_argument("--day")
    aw.add_argument("--url", default="http://localhost:5600")
    aw.set_defaults(func=cmd_import_aw)

    today = sub.add_parser("today", help="what the tide left today")
    today.add_argument("--day")
    today.set_defaults(func=cmd_today)

    sub.add_parser("threads", help="the currents and tonight's undercurrents").set_defaults(func=cmd_threads)

    sparks = sub.add_parser("sparks", help="the shoal: every fish the dives kept")
    sparks.add_argument("--status", choices=["new", "kept", "pursuing", "dismissed", "done", "drifted"])
    sparks.add_argument("--limit", type=int, default=30)
    sparks.set_defaults(func=cmd_sparks)

    dive = sub.add_parser("dive", help="dive to the seabed for prior work on a fish")
    dive.add_argument("spark", type=int)
    dive.set_defaults(func=cmd_dive)

    doctor = sub.add_parser("doctor", help="who is aboard, and what the tide watcher can see")
    doctor.add_argument("--ping", action="store_true", help="send each crew member a tiny real errand")
    doctor.set_defaults(func=cmd_doctor)

    config = sub.add_parser("config", help="show or change the harbour settings")
    config.add_argument("action", nargs="?", choices=["show", "set", "path", "edit"], default="show")
    config.add_argument("key", nargs="?")
    config.add_argument("value", nargs="?")
    config.set_defaults(func=cmd_config)

    forget = sub.add_parser("forget", help="let the shark eat a day (or feed everything to the sharks)")
    forget.add_argument("--day")
    forget.add_argument("--everything", action="store_true")
    forget.set_defaults(func=cmd_forget)

    demo = sub.add_parser("demo", help="open the app on a borrowed sea: three weeks of sample memory")
    demo.add_argument("--home")
    demo.add_argument("--language", choices=["en", "zh"], default="en")
    demo.add_argument("--no-open", action="store_true", help="only seed the sample memory")
    demo.set_defaults(func=cmd_demo)

    service = sub.add_parser("service", help="put the watcher out to sea at every login")
    service.add_argument("action", choices=["install", "uninstall", "status"])
    service.set_defaults(func=cmd_service)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.home:
        os.environ["DUN_HOME"] = str(Path(args.home).expanduser())
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not getattr(args, "func", None):
        return cmd_app(args)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
