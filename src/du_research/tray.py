"""System tray application for Digital Unconscious.

Shows a small icon in the bottom-right corner. Click to:
- View today's briefing
- Browse idea backlog
- Open settings / config
- Trigger "Research best idea now"
- Open the web dashboard
- Start/stop the observation service
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from du_research.config import AppConfig, load_config


def _workspace(config: AppConfig) -> Path:
    return Path(config.pipeline.workspace_dir).resolve()


def _latest_briefing_path(config: AppConfig) -> Path | None:
    daily_dir = _workspace(config) / "daily"
    if not daily_dir.exists():
        return None
    for d in sorted(daily_dir.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith("cycle_"):
            date = d.name.replace("cycle_", "")
            path = d / f"briefing_{date}.md"
            if path.exists():
                return path
    return None


def _top_idea(config: AppConfig) -> dict[str, Any] | None:
    backlog = _workspace(config) / "ideas" / "idea_backlog.jsonl"
    if not backlog.exists():
        return None
    best = None
    best_score = -1.0
    for line in backlog.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        score = float(obj.get("total_score", 0))
        if score > best_score:
            best_score = score
            best = obj
    return best


def _idea_count(config: AppConfig) -> int:
    backlog = _workspace(config) / "ideas" / "idea_backlog.jsonl"
    if not backlog.exists():
        return 0
    return sum(1 for line in backlog.read_text(encoding="utf-8").splitlines() if line.strip())


def _run_command(*args: str) -> None:
    """Run a du CLI command in a background subprocess."""
    python = sys.executable
    cmd = [python, "-m", "du_research.cli", *args]
    subprocess.Popen(
        cmd,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


def _create_icon_image():
    """Create a small icon image for the system tray."""
    from PIL import Image, ImageDraw
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Dark circle background
    draw.ellipse([2, 2, size - 2, size - 2], fill=(26, 29, 39, 255))
    # "DU" text approximated with shapes — a lightbulb-like symbol
    # Bulb
    draw.ellipse([18, 8, 46, 36], fill=(108, 138, 255, 255))
    # Stem
    draw.rectangle([24, 34, 40, 46], fill=(108, 138, 255, 255))
    # Base lines
    draw.rectangle([22, 46, 42, 50], fill=(225, 228, 235, 255))
    draw.rectangle([24, 51, 40, 55], fill=(225, 228, 235, 255))
    return img


def run_tray(config: AppConfig | None = None, config_path: str | None = None):
    """Launch the system tray icon."""
    try:
        import pystray  # type: ignore[import-untyped]
    except ImportError:
        print("System tray requires pystray: pip install pystray Pillow")
        print("Falling back to web dashboard...")
        from du_research.dashboard import run_dashboard
        run_dashboard(config or load_config(config_path))
        return

    config = config or load_config(config_path)

    def open_dashboard(icon, item):
        _run_command("dashboard", "--no-open")
        webbrowser.open("http://localhost:9830")

    def open_briefing(icon, item):
        path = _latest_briefing_path(config)
        if path:
            webbrowser.open(f"http://localhost:9830/briefing")
        else:
            _notify(icon, "No briefings yet", "Run 'du daily' to generate your first briefing.")

    def run_daily_now(icon, item):
        _notify(icon, "Daily cycle started", "Generating ideas from today's observations...")
        threading.Thread(target=lambda: _run_command("daily"), daemon=True).start()

    def research_best(icon, item):
        idea = _top_idea(config)
        if idea:
            title = idea.get("title", "Unknown")
            _notify(icon, "Research started", f"Researching: {title[:60]}")
            threading.Thread(target=lambda: _run_command("research", "--auto"), daemon=True).start()
        else:
            _notify(icon, "No ideas yet", "Run a daily cycle first to generate ideas.")

    def show_status(icon, item):
        n = _idea_count(config)
        briefing = _latest_briefing_path(config)
        date = briefing.parent.name.replace("cycle_", "") if briefing else "none"
        _notify(icon, "Digital Unconscious",
                f"Ideas in backlog: {n}\nLatest briefing: {date}")

    def open_config(icon, item):
        config_file = Path("config/pipeline.toml").resolve()
        if sys.platform == "win32":
            subprocess.Popen(["notepad", str(config_file)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(config_file)])
        else:
            subprocess.Popen(["xdg-open", str(config_file)])

    def quit_app(icon, item):
        icon.stop()

    def _notify(icon, title: str, message: str):
        try:
            icon.notify(message, title)
        except Exception:
            pass  # notification not supported on all platforms

    menu = pystray.Menu(
        pystray.MenuItem("Open Dashboard", open_dashboard, default=True),
        pystray.MenuItem("Today's Briefing", open_briefing),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Run Daily Cycle Now", run_daily_now),
        pystray.MenuItem("Research Best Idea", research_best),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Status", show_status),
        pystray.MenuItem("Edit Config", open_config),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )

    icon = pystray.Icon(
        name="digital-unconscious",
        icon=_create_icon_image(),
        title="Digital Unconscious",
        menu=menu,
    )
    icon.run()
