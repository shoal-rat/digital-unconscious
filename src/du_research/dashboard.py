"""Local web dashboard for Digital Unconscious.

A lightweight, zero-dependency web server that serves the daily briefings,
idea backlog, system status, and learning insights in a clean browser UI.

Usage:
    du dashboard          # opens http://localhost:9830
    du dashboard --port 8080
"""
from __future__ import annotations

import json
import sys
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from du_research.config import AppConfig, load_config


def _workspace(config: AppConfig) -> Path:
    return Path(config.pipeline.workspace_dir).resolve()


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _list_daily_cycles(workspace: Path) -> list[dict[str, Any]]:
    daily_dir = workspace / "daily"
    if not daily_dir.exists():
        return []
    cycles = []
    for d in sorted(daily_dir.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith("cycle_"):
            date = d.name.replace("cycle_", "")
            briefing_path = d / f"briefing_{date}.md"
            ideas_path = d / "ideas_all.json"
            ideas_included_path = d / "ideas_included.json"
            n_ideas = 0
            n_included = 0
            if ideas_path.exists():
                try:
                    n_ideas = len(json.loads(ideas_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
            if ideas_included_path.exists():
                try:
                    n_included = len(json.loads(ideas_included_path.read_text(encoding="utf-8")))
                except Exception:
                    pass
            cycles.append({
                "date": date,
                "dir": str(d),
                "has_briefing": briefing_path.exists(),
                "ideas_total": n_ideas,
                "ideas_included": n_included,
            })
    return cycles[:30]


def _load_briefing(workspace: Path, date: str) -> str | None:
    path = workspace / "daily" / f"cycle_{date}" / f"briefing_{date}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _load_idea_backlog(workspace: Path) -> list[dict[str, Any]]:
    backlog = workspace / "ideas" / "idea_backlog.jsonl"
    if not backlog.exists():
        return []
    ideas = []
    for line in backlog.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ideas.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return ideas


def _load_learning_status(workspace: Path) -> dict[str, Any]:
    model = _load_json(workspace / "learning" / "human_idea_model.json") or {}
    outcomes = _load_json(workspace / "learning" / "run_outcomes.json") or {}
    changes_path = workspace / "learning" / "learning_changes.md"
    changes = changes_path.read_text(encoding="utf-8") if changes_path.exists() else ""
    return {"model": model, "outcomes": outcomes, "changes": changes}


def _service_status(workspace: Path) -> dict[str, Any]:
    status = _load_json(workspace / "service" / "status.json") or {}
    state = _load_json(workspace / "service" / "service_state.json") or {}
    return {"status": status, "state": state}


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_CSS = """
:root { --bg: #0f1117; --card: #1a1d27; --border: #2a2d37; --text: #e1e4eb;
  --muted: #8b8fa3; --accent: #6c8aff; --accent2: #ff6c8a; --green: #4ade80; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont,
  'Segoe UI', system-ui, sans-serif; line-height: 1.6; }
.container { max-width: 900px; margin: 0 auto; padding: 20px; }
nav { display: flex; gap: 12px; padding: 16px 0; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
nav a { color: var(--muted); text-decoration: none; padding: 8px 16px; border-radius: 8px;
  font-size: 14px; font-weight: 500; transition: all 0.2s; }
nav a:hover, nav a.active { color: var(--text); background: var(--card); }
h1 { font-size: 24px; font-weight: 600; margin-bottom: 8px; }
h2 { font-size: 18px; font-weight: 600; margin: 24px 0 12px; color: var(--accent); }
h3 { font-size: 15px; font-weight: 600; margin: 16px 0 8px; }
.subtitle { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 20px; margin-bottom: 16px; }
.card:hover { border-color: var(--accent); }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 24px; }
.stat { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px; text-align: center; }
.stat .value { font-size: 28px; font-weight: 700; color: var(--accent); }
.stat .label { font-size: 12px; color: var(--muted); margin-top: 4px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; }
.badge-include { background: rgba(74,222,128,0.15); color: var(--green); }
.badge-hold { background: rgba(108,138,255,0.15); color: var(--accent); }
.badge-discard { background: rgba(255,108,138,0.15); color: var(--accent2); }
.idea-row { display: flex; justify-content: space-between; align-items: center; padding: 12px 0;
  border-bottom: 1px solid var(--border); }
.idea-row:last-child { border-bottom: none; }
.idea-title { font-weight: 500; font-size: 14px; }
.idea-score { font-size: 14px; font-weight: 600; }
.briefing { line-height: 1.8; font-size: 15px; }
.briefing h1 { font-size: 22px; margin: 24px 0 8px; }
.briefing h2 { font-size: 18px; margin: 20px 0 8px; }
.briefing h3 { font-size: 15px; }
.briefing ul, .briefing ol { margin-left: 24px; margin-bottom: 12px; }
.briefing li { margin-bottom: 4px; }
.briefing hr { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
.briefing strong { color: var(--accent); }
.briefing em { color: var(--muted); }
.briefing code { background: var(--card); padding: 2px 6px; border-radius: 4px; font-size: 13px; }
.cycle-link { display: block; text-decoration: none; color: inherit; }
.empty { text-align: center; padding: 60px 20px; color: var(--muted); }
.empty p { margin-top: 12px; font-size: 14px; }
a { color: var(--accent); }
.tag { display: inline-block; background: rgba(108,138,255,0.1); color: var(--accent);
  padding: 2px 8px; border-radius: 4px; font-size: 12px; margin: 2px; }
pre { background: var(--card); padding: 16px; border-radius: 8px; overflow-x: auto;
  font-size: 13px; line-height: 1.5; border: 1px solid var(--border); }
input[type=text], textarea, select { background: var(--card); color: var(--text); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 14px; font-size: 14px; width: 100%; margin-bottom: 12px; }
input[type=text]:focus, textarea:focus { border-color: var(--accent); outline: none; }
button { background: var(--accent); color: white; border: none; border-radius: 8px; padding: 12px 24px;
  font-size: 14px; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
button:hover { opacity: 0.85; }
label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 4px; }
.form-group { margin-bottom: 20px; }
.success { background: rgba(74,222,128,0.15); color: var(--green); padding: 12px 16px; border-radius: 8px; margin-bottom: 16px; }
"""


def _md_to_html(md: str) -> str:
    """Minimal Markdown → HTML (no dependencies)."""
    import re
    lines = md.split("\n")
    html_lines: list[str] = []
    in_list = False
    in_code = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                html_lines.append("</pre>")
                in_code = False
            else:
                html_lines.append("<pre>")
                in_code = True
            continue
        if in_code:
            html_lines.append(line)
            continue

        if stripped.startswith("---"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("<hr>")
            continue

        if stripped.startswith("# "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h1>{_inline_md(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{_inline_md(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{_inline_md(stripped[4:])}</h3>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{_inline_md(stripped[2:])}</li>")
        elif re.match(r"^\d+\.\s", stripped):
            content = re.sub(r"^\d+\.\s*", "", stripped)
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{_inline_md(content)}</li>")
        elif stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{_inline_md(stripped)}</p>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False

    if in_list:
        html_lines.append("</ul>")
    if in_code:
        html_lines.append("</pre>")
    return "\n".join(html_lines)


def _inline_md(text: str) -> str:
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def _page(title: str, content: str, active: str = "") -> str:
    nav_items = [
        ("", "Dashboard"),
        ("briefing", "Briefings"),
        ("ideas", "Idea Backlog"),
        ("learning", "Learning"),
        ("status", "Status"),
    ]
    nav_html = ""
    for href, label in nav_items:
        cls = ' class="active"' if href == active else ""
        nav_html += f'<a href="/{href}"{cls}>{label}</a>'

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Digital Unconscious</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><circle cx='32' cy='32' r='28' fill='%231a1d27'/><circle cx='32' cy='32' r='10' fill='%236c8aff'/><circle cx='32' cy='32' r='5' fill='%23b4c8ff'/></svg>">
<style>{_CSS}</style>
</head><body>
<div class="container">
<nav>{nav_html}</nav>
{content}
</div></body></html>"""


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class DashboardHandler(BaseHTTPRequestHandler):
    config: AppConfig

    def log_message(self, format, *args):
        pass  # silent

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        workspace = _workspace(self.config)

        # Redirect to setup if first run
        setup_done = (workspace / "setup" / "user_settings.json").exists()
        if not setup_done and path == "/":
            self._serve_setup(workspace)
            return

        if path == "/":
            self._serve_dashboard(workspace)
        elif path == "/setup":
            self._serve_setup(workspace)
        elif path == "/briefing":
            params = parse_qs(parsed.query)
            date = params.get("date", [None])[0]
            self._serve_briefing(workspace, date)
        elif path == "/ideas":
            self._serve_ideas(workspace)
        elif path == "/learning":
            self._serve_learning(workspace)
        elif path == "/status":
            self._serve_status(workspace)
        elif path == "/api/cycles":
            self._json_response(_list_daily_cycles(workspace))
        elif path == "/api/ideas":
            self._json_response(_load_idea_backlog(workspace))
        elif path == "/api/status":
            self._json_response(_service_status(workspace))
        else:
            self._respond(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path == "/setup":
            self._handle_setup_post()
        else:
            self._respond(404, "Not found")

    def _serve_setup(self, workspace: Path):
        content = f"""
<h1>Welcome to Digital Unconscious</h1>
<p class="subtitle">Let's set up your personal AI research companion. This takes 30 seconds.</p>

<form method="POST" action="/setup">
<div class="card">
  <h2>1. What fields do you work in?</h2>
  <p style="color:var(--muted);font-size:13px;margin-bottom:12px">
    Ideas will be filtered to land in these fields. Cross-domain inspiration is still welcome.
  </p>
  <div class="form-group">
    <label>Focus fields (comma-separated)</label>
    <input type="text" name="focus_fields" placeholder="e.g. economics research, management, behavioral finance" value="{', '.join(self.config.idea.focus_fields)}">
  </div>
  <div class="form-group">
    <label>Primary domains (your core expertise)</label>
    <input type="text" name="primary_domains" placeholder="e.g. AI tools, product design" value="{', '.join(self.config.idea.primary_domains)}">
  </div>
  <div class="form-group">
    <label>Secondary domains (adjacent interests)</label>
    <input type="text" name="secondary_domains" placeholder="e.g. cognitive science, business models" value="{', '.join(self.config.idea.secondary_domains)}">
  </div>
</div>

<div class="card">
  <h2>2. Observation source</h2>
  <p style="color:var(--muted);font-size:13px;margin-bottom:12px">
    How should the system observe your screen behaviour?
  </p>
  <div class="form-group">
    <label>Source</label>
    <select name="observation_mode">
      <option value="screenpipe">Screenpipe (recommended — install from screenpipe.com)</option>
      <option value="logfile">Manual log file (JSONL or text)</option>
    </select>
  </div>
</div>

<div class="card">
  <h2>3. Briefing schedule</h2>
  <div class="form-group">
    <label>Daily briefing time</label>
    <input type="text" name="briefing_time" placeholder="22:00" value="{self.config.daily.briefing_time}">
  </div>
</div>

<div style="text-align:center;margin-top:24px">
  <button type="submit">Start Digital Unconscious</button>
  <p style="color:var(--muted);font-size:12px;margin-top:12px">
    This will enable autostart so the system runs silently in the background.
    You'll receive daily briefings automatically.
  </p>
</div>
</form>
"""
        self._html_response(_page("Setup", content, active=""))

    def _handle_setup_post(self):
        import subprocess as _subprocess
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        params = parse_qs(body)

        focus = params.get("focus_fields", [""])[0]
        primary = params.get("primary_domains", [""])[0]
        secondary = params.get("secondary_domains", [""])[0]
        briefing_time = params.get("briefing_time", ["22:00"])[0]

        # Update config
        if focus:
            self.config.idea.focus_fields = [f.strip() for f in focus.split(",") if f.strip()]
        if primary:
            self.config.idea.primary_domains = [d.strip() for d in primary.split(",") if d.strip()]
        if secondary:
            self.config.idea.secondary_domains = [d.strip() for d in secondary.split(",") if d.strip()]
        self.config.daily.briefing_time = briefing_time.strip() or "22:00"

        # Save settings
        workspace = _workspace(self.config)
        setup_dir = workspace / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "focus_fields": self.config.idea.focus_fields,
            "primary_domains": self.config.idea.primary_domains,
            "secondary_domains": self.config.idea.secondary_domains,
            "briefing_time": self.config.daily.briefing_time,
            "setup_completed_at": datetime.now(timezone.utc).isoformat(),
        }
        (setup_dir / "user_settings.json").write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Initialize workspace dirs
        for subdir in ["runs", "learning", "daily", "ideas", "prompts", "queue", "knowledge"]:
            (workspace / subdir).mkdir(parents=True, exist_ok=True)

        # Try to enable autostart
        try:
            from du_research.onboarding import enable_autostart
            project_root = Path(__file__).resolve().parents[1]
            enable_autostart(
                project_root=project_root,
                config_path=self.config.config_path or (project_root / "config" / "pipeline.toml"),
                workspace_dir=workspace,
            )
        except Exception:
            pass

        # Start background service
        try:
            python = sys.executable
            _subprocess.Popen(
                [python, "-m", "du_research.cli", "service", "start"],
                cwd=str(Path(__file__).resolve().parents[2]),
                creationflags=_subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception:
            pass

        # Redirect to dashboard
        self.send_response(302)
        self.send_header("Location", "/?setup=done")
        self.end_headers()

    def _serve_dashboard(self, workspace: Path):
        cycles = _list_daily_cycles(workspace)
        ideas = _load_idea_backlog(workspace)
        service = _service_status(workspace)
        learning = _load_learning_status(workspace)

        latest_date = cycles[0]["date"] if cycles else "—"
        total_ideas = len(ideas)
        total_cycles = len(cycles)
        model_version = learning["model"].get("model_version", 0)
        is_running = service["status"].get("running", False)

        # Check for setup=done query param
        setup_banner = ""
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)
        if "setup" in query_params:
            setup_banner = '<div class="success">Setup complete. The background service is starting. You\'ll receive your first briefing at your configured time. From now on, everything runs automatically.</div>'

        stats = f"""
<h1>Digital Unconscious</h1>
<p class="subtitle">Your AI research companion — passive observation, creative ideas, autonomous research</p>
{setup_banner}
<div class="stat-grid">
  <div class="stat"><div class="value">{total_cycles}</div><div class="label">Daily Cycles</div></div>
  <div class="stat"><div class="value">{total_ideas}</div><div class="label">Ideas Generated</div></div>
  <div class="stat"><div class="value">v{model_version}</div><div class="label">Idea Model</div></div>
  <div class="stat"><div class="value">{"ON" if is_running else "OFF"}</div><div class="label">Service</div></div>
</div>"""

        if not cycles:
            content = stats + """
<div class="empty">
<h2>No briefings yet</h2>
<p>Run <code>du daily --log-file your_log.jsonl</code> to generate your first briefing,<br>
or <code>du start</code> to begin passive observation.</p>
</div>"""
        else:
            cycle_html = ""
            for c in cycles[:10]:
                inc = f'<span class="badge badge-include">{c["ideas_included"]} included</span>' if c["ideas_included"] else ""
                cycle_html += f"""
<a class="cycle-link" href="/briefing?date={c['date']}">
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-weight:600">{c['date']}</div>
      <div style="color:var(--muted);font-size:13px">{c['ideas_total']} ideas generated {inc}</div>
    </div>
    <div style="color:var(--muted);font-size:20px">&rarr;</div>
  </div>
</div></a>"""
            content = stats + f"<h2>Recent Briefings</h2>{cycle_html}"

        self._html_response(_page("Dashboard", content, active=""))

    def _serve_briefing(self, workspace: Path, date: str | None):
        if not date:
            cycles = _list_daily_cycles(workspace)
            date = cycles[0]["date"] if cycles else None
        if not date:
            self._html_response(_page("Briefing", '<div class="empty"><h2>No briefings yet</h2></div>', active="briefing"))
            return

        md = _load_briefing(workspace, date)
        if not md:
            self._html_response(_page("Briefing", f'<div class="empty"><h2>No briefing for {date}</h2></div>', active="briefing"))
            return

        # Date navigation
        cycles = _list_daily_cycles(workspace)
        dates = [c["date"] for c in cycles]
        idx = dates.index(date) if date in dates else 0
        prev_link = f'<a href="/briefing?date={dates[idx+1]}">&larr; {dates[idx+1]}</a>' if idx + 1 < len(dates) else ""
        next_link = f'<a href="/briefing?date={dates[idx-1]}">{dates[idx-1]} &rarr;</a>' if idx > 0 else ""
        nav = f'<div style="display:flex;justify-content:space-between;margin-bottom:16px">{prev_link}<span></span>{next_link}</div>'

        briefing_html = _md_to_html(md)
        content = f'{nav}<div class="briefing">{briefing_html}</div>'
        self._html_response(_page(f"Briefing — {date}", content, active="briefing"))

    def _serve_ideas(self, workspace: Path):
        ideas = _load_idea_backlog(workspace)
        if not ideas:
            self._html_response(_page("Idea Backlog", '<div class="empty"><h2>No ideas yet</h2><p>Run a daily cycle to generate ideas.</p></div>', active="ideas"))
            return

        ideas.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        rows = ""
        for idea in ideas[:50]:
            title = idea.get("title", idea.get("idea_text", "Untitled"))
            score = idea.get("total_score", 0)
            verdict = idea.get("verdict", "hold")
            badge_cls = f"badge-{verdict}" if verdict in ("include", "hold", "discard") else "badge-hold"
            domains = idea.get("domains", [])
            domain_tags = "".join(f'<span class="tag">{d}</span>' for d in domains[:3])
            date = idea.get("date", "")
            rows += f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:start">
    <div>
      <div class="idea-title">{title}</div>
      <div style="margin-top:4px">{domain_tags}</div>
      <div style="color:var(--muted);font-size:12px;margin-top:4px">{date}</div>
    </div>
    <div style="text-align:right">
      <div class="idea-score" style="color:var(--accent)">{score:.0f}</div>
      <span class="badge {badge_cls}">{verdict}</span>
    </div>
  </div>
  <p style="color:var(--muted);font-size:13px;margin-top:8px">{idea.get("description", "")[:200]}</p>
</div>"""

        content = f"""
<h1>Idea Backlog</h1>
<p class="subtitle">{len(ideas)} ideas across all daily cycles</p>
{rows}"""
        self._html_response(_page("Idea Backlog", content, active="ideas"))

    def _serve_learning(self, workspace: Path):
        data = _load_learning_status(workspace)
        model = data["model"]
        outcomes = data["outcomes"]

        if not model:
            self._html_response(_page("Learning", '<div class="empty"><h2>No learning data yet</h2><p>Run <code>du learn</code> after a few research runs.</p></div>', active="learning"))
            return

        obsessions = ""
        for obs in model.get("core_obsessions", []):
            obsessions += f'<div class="card"><strong>{obs.get("theme", "")}</strong> — strength {obs.get("strength", 0)}, trend: {obs.get("trend", "stable")}</div>'

        blind_spots = ""
        for spot in model.get("recurring_blind_spots", []):
            blind_spots += f"<li>{spot}</li>"

        patterns = ""
        for p in outcomes.get("patterns", []):
            patterns += f'<div class="card"><strong>{p.get("type", "")}</strong><br><span style="color:var(--muted)">{p.get("insight", "")}</span><br>Action: {p.get("action", "")}</div>'

        changes_html = _md_to_html(data["changes"]) if data["changes"] else "<p>No changes recorded yet.</p>"

        content = f"""
<h1>Learning Engine</h1>
<p class="subtitle">Model version {model.get("model_version", 0)} &mdash; last updated {model.get("last_updated", "never")}</p>

<h2>Core Obsessions</h2>
{obsessions or "<p>None detected yet.</p>"}

<h2>Known Blind Spots</h2>
<ul>{blind_spots or "<li>None detected yet.</li>"}</ul>

<h2>Detected Patterns</h2>
{patterns or "<p>No patterns found yet.</p>"}

<h2>Learning Changelog</h2>
<div class="briefing">{changes_html}</div>
"""
        self._html_response(_page("Learning", content, active="learning"))

    def _serve_status(self, workspace: Path):
        service = _service_status(workspace)
        status = service["status"]
        state = service["state"]

        is_running = status.get("running", False)
        interval = status.get("interval_minutes", "—")
        completed = status.get("completed_cycles", 0)
        last_gc = state.get("last_gc_at", "—")

        recent = status.get("recent_runs", [])
        runs_html = ""
        for run in recent[-10:]:
            ts = run.get("timestamp", "")[:19]
            new_frames = run.get("new_frames", 0)
            briefing = "Briefing generated" if run.get("briefing_generated") else ""
            error = run.get("error", "")
            color = "var(--accent2)" if error else "var(--green)" if briefing else "var(--muted)"
            runs_html += f'<div style="padding:8px 0;border-bottom:1px solid var(--border);font-size:13px"><span style="color:var(--muted)">{ts}</span> &mdash; {new_frames} new frames <span style="color:{color}">{briefing}{error}</span></div>'

        content = f"""
<h1>System Status</h1>
<div class="stat-grid">
  <div class="stat"><div class="value" style="color:{"var(--green)" if is_running else "var(--accent2)"}">{"Running" if is_running else "Stopped"}</div><div class="label">Service</div></div>
  <div class="stat"><div class="value">{interval}</div><div class="label">Interval (min)</div></div>
  <div class="stat"><div class="value">{completed}</div><div class="label">Cycles Done</div></div>
</div>

<h2>Recent Activity</h2>
<div class="card">{runs_html or "<p>No recent activity.</p>"}</div>

<h2>Quick Actions</h2>
<div class="card">
  <p>Start service: <code>du start</code> or <code>du service start</code></p>
  <p>Run daily cycle: <code>du daily</code></p>
  <p>Run learning: <code>du learn</code></p>
  <p>Configure domains: <code>du config --primary "AI,design" --secondary "psychology"</code></p>
</div>
"""
        self._html_response(_page("Status", content, active="status"))

    def _html_response(self, html: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _json_response(self, data: Any, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _respond(self, status: int, message: str):
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))


def run_dashboard(config: AppConfig, port: int = 9830, open_browser: bool = True):
    DashboardHandler.config = config
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://localhost:{port}"
    print(f"Dashboard running at {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()
