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


def _fmt_compact(n: int) -> str:
    """Format a token count compactly (1234 -> '1.2k', 2_000_000 -> '2.0M')."""
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


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
            usage = _load_json(d / "usage.json") or {}
            cycles.append({
                "date": date,
                "dir": str(d),
                "has_briefing": briefing_path.exists(),
                "ideas_total": n_ideas,
                "ideas_included": n_included,
                "tokens": usage.get("total_tokens", 0),
                "cost_usd": usage.get("cost_usd", 0.0),
            })
    return cycles[:30]


def _load_briefing(workspace: Path, date: str) -> str | None:
    path = workspace / "daily" / f"cycle_{date}" / f"briefing_{date}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _briefing_focus(workspace: Path, date: str) -> str:
    """Extract the first paragraph under a '## ...Focus...' heading for the Today card."""
    md = _load_briefing(workspace, date) or ""
    out: list[str] = []
    capture = False
    for line in md.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            if capture:
                break
            capture = "focus" in stripped.lower()
            continue
        if capture and stripped and not stripped.startswith("#"):
            out.append(stripped)
    return " ".join(out)[:320]


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
:root { --bg:#0b0d13; --panel:#10131c; --card:#161a24; --card2:#1b2030; --border:#262c3a;
  --text:#e7eaf2; --muted:#9aa1b6; --faint:#6a7188; --accent:#7c8cff; --accent2:#ff6f91;
  --green:#43d98c; --grad:linear-gradient(135deg,#7c8cff 0%,#b07cff 100%); --radius:14px; }
* { margin:0; padding:0; box-sizing:border-box; }
body { color:var(--text); line-height:1.6; min-height:100vh;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:
    radial-gradient(900px 520px at 100% -10%, rgba(124,140,255,.10), transparent 60%),
    radial-gradient(760px 520px at -10% 0%, rgba(176,124,255,.08), transparent 55%),
    var(--bg); }
.container { max-width:920px; margin:0 auto; padding:0 20px 32px; }
.topbar { display:flex; align-items:center; justify-content:space-between; gap:16px;
  padding:16px 0; margin-bottom:24px; border-bottom:1px solid var(--border);
  position:sticky; top:0; background:rgba(11,13,19,.82); backdrop-filter:blur(8px); z-index:10; }
.brand { display:flex; align-items:center; gap:10px; text-decoration:none; color:var(--text); }
.brand .mark { width:30px; height:30px; flex:none; }
.brand .name { font-weight:700; font-size:16px; letter-spacing:-.01em; }
.brand .name span { background:var(--grad); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
nav { display:flex; gap:4px; flex-wrap:wrap; }
nav a { color:var(--muted); text-decoration:none; padding:7px 13px; border-radius:9px;
  font-size:13.5px; font-weight:500; transition:all .15s; }
nav a:hover { color:var(--text); background:var(--card); }
nav a.active { color:var(--text); background:var(--card2); }
.hero { margin-bottom:18px; }
.hero h1 { font-size:26px; font-weight:700; letter-spacing:-.02em; margin-bottom:4px; }
h1 { font-size:24px; font-weight:700; letter-spacing:-.01em; margin-bottom:8px; }
h2 { font-size:13px; font-weight:600; margin:28px 0 12px; color:var(--muted); text-transform:uppercase; letter-spacing:.07em; }
h3 { font-size:15px; font-weight:600; margin:0 0 8px; }
.subtitle { color:var(--muted); font-size:14px; margin-bottom:24px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:var(--radius);
  padding:20px; margin-bottom:16px; transition:border-color .15s, transform .15s; }
.card:hover { border-color:#333b4d; }
.panel { background:linear-gradient(180deg,var(--card),var(--panel)); }
.stat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:16px; }
.stat { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; }
.stat .value { font-size:30px; font-weight:800; letter-spacing:-.02em;
  background:var(--grad); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.stat .label { font-size:12px; color:var(--muted); margin-top:2px; }
.row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
.btn { display:inline-flex; align-items:center; gap:8px; border:1px solid transparent; border-radius:10px;
  padding:11px 18px; font-size:14px; font-weight:600; cursor:pointer; transition:all .15s; text-decoration:none; }
.btn-primary { background:var(--grad); color:#fff; box-shadow:0 6px 18px rgba(124,140,255,.30); }
.btn-primary:hover { filter:brightness(1.07); transform:translateY(-1px); }
.btn-secondary { background:var(--card2); color:var(--text); border-color:var(--border); }
.btn-secondary:hover { border-color:var(--accent); }
.btn-ghost { background:transparent; color:var(--muted); }
.btn-ghost:hover { color:var(--text); background:var(--card); }
.btn:disabled { opacity:.55; cursor:default; transform:none; }
.pill { display:inline-flex; align-items:center; gap:7px; font-size:12.5px; font-weight:600;
  padding:5px 12px; border-radius:999px; border:1px solid var(--border); color:var(--muted); }
.pill .dot { width:7px; height:7px; border-radius:50%; background:var(--faint); }
.pill.on { color:var(--green); border-color:rgba(67,217,140,.35); }
.pill.on .dot { background:var(--green); box-shadow:0 0 8px var(--green); }
.badge { display:inline-block; padding:2px 9px; border-radius:6px; font-size:12px; font-weight:600; }
.badge-include { background:rgba(67,217,140,.14); color:var(--green); }
.badge-hold { background:rgba(124,140,255,.14); color:var(--accent); }
.badge-discard { background:rgba(255,111,145,.14); color:var(--accent2); }
.today { background:linear-gradient(180deg,var(--card2),var(--card)); border-left:3px solid #7c8cff; }
.today .focus { font-size:15.5px; line-height:1.75; color:var(--text); }
.idea-row { display:flex; justify-content:space-between; align-items:center; padding:12px 0; border-bottom:1px solid var(--border); }
.idea-row:last-child { border-bottom:none; }
.idea-title { font-weight:500; font-size:14px; } .idea-score { font-size:14px; font-weight:700; }
.briefing { line-height:1.8; font-size:15px; }
.briefing h1 { font-size:22px; margin:8px 0 8px; }
.briefing h2 { font-size:16px; margin:22px 0 8px; color:var(--accent); text-transform:none; letter-spacing:0; }
.briefing h3 { font-size:15px; }
.briefing ul, .briefing ol { margin-left:24px; margin-bottom:12px; } .briefing li { margin-bottom:4px; }
.briefing hr { border:none; border-top:1px solid var(--border); margin:20px 0; }
.briefing strong { color:var(--accent); } .briefing em { color:var(--muted); }
.briefing code { background:var(--card2); padding:2px 6px; border-radius:5px; font-size:13px; }
.cycle-link { display:block; text-decoration:none; color:inherit; }
.empty { text-align:center; padding:44px 24px; }
.empty .eyebrow { color:var(--accent); font-size:13px; font-weight:600; letter-spacing:.05em; text-transform:uppercase; }
.empty h2 { color:var(--text); text-transform:none; letter-spacing:0; font-size:22px; margin:8px 0 6px; }
.empty p { color:var(--muted); font-size:14px; margin-bottom:18px; }
.steps { display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin-top:22px; }
.step { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:14px 16px; width:190px; text-align:left; }
.step .n { color:var(--accent); font-weight:700; font-size:13px; } .step .t { font-size:13px; color:var(--muted); margin-top:4px; }
a { color:var(--accent); }
.tag { display:inline-block; background:rgba(124,140,255,.10); color:var(--accent); padding:2px 8px; border-radius:6px; font-size:12px; margin:2px; }
pre { background:var(--card); padding:16px; border-radius:10px; overflow-x:auto; font-size:13px; line-height:1.5; border:1px solid var(--border); }
input[type=text], input[type=password], textarea, select { background:var(--card); color:var(--text); border:1px solid var(--border);
  border-radius:10px; padding:11px 14px; font-size:14px; width:100%; margin-bottom:12px; }
input:focus, textarea:focus, select:focus { border-color:var(--accent); outline:none; }
button { background:var(--grad); color:#fff; border:none; border-radius:10px; padding:12px 22px;
  font-size:14px; font-weight:600; cursor:pointer; transition:all .15s; }
button:hover { filter:brightness(1.07); }
label { display:block; color:var(--muted); font-size:13px; margin-bottom:4px; }
.form-group { margin-bottom:20px; }
.success { background:rgba(67,217,140,.14); color:var(--green); padding:12px 16px; border-radius:10px; margin-bottom:16px; border:1px solid rgba(67,217,140,.25); }
.footer { color:var(--faint); font-size:12px; text-align:center; padding-top:26px; border-top:1px solid var(--border); margin-top:36px; }
.footer a { color:var(--muted); }
@media (max-width:640px){ .topbar{flex-direction:column; align-items:flex-start; gap:10px;} .stat .value{font-size:26px;} }
"""

_DASH_JS = """
<script>
function duMsg(t){var e=document.getElementById('du-msg'); if(e) e.textContent=t;}
async function duRun(){var b=document.getElementById('run-btn'); if(b){b.disabled=true; b.textContent='Running...';}
  duMsg('Running a cycle - this can take up to a minute. The page will refresh automatically.');
  try{await fetch('/api/run',{method:'POST'});}catch(e){}
  setTimeout(function(){location.reload();}, 12000);}
async function duService(a){duMsg('Service '+a+'...');
  try{await fetch('/api/service?action='+a,{method:'POST'});}catch(e){}
  setTimeout(function(){location.reload();}, 1500);}
</script>"""


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
        ("setup", "Settings"),
    ]
    nav_html = ""
    for href, label in nav_items:
        cls = ' class="active"' if href == active else ""
        nav_html += f'<a href="/{href}"{cls}>{label}</a>'

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Digital Unconscious</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><rect x='4' y='4' width='56' height='56' rx='18' fill='%230d1220'/><path d='M13 27c8-8 14 8 22 0s12-6 16 0' fill='none' stroke='%2372e6c1' stroke-width='4' stroke-linecap='round'/><path d='m17 40 14 9 16-13M31 49l5-17' fill='none' stroke='%238b9cff' stroke-width='2'/></svg>">
<style>{_CSS}</style>
</head><body>
<div class="container">
<div class="topbar">
<a class="brand" href="/"><svg class="mark" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="duG" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#72e6c1"/><stop offset=".55" stop-color="#8b9cff"/><stop offset="1" stop-color="#d28cff"/></linearGradient></defs><rect x="1" y="1" width="30" height="30" rx="10" fill="#11131c" stroke="#293149"/><path d="M6 13c5-5 8 5 13 0s6-3 8 0" stroke="url(#duG)" stroke-width="2.2" stroke-linecap="round"/><path d="m8 20 7 5 9-7m-9 7 3-10" stroke="url(#duG)" stroke-width="1.2" opacity=".8"/><circle cx="8" cy="20" r="1.5" fill="#72e6c1"/><circle cx="15" cy="25" r="1.5" fill="#8b9cff"/><circle cx="24" cy="18" r="1.5" fill="#d28cff"/></svg><span class="name">Digital <span>Unconscious</span></span></a>
<nav>{nav_html}</nav>
</div>
{content}
<div class="footer">Digital Unconscious &middot; local-first AI research companion &middot; <a href="/status">status</a> &middot; <a href="/setup">settings</a></div>
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

        # Redirect to setup until the user finishes the web wizard (marker is
        # written only by _handle_setup_post, not by the non-interactive defaults).
        setup_done = (workspace / "setup" / "setup_complete.json").exists()
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
        elif path == "/api/run":
            self._handle_run_now()
        elif path == "/api/service":
            self._handle_service(parsed)
        else:
            self._respond(404, "Not found")

    def _spawn_cli(self, *cli_args: str) -> None:
        import subprocess as _sp
        import sys as _sys
        cmd = [_sys.executable, "-m", "du_research.cli"]
        config_path = getattr(self.config, "config_path", None)
        if config_path:
            cmd += ["--config", str(config_path)]  # use the same config the dashboard rendered
        cmd += list(cli_args)
        _sp.Popen(
            cmd,
            creationflags=_sp.CREATE_NO_WINDOW if _sys.platform == "win32" else 0,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
        )

    def _handle_run_now(self):
        try:
            self._spawn_cli("daily")
            self._json_response({"status": "started", "message": "Daily cycle started — refresh in a minute."})
        except Exception as exc:
            self._json_response({"status": "error", "error": str(exc)}, status=500)

    def _handle_service(self, parsed):
        action = parse_qs(parsed.query).get("action", ["start"])[0]
        action = action if action in {"start", "stop", "restart"} else "start"
        try:
            self._spawn_cli("service", action)
            self._json_response({"status": "ok", "action": action})
        except Exception as exc:
            self._json_response({"status": "error", "error": str(exc)}, status=500)

    def _serve_setup(self, workspace: Path):
        content = f"""
<h1>Welcome to Digital Unconscious</h1>
<p class="subtitle">Let's set up your personal AI research companion. This takes 30 seconds.</p>

<form method="POST" action="/setup">
<div class="card">
  <h2>1. What fields do you work in?</h2>
  <p style="color:var(--muted);font-size:13px;margin-bottom:12px">
    Topics you want ideas to stay close to. Leave blank to get ideas from everything.
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
    How should the system observe your screen? <b>Automatic</b> uses whatever is available.
    <b>Vision</b> uses your signed-in Codex or Claude Code subscription.
  </p>
  <div class="form-group">
    <label>Source</label>
    <select name="observation_mode">
      <option value="auto">Automatic — use the best available (recommended)</option>
      <option value="vision">Vision — local subscription model</option>
      <option value="screenpipe">Screenpipe (if installed)</option>
      <option value="logfile">Manual log file (JSONL or text)</option>
    </select>
  </div>
  <p style="color:var(--muted);font-size:13px">
    No API key is needed. Run <code>du doctor</code> to check Codex and Claude Code.
    Optional hosted providers are configured with environment variables, never stored by this form.
  </p>
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
    On supported desktops, setup also enables the background schedule.
    You can always manage it later from the Status page.
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
        source = params.get("observation_mode", ["auto"])[0].strip().lower()

        # Update config
        if focus:
            self.config.idea.focus_fields = [f.strip() for f in focus.split(",") if f.strip()]
        if primary:
            self.config.idea.primary_domains = [d.strip() for d in primary.split(",") if d.strip()]
        if secondary:
            self.config.idea.secondary_domains = [d.strip() for d in secondary.split(",") if d.strip()]
        self.config.daily.briefing_time = briefing_time.strip() or "22:00"

        source = {"logfile": "file"}.get(source, source)
        if source in {"auto", "vision", "screenpipe", "file"}:
            self.config.observation.source = source

        # Save settings as a nested dict so apply_user_settings reloads them on
        # every future start (a flat dict would be silently ignored).
        workspace = _workspace(self.config)
        setup_dir = workspace / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "idea": {
                "focus_fields": self.config.idea.focus_fields,
                "primary_domains": self.config.idea.primary_domains,
                "secondary_domains": self.config.idea.secondary_domains,
            },
            "daily": {"briefing_time": self.config.daily.briefing_time},
            "observation": {"enabled": True, "source": self.config.observation.source},
        }
        (setup_dir / "user_settings.json").write_text(
            json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Mark setup complete so the wizard stops auto-opening on launch.
        (setup_dir / "setup_complete.json").write_text(
            json.dumps({"completed_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
            encoding="utf-8",
        )

        # Initialize workspace dirs
        for subdir in ["runs", "learning", "daily", "ideas", "prompts", "queue", "knowledge"]:
            (workspace / subdir).mkdir(parents=True, exist_ok=True)

        # Try to enable autostart
        try:
            from du_research.onboarding import enable_autostart
            project_root = Path(__file__).resolve().parents[2]  # repo root (src/du_research/dashboard.py)
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
        total_cost = sum(c.get("cost_usd", 0.0) for c in cycles)
        total_tokens = sum(c.get("tokens", 0) for c in cycles)
        usage_value = f"${total_cost:.2f}" if total_cost else _fmt_compact(total_tokens)
        usage_label = "Est. Spend" if total_cost else "Tokens Used"

        setup_banner = ""
        if "setup" in parse_qs(urlparse(self.path).query):
            setup_banner = '<div class="success">Setup complete — the background service is starting. Run a cycle now, or wait for your daily briefing.</div>'

        service_pill = (
            '<span class="pill on"><span class="dot"></span>Service running</span>'
            if is_running else
            '<span class="pill"><span class="dot"></span>Service stopped</span>'
        )
        controls = f"""
<div class="card panel" style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
  <div class="row">{service_pill}</div>
  <div class="row">
    <button class="btn btn-primary" id="run-btn" onclick="duRun()">Run a cycle now</button>
    <button class="btn btn-secondary" onclick="duService('start')">Start service</button>
    <button class="btn btn-ghost" onclick="duService('stop')">Stop</button>
  </div>
</div>
<div id="du-msg" class="subtitle" style="margin:-6px 0 10px"></div>
{_DASH_JS}"""

        if not cycles:
            content = f"""
<div class="hero"><h1>Welcome</h1></div>
{setup_banner}
{controls}
<div class="empty">
  <div class="eyebrow">Get started</div>
  <h2>Let's capture your first idea</h2>
  <p>Digital Unconscious watches what you read and build, then proposes research ideas each day.</p>
  <button class="btn btn-primary" onclick="duRun()">Run your first cycle</button>
  <div class="steps">
    <div class="step"><div class="n">1</div><div class="t">It reads your screen (or a log) and compresses the day.</div></div>
    <div class="step"><div class="n">2</div><div class="t">A creative model proposes cross-domain ideas.</div></div>
    <div class="step"><div class="n">3</div><div class="t">You get a short briefing with the strongest ones.</div></div>
  </div>
</div>"""
            self._html_response(_page("Dashboard", content, active=""))
            return

        stats = f"""
<div class="hero"><h1>Today</h1><p class="subtitle">Latest cycle {latest_date} &middot; {total_cycles} cycles &middot; {total_ideas} ideas captured</p></div>
{setup_banner}
<div class="stat-grid">
  <div class="stat"><div class="value">{total_cycles}</div><div class="label">Daily cycles</div></div>
  <div class="stat"><div class="value">{total_ideas}</div><div class="label">Ideas generated</div></div>
  <div class="stat"><div class="value">v{model_version}</div><div class="label">Idea model</div></div>
  <div class="stat"><div class="value">{usage_value}</div><div class="label">{usage_label}</div></div>
</div>
{controls}"""

        focus = _briefing_focus(workspace, latest_date)
        today_html = ""
        if focus:
            today_html = f"""
<div class="card today">
  <h3>Today's Focus</h3>
  <div class="focus">{focus}</div>
  <div style="margin-top:12px"><a href="/briefing?date={latest_date}">Read the full briefing &rarr;</a></div>
</div>"""

        cycle_html = ""
        for c in cycles[:10]:
            inc = f'<span class="badge badge-include">{c["ideas_included"]} included</span>' if c["ideas_included"] else ""
            if c.get("cost_usd"):
                usage_note = f' &middot; ${c["cost_usd"]:.4f}'
            elif c.get("tokens"):
                usage_note = f' &middot; {_fmt_compact(c["tokens"])} tokens'
            else:
                usage_note = ""
            cycle_html += f"""
<a class="cycle-link" href="/briefing?date={c['date']}"><div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div><div style="font-weight:600">{c['date']}</div>
      <div style="color:var(--muted);font-size:13px">{c['ideas_total']} ideas {inc}{usage_note}</div></div>
    <div style="color:var(--faint);font-size:20px">&rarr;</div>
  </div></div></a>"""
        content = stats + today_html + "<h2>Recent briefings</h2>" + cycle_html
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

        from du_research.ai_backend import resolve_routing
        cycles = _list_daily_cycles(workspace)
        total_tokens = sum(c.get("tokens", 0) for c in cycles)
        total_cost = sum(c.get("cost_usd", 0.0) for c in cycles)
        usage_line = (
            f"${total_cost:.4f} &middot; {total_tokens:,} tokens across {len(cycles)} cycle(s)"
            if cycles else "No usage yet — run a cycle from the Dashboard."
        )
        info = resolve_routing(self.config)
        routing_rows = ""
        for entry in info["routing"]:
            target = f'{entry["provider"]}:{entry["model"]}'
            if not entry["available"]:
                fallback = entry.get("fallback_provider") or "unavailable"
                fallback_model = entry.get("fallback_model")
                if fallback_model:
                    fallback += f":{fallback_model}"
                target += f' <span style="color:var(--muted)">→ {fallback}</span>'
            routing_rows += (
                f'<tr><td style="padding:4px 14px 4px 0">{entry["agent"]}</td>'
                f'<td style="padding:4px 14px 4px 0;color:var(--muted)">{entry["configured"]}</td>'
                f'<td style="padding:4px 0">{target}</td></tr>'
            )

        content = f"""
<h1>System Status</h1>
<div class="stat-grid">
  <div class="stat"><div class="value" style="color:{"var(--green)" if is_running else "var(--accent2)"}">{"Running" if is_running else "Stopped"}</div><div class="label">Service</div></div>
  <div class="stat"><div class="value">{interval}</div><div class="label">Interval (min)</div></div>
  <div class="stat"><div class="value">{completed}</div><div class="label">Cycles Done</div></div>
</div>

<h2>Usage</h2>
<div class="card">{usage_line}</div>

<h2>Model Routing</h2>
<div class="card">
  <p style="color:var(--muted);font-size:13px">Mode: {info['mode']} &middot; Providers: {', '.join(info['available_providers'])}</p>
  <table style="font-size:13px;border-collapse:collapse">{routing_rows}</table>
</div>

<h2>Recent Activity</h2>
<div class="card">{runs_html or "<p>No recent activity.</p>"}</div>

<div class="card">
  <p>Change your focus fields or observation source on the <a href="/setup">Settings</a> page.</p>
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
