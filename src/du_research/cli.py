from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from du_research.ai_backend import create_backend
from du_research.backlog import IdeaBacklog
from du_research.circuit_breaker import CircuitBreaker
from du_research.config import load_config
from du_research.onboarding import (
    apply_user_settings,
    autostart_enabled,
    disable_autostart,
    enable_autostart,
    ensure_first_run_setup,
)
from du_research.pipeline import ResearchPipeline
from du_research.service_manager import ServiceManager


def _pick_top_backlog_idea(config) -> str | None:
    """Pick the highest-scoring unresearched idea from the backlog."""
    backlog = IdeaBacklog(config.pipeline.workspace_dir, config.retention.idea_backlog_max)
    idea = backlog.top(exclude_researched=True)
    return backlog.title_of(idea) if idea else None


def _build_backend(config):
    backend_kwargs = {}
    if config.ai.api_key:
        backend_kwargs["api_key"] = config.ai.api_key
    if config.ai.openai_api_key:
        backend_kwargs["openai_api_key"] = config.ai.openai_api_key
    if config.ai.kimi_api_key:
        backend_kwargs["kimi_api_key"] = config.ai.kimi_api_key
    if config.ai.deepseek_api_key:
        backend_kwargs["deepseek_api_key"] = config.ai.deepseek_api_key
    if config.ai.glm_api_key:
        backend_kwargs["glm_api_key"] = config.ai.glm_api_key
    backend_kwargs["default_model"] = config.ai.default_model
    backend_kwargs["openai_default_model"] = config.ai.openai_default_model
    backend_kwargs["kimi_default_model"] = config.ai.kimi_default_model
    backend_kwargs["deepseek_default_model"] = config.ai.deepseek_default_model
    backend_kwargs["glm_default_model"] = config.ai.glm_default_model
    backend_kwargs["enable_fallback"] = config.ai.fallback
    backend_kwargs["fallback_order"] = config.ai.fallback_order
    raw_backend = create_backend(config.ai.mode, **backend_kwargs)
    return CircuitBreaker(
        backend=raw_backend,
        max_retries=config.circuit_breaker.max_retries,
        initial_wait=config.circuit_breaker.initial_wait,
        failure_threshold=config.circuit_breaker.failure_threshold,
        recovery_timeout=config.circuit_breaker.recovery_timeout,
    )


def _cmd_usage(config, as_json: bool = False) -> int:
    """Report token/cost usage aggregated across daily cycles."""
    workspace = Path(config.pipeline.workspace_dir).resolve()
    daily_dir = workspace / "daily"
    rows: list[dict] = []
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
    by_model: dict[str, dict] = {}
    if daily_dir.exists():
        for cycle_dir in sorted(daily_dir.glob("cycle_*")):
            usage_path = cycle_dir / "usage.json"
            if not usage_path.exists():
                continue
            try:
                usage = json.loads(usage_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            rows.append({
                "date": cycle_dir.name.replace("cycle_", ""),
                "calls": usage.get("calls", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost_usd": usage.get("cost_usd", 0.0),
            })
            for key in ("calls", "input_tokens", "output_tokens", "total_tokens"):
                totals[key] += usage.get(key, 0)
            totals["cost_usd"] += usage.get("cost_usd", 0.0)
            for model, bucket in (usage.get("by_model") or {}).items():
                agg = by_model.setdefault(model, {"calls": 0, "total_tokens": 0, "cost_usd": 0.0})
                agg["calls"] += bucket.get("calls", 0)
                agg["total_tokens"] += bucket.get("input_tokens", 0) + bucket.get("output_tokens", 0)
                agg["cost_usd"] += bucket.get("cost_usd", 0.0)

    if as_json:
        print(json.dumps({"by_day": rows, "totals": totals, "by_model": by_model}, indent=2, ensure_ascii=False))
        return 0

    if not rows:
        print("\n  No usage recorded yet. Run `du daily` to generate your first cycle.\n")
        return 0

    print("\n  Model usage — Digital Unconscious\n")
    print(f"  {'Date':<12}{'Calls':>8}{'Tokens':>14}{'Cost':>12}")
    print(f"  {'-' * 44}")
    for row in rows[-30:]:
        print(f"  {row['date']:<12}{row['calls']:>8}{row['total_tokens']:>14,}{'$' + format(row['cost_usd'], '.4f'):>12}")
    print(f"  {'-' * 44}")
    print(f"  {'Total':<12}{totals['calls']:>8}{totals['total_tokens']:>14,}{'$' + format(totals['cost_usd'], '.4f'):>12}")
    if by_model:
        print("\n  By model:")
        for model, agg in sorted(by_model.items(), key=lambda kv: kv[1]['total_tokens'], reverse=True):
            print(f"    {model:<24}{agg['calls']:>6} calls{agg['total_tokens']:>14,} tok   ${agg['cost_usd']:.4f}")
    print()
    return 0


def _cmd_models(config, as_json: bool = False) -> int:
    """Show how each agent routes to a provider/model under the current config."""
    from du_research.ai_backend import resolve_routing

    info = resolve_routing(config)
    if as_json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    print("\n  Model routing — Digital Unconscious\n")
    print(f"  Mode: {info['mode']}   Fallback: {'on' if info['fallback'] else 'off'}")
    print(f"  Available providers: {', '.join(info['available_providers']) or 'none (local Claude Code only)'}")
    if info["fallback"]:
        print(f"  Fallback order: {' -> '.join(info['fallback_order'])}")
    print(f"\n  {'Agent':<16}{'Configured':<16}{'Resolves to':<30}")
    print(f"  {'-' * 60}")
    for entry in info["routing"]:
        flag = "" if entry["available"] else f"  (falls back to {entry['fallback_provider'] or 'none'})"
        print(f"  {entry['agent']:<16}{entry['configured']:<16}{entry['provider'] + ':' + entry['model']:<30}{flag}")
    print()
    return 0


def _cmd_doctor(as_json: bool = False) -> int:
    """Explain provider readiness without exposing credentials."""
    from du_research.backends.router import diagnose_providers

    rows = diagnose_providers()
    if as_json:
        print(json.dumps({"providers": rows}, indent=2, ensure_ascii=False))
        return 0
    print("\n  Runtime doctor — Digital Unconscious\n")
    for row in rows:
        marker = "ready" if row["ready"] else "optional"
        print(f"  {row['provider']:<14} {marker:<10} {row['kind']:<18} {row['detail']}")
    local_ready = any(row["ready"] for row in rows if row["kind"] == "subscription CLI")
    print("\n  " + ("No API key is required: a local subscription CLI is ready." if local_ready else "Install and sign in to Codex or Claude Code for zero-key execution."))
    print()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="du",
        description="Digital Unconscious — Passive Screen Observation × AI Idea Generation",
    )
    parser.add_argument("--config", default="config/pipeline.toml", help="Path to TOML config file")
    subparsers = parser.add_subparsers(dest="command")

    # --- Original research pipeline commands --------------------------------
    research = subparsers.add_parser("research", help="Run the autonomous research pipeline on an idea")
    research.add_argument("--idea", help="Idea text to research")
    research.add_argument("--idea-id", help="Optional idea UUID from backlog")
    research.add_argument("--data-file", help="Optional local CSV file for descriptive analysis")
    research.add_argument("--run-id", help="Optional run id override")
    research.add_argument("--dry-run", action="store_true", help="Skip network calls; produce planning artifacts only")
    research.add_argument("--auto", action="store_true", help="Auto-select the top idea from the backlog instead of requiring --idea")
    research.add_argument("--resume", action="store_true", help="Resume an existing run (requires --run-id)")

    ideate = subparsers.add_parser(
        "ideate",
        aliases=["explore"],
        help="Turn local papers and dataset structure into evidence-backed study cards",
    )
    ideate.add_argument("--paper", action="append", default=[], help="Paper PDF/text/Markdown/JSON path; repeatable")
    ideate.add_argument("--data", action="append", default=[], help="CSV/TSV/JSON dataset path; repeatable")
    ideate.add_argument("--context", default="", help="Your domain, question, or constraints")
    ideate.add_argument("--context-file", help="Read longer context from a UTF-8 text file")
    ideate.add_argument("--max-ideas", type=int, help="Maximum study cards (default from config)")
    ideate.add_argument("--session-id", help="Stable session id override")
    ideate.add_argument("--no-backlog", action="store_true", help="Do not append viable cards to the idea backlog")
    ideate.add_argument("--dry-run", action="store_true", help="Ingest/profile locally without calling a model")

    learn_cmd = subparsers.add_parser("learn", help="Aggregate learning signals across runs")
    learn_cmd.add_argument("--workspace-dir", help="Override workspace directory")

    status_cmd = subparsers.add_parser("status", help="Show run status")
    status_cmd.add_argument("--run-id", required=True, help="Run id to inspect")

    export_task = subparsers.add_parser("export-computer-task", help="Export a supervised Claude Code computer-use task pack")
    export_task.add_argument("--run-id", required=True, help="Run id to export")
    run_task = subparsers.add_parser("run-computer-task", help="Execute a browser automation task pack")
    run_task.add_argument("--task-path", required=True, help="Path to computer_use_task.json or selenium flow pack")

    credential = subparsers.add_parser("credential", help="Manage encrypted credentials")
    credential_sub = credential.add_subparsers(dest="credential_command", required=True)
    credential_add = credential_sub.add_parser("add", help="Add or update a credential")
    credential_add.add_argument("--resource", required=True)
    credential_add.add_argument("--username", required=True)
    credential_add.add_argument("--password", required=True)
    credential_add.add_argument("--login-url")
    credential_add.add_argument("--notes", default="")
    credential_add.add_argument("--extra-json", default="", help="Optional JSON object with selectors such as username_selector/password_selector")
    credential_sub.add_parser("list", help="List stored credential resources")

    submit = subparsers.add_parser("submit", help="Update submission approval status")
    submit.add_argument("--run-id", required=True)
    submit.add_argument("--status", required=True, choices=["pending_approval", "approved", "rejected", "submitted"])
    submit.add_argument("--notes", default="")

    # --- New Digital Unconscious commands -----------------------------------
    daily = subparsers.add_parser(
        "daily",
        help="Run the full daily cycle: observe → compress → generate ideas → judge → briefing",
    )
    daily.add_argument("--log-file", help="Path to a manual daily-log file (fallback for screenpipe)")
    daily.add_argument("--date", help="Override date string (YYYY-MM-DD)")

    capture = subparsers.add_parser(
        "daily-capture",
        help="Extract candidate research ideas from a daily log (legacy heuristic mode)",
    )
    capture.add_argument("--input", required=True, help="Path to a text or JSONL daily activity file")

    init_cmd = subparsers.add_parser("init", help="Initialise the workspace and configuration")
    init_cmd.add_argument(
        "--mode",
        choices=["auto", "codex", "claude_code", "deepseek", "glm", "openai", "anthropic"],
        default="auto",
        help="AI backend mode",
    )
    init_cmd.add_argument("--force", action="store_true", help="Re-run first-launch setup even if already initialized")

    setup_cmd = subparsers.add_parser("setup", help="Run the one-time setup wizard")
    setup_cmd.add_argument("--force", action="store_true", help="Overwrite previous setup state")

    autostart_cmd = subparsers.add_parser("autostart", help="Manage launch-on-login")
    autostart_sub = autostart_cmd.add_subparsers(dest="autostart_command", required=True)
    autostart_sub.add_parser("enable", help="Enable autostart")
    autostart_sub.add_parser("disable", help="Disable autostart")
    autostart_sub.add_parser("status", help="Show autostart status")

    config_cmd = subparsers.add_parser("config", help="Show or update domain configuration")
    config_cmd.add_argument("--primary", help="Comma-separated primary domains")
    config_cmd.add_argument("--secondary", help="Comma-separated secondary domains")
    config_cmd.add_argument("--focus", help="Comma-separated focus fields (e.g. 'economics research,management')")
    config_cmd.add_argument("--show", action="store_true", help="Show current configuration")

    start_cmd = subparsers.add_parser("start", help="Start the passive observation service loop in the foreground")
    start_cmd.add_argument("--log-file", help="Optional fallback log file when screenpipe is unavailable")
    start_cmd.add_argument("--interval-minutes", type=int, help="Override service interval in minutes")
    start_cmd.add_argument("--iterations", type=int, help="How many service polls to run before exiting")

    service_cmd = subparsers.add_parser("service", help="Manage the background service daemon")
    service_sub = service_cmd.add_subparsers(dest="service_command", required=True)
    service_start = service_sub.add_parser("start", help="Start the background daemon")
    service_start.add_argument("--log-file", help="Optional fallback log file when screenpipe is unavailable")
    service_start.add_argument("--interval-minutes", type=int, help="Override service interval in minutes")
    service_sub.add_parser("stop", help="Stop the background daemon")
    service_restart = service_sub.add_parser("restart", help="Restart the background daemon")
    service_restart.add_argument("--log-file", help="Optional fallback log file when screenpipe is unavailable")
    service_restart.add_argument("--interval-minutes", type=int, help="Override service interval in minutes")
    service_sub.add_parser("status", help="Show background daemon status")

    subparsers.add_parser("tray", help="Launch system tray icon (bottom-right corner)")

    dash_cmd = subparsers.add_parser("dashboard", help="Open the web dashboard in your browser")
    dash_cmd.add_argument("--port", type=int, default=9830, help="Port for the dashboard server")
    dash_cmd.add_argument("--no-open", action="store_true", help="Don't auto-open browser")

    subparsers.add_parser("drain", help="Process pending tasks in the queue (retry failed LLM calls)")
    subparsers.add_parser("update", help="Update Digital Unconscious to the latest version from GitHub")

    logs_cmd = subparsers.add_parser("logs", help="Show logs for a run")
    logs_cmd.add_argument("--run-id", required=True, help="Run id to inspect")
    logs_cmd.add_argument("--follow", action="store_true", help="Follow logs in real time")

    usage_cmd = subparsers.add_parser("usage", help="Show token and cost usage across daily cycles")
    usage_cmd.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    models_cmd = subparsers.add_parser("models", help="Show how each agent routes to a provider/model")
    models_cmd.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    doctor_cmd = subparsers.add_parser("doctor", help="Check local CLIs and optional model providers")
    doctor_cmd.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    return parser


def _launch_background_service(config_path: str | None = None) -> None:
    """Start the observation service as a detached background process."""
    import subprocess as _sp
    cmd = [sys.executable, "-m", "du_research.cli", "service", "start"]
    if config_path:
        cmd.extend(["--config", config_path])
    try:
        _sp.Popen(
            cmd,
            creationflags=_sp.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
        )
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    argv = list(argv) if argv is not None else sys.argv[1:]
    if not argv:
        # No arguments = launch the full desktop experience via pythonw.exe
        # This detaches from the terminal so the user gets their prompt back
        if sys.platform == "win32":
            from du_research.launcher import launch_windowless
            if launch_windowless():
                print("  Digital Unconscious started. Look for the icon in your taskbar.")
                print("  (You may need to click the ^ arrow to find it.)")
                return 0
        # Fallback: run tray inline (Mac/Linux or if launch failed)
        argv = ["tray"]
    args = parser.parse_args(argv)
    config = load_config(args.config)
    project_root = Path(__file__).resolve().parents[2]
    apply_user_settings(config)

    if getattr(args, "workspace_dir", None):
        config.pipeline.workspace_dir = args.workspace_dir
        apply_user_settings(config)

    if args.command not in {"setup", "init", "autostart", "config", "doctor", "models", "usage"}:
        ensure_first_run_setup(config, project_root=project_root, interactive=None)

    service_manager = ServiceManager(config=config, project_root=project_root)

    # --- research (original pipeline) ---------------------------------------
    if args.command == "research":
        pipeline = ResearchPipeline(config, backend=_build_backend(config))

        # --resume: resume an existing run
        if args.resume:
            if not args.run_id:
                parser.error("--resume requires --run-id")
            result = pipeline.run(
                idea_text=args.idea or "",
                data_file=args.data_file,
                run_id=args.run_id,
                dry_run=args.dry_run,
                resume=True,
            )
        # --auto: pick the top idea from the backlog automatically
        elif args.auto:
            backlog = IdeaBacklog(config.pipeline.workspace_dir, config.retention.idea_backlog_max)
            selected = backlog.top(exclude_researched=True)
            if not selected:
                print('{"error": "No ideas in backlog. Run `du daily` first."}', file=sys.stderr)
                return 1
            result = pipeline.run(
                idea_text=backlog.title_of(selected),
                idea_id=backlog.id_of(selected) or None,
                research_seed=selected,
                data_file=args.data_file,
                run_id=args.run_id,
                dry_run=args.dry_run,
            )
        else:
            if not args.idea and not args.idea_id:
                parser.error("research requires --idea, --idea-id, or --auto")
            result = pipeline.run(
                idea_text=args.idea,
                idea_id=args.idea_id,
                data_file=args.data_file,
                run_id=args.run_id,
                dry_run=args.dry_run,
            )

        summary = {
            "run_id": result["run_id"],
            "run_dir": result["run_dir"],
            "final_quality_score": result["review"]["overall_score"],
            "analysis_executed": result["analysis"].get("analysis_executed", False),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command in {"ideate", "explore"}:
        from du_research.ideation import ResearchIdeationLab

        context = args.context
        if args.context_file:
            context_path = Path(args.context_file).expanduser().resolve()
            context = "\n\n".join(part for part in [context, context_path.read_text(encoding="utf-8")] if part)
        lab = ResearchIdeationLab(config, None if args.dry_run else _build_backend(config))
        result = lab.run(
            paper_paths=args.paper,
            data_paths=args.data,
            context=context,
            max_ideas=args.max_ideas,
            add_to_backlog=not args.no_backlog,
            dry_run=args.dry_run,
            session_id=args.session_id,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # --- daily (new: full AI-powered daily cycle) ---------------------------
    if args.command == "daily":
        from du_research.engine import DigitalUnconsciousEngine
        engine = DigitalUnconsciousEngine(config)
        result = engine.run_daily_cycle(
            log_file=args.log_file,
            date_str=args.date,
        )
        # Human-friendly output
        print(f"\n  Daily cycle complete for {result['date']}")
        print(f"  Observed {result['frames_observed']} behaviour frames")
        print(f"  Compressed into {result['windows_compressed']} time windows")
        print(f"  Generated {result['ideas_generated']} ideas")
        print(f"  Included: {result['ideas_included']}  |  Held: {result['ideas_held']}")
        if result.get("research_runs_started"):
            print(f"  Auto-research started for {result['research_runs_started']} idea(s)")
        print(f"\n  Briefing: {result['briefing_path']}")
        print("  Dashboard: du dashboard\n")
        return 0

    # --- daily-capture (legacy heuristic extraction) ------------------------
    if args.command == "daily-capture":
        pipeline = ResearchPipeline(config)
        result = pipeline.capture_daily(args.input)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # --- learn --------------------------------------------------------------
    if args.command == "learn":
        from du_research.engine import DigitalUnconsciousEngine
        engine = DigitalUnconsciousEngine(config)
        result = engine.run_learning_cycle()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # --- status -------------------------------------------------------------
    if args.command == "status":
        pipeline = ResearchPipeline(config)
        manifest = pipeline.status(args.run_id)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    # --- export-computer-task -----------------------------------------------
    if args.command == "export-computer-task":
        pipeline = ResearchPipeline(config)
        task_path = pipeline.export_computer_task(args.run_id)
        print(json.dumps({"task_path": task_path}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "run-computer-task":
        pipeline = ResearchPipeline(config, backend=_build_backend(config))
        result = pipeline.run_computer_task(args.task_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "credential":
        pipeline = ResearchPipeline(config)
        if args.credential_command == "add":
            pipeline.set_credential(
                args.resource,
                args.username,
                args.password,
                login_url=args.login_url,
                notes=args.notes,
                extra_fields=json.loads(args.extra_json) if args.extra_json else None,
            )
            print(json.dumps({"status": "stored", "resource": args.resource}, indent=2, ensure_ascii=False))
            return 0
        if args.credential_command == "list":
            print(json.dumps({"resources": pipeline.list_credentials()}, indent=2, ensure_ascii=False))
            return 0

    if args.command == "submit":
        pipeline = ResearchPipeline(config)
        path = pipeline.update_submission_status(args.run_id, args.status, args.notes)
        print(json.dumps({"submission_path": path, "status": args.status}, indent=2, ensure_ascii=False))
        return 0

    # --- init ---------------------------------------------------------------
    if args.command == "init":
        workspace = Path(config.pipeline.workspace_dir).resolve()
        for subdir in ["runs", "learning", "daily", "ideation", "ideas", "prompts"]:
            (workspace / subdir).mkdir(parents=True, exist_ok=True)
        state = ensure_first_run_setup(config, project_root=project_root, force=args.force, interactive=None)
        print(json.dumps({
            "workspace": str(workspace),
            "mode": args.mode,
            "status": "initialized",
            "setup": state,
        }, indent=2))
        return 0

    if args.command == "setup":
        state = ensure_first_run_setup(config, project_root=project_root, force=args.force, interactive=True)
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return 0

    if args.command == "autostart":
        workspace = Path(config.pipeline.workspace_dir).resolve()
        if args.autostart_command == "enable":
            script = enable_autostart(
                project_root=project_root,
                config_path=config.config_path.resolve() if config.config_path else (project_root / "config" / "pipeline.toml"),
                workspace_dir=workspace,
            )
            print(json.dumps({"enabled": True, "script_path": str(script) if script else None}, indent=2, ensure_ascii=False))
            return 0
        if args.autostart_command == "disable":
            removed = disable_autostart()
            print(json.dumps({"enabled": False, "removed": removed}, indent=2, ensure_ascii=False))
            return 0
        if args.autostart_command == "status":
            print(json.dumps({"enabled": autostart_enabled(), "service": service_manager.status()}, indent=2, ensure_ascii=False))
            return 0

    # --- config -------------------------------------------------------------
    if args.command == "config":
        if args.show or (not args.primary and not args.secondary and not args.focus):
            info = {
                "ai_mode": config.ai.mode,
                "primary_domains": config.idea.primary_domains,
                "secondary_domains": config.idea.secondary_domains,
                "focus_fields": config.idea.focus_fields,
                "observation_enabled": config.observation.enabled,
                "learning_enabled": config.learning.prompt_evolution,
                "automation_enabled": config.automation.enabled,
                "automation_auto_execute": config.automation.auto_execute,
                "autostart_enabled": autostart_enabled(),
                "background_service": service_manager.status(),
                "workspace": config.pipeline.workspace_dir,
            }
            print(json.dumps(info, indent=2, ensure_ascii=False))
        else:
            if args.primary:
                config.idea.primary_domains = [d.strip() for d in args.primary.split(",")]
            if args.secondary:
                config.idea.secondary_domains = [d.strip() for d in args.secondary.split(",")]
            if args.focus:
                config.idea.focus_fields = [d.strip() for d in args.focus.split(",")]
            print(json.dumps({
                "primary_domains": config.idea.primary_domains,
                "secondary_domains": config.idea.secondary_domains,
                "focus_fields": config.idea.focus_fields,
                "note": "Changes apply to this session only. Edit config/pipeline.toml for persistence.",
            }, indent=2, ensure_ascii=False))
        return 0

    # --- start (foreground observation service) ------------------------------
    if args.command == "start":
        from du_research.engine import DigitalUnconsciousEngine
        engine = DigitalUnconsciousEngine(config)
        interval = args.interval_minutes or config.observation.service_interval_minutes
        print("\n  Digital Unconscious — observation service started")
        print(f"  Mode: {config.ai.mode}  |  Interval: {interval}min  |  Briefing at: {config.daily.briefing_time}")
        print("  Dashboard: du dashboard  |  Stop: Ctrl+C\n")
        result = engine.run_observation_service(
            log_file=args.log_file,
            interval_minutes=args.interval_minutes,
            iterations=args.iterations,
        )
        return 0

    if args.command == "service":
        config_path = config.config_path.resolve() if config.config_path else (project_root / "config" / "pipeline.toml")
        if args.service_command == "start":
            result = service_manager.start_background(
                config_path=config_path,
                log_file=args.log_file,
                interval_minutes=args.interval_minutes,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.service_command == "stop":
            print(json.dumps(service_manager.stop(), indent=2, ensure_ascii=False))
            return 0
        if args.service_command == "restart":
            result = service_manager.restart(
                config_path=config_path,
                log_file=args.log_file,
                interval_minutes=args.interval_minutes,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.service_command == "status":
            print(json.dumps(service_manager.status(), indent=2, ensure_ascii=False))
            return 0

    # --- tray (system tray icon — the default "du" experience) ----------------
    if args.command == "tray":
        from du_research.launcher import create_launcher_script
        from du_research.tray import run_tray

        workspace = Path(config.pipeline.workspace_dir).resolve()
        # Gate on a marker written only when the user finishes the web wizard —
        # NOT on user_settings.json, which ensure_first_run_setup writes with
        # defaults non-interactively (which would suppress the wizard forever).
        setup_done = (workspace / "setup" / "setup_complete.json").exists()

        # Create launcher scripts for future use (desktop shortcut, startup)
        try:
            create_launcher_script()
        except Exception:
            pass

        # 1. Start dashboard server in background
        threading.Thread(
            target=lambda: __import__("du_research.dashboard", fromlist=["run_dashboard"]).run_dashboard(config, open_browser=False),
            daemon=True,
        ).start()

        # 2. Start observation service in background
        _launch_background_service()

        # 3. If first run, open browser to setup wizard
        if not setup_done:
            import time as _time
            _time.sleep(0.5)  # let dashboard server start
            import webbrowser
            webbrowser.open("http://localhost:9830/setup")

        # 4. Show tray icon (main loop — user sees icon in taskbar)
        run_tray(config)
        return 0

    # --- update ---------------------------------------------------------------
    if args.command == "update":
        import subprocess as _sp
        print("  Updating Digital Unconscious from GitHub...")
        result = _sp.run(
            [sys.executable, "-m", "pip", "install", "--upgrade",
             "digital-unconscious[full] @ git+https://github.com/shoal-rat/digital-unconscious.git"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("  Updated successfully.")
        else:
            print(f"  Update failed: {result.stderr[:500]}")
        return result.returncode

    # --- drain (process queued tasks) -----------------------------------------
    if args.command == "drain":
        from du_research.engine import DigitalUnconsciousEngine
        engine = DigitalUnconsciousEngine(config)
        pending = engine.task_queue.count_pending()
        if pending == 0:
            print("  No pending tasks in queue.")
            return 0
        print(f"  Processing {pending} pending task(s)...")
        result = engine.drain_queue()
        print(f"  Done: {result['succeeded']} succeeded, {result['failed']} failed, {result['remaining']} remaining")
        return 0

    # --- dashboard ------------------------------------------------------------
    if args.command == "dashboard":
        from du_research.dashboard import run_dashboard
        run_dashboard(config, port=args.port, open_browser=not args.no_open)
        return 0

    # --- logs ---------------------------------------------------------------
    if args.command == "logs":
        trace_file = Path(config.pipeline.workspace_dir) / "runs" / args.run_id / "execution_trace.jsonl"
        if not trace_file.exists():
            print(f"No trace found for run {args.run_id}", file=sys.stderr)
            return 1
        for line in trace_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                ts = entry.get("timestamp", "")[:19]
                stage = entry.get("stage", "")
                event = entry.get("event", "")
                print(f"[{ts}] {stage}: {event}")
        return 0

    if args.command == "usage":
        return _cmd_usage(config, as_json=args.json)

    if args.command == "models":
        return _cmd_models(config, as_json=args.json)

    if args.command == "doctor":
        return _cmd_doctor(as_json=args.json)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
