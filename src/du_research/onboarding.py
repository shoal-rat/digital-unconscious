from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from du_research.config import AppConfig
from du_research.utils import iso_now


def _workspace_dir(config: AppConfig) -> Path:
    return Path(config.pipeline.workspace_dir).resolve()


def _setup_dir(workspace_dir: Path) -> Path:
    return workspace_dir / "setup"


def settings_path(workspace_dir: Path) -> Path:
    return _setup_dir(workspace_dir) / "user_settings.json"


def state_path(workspace_dir: Path) -> Path:
    return _setup_dir(workspace_dir) / "setup_state.json"


def startup_script_path() -> Path | None:
    if os.name != "nt":
        return None
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "digital_unconscious_start.cmd"


def load_user_settings(workspace_dir: Path) -> dict[str, Any]:
    path = settings_path(workspace_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_user_settings(workspace_dir: Path, settings: dict[str, Any]) -> Path:
    path = settings_path(workspace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def apply_user_settings(config: AppConfig) -> dict[str, Any]:
    workspace_dir = _workspace_dir(config)
    settings = load_user_settings(workspace_dir)
    for section_name, values in settings.items():
        section = getattr(config, section_name, None)
        if section is None or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if hasattr(section, key):
                setattr(section, key, value)
    return settings


def enable_autostart(*, project_root: Path, config_path: Path, workspace_dir: Path) -> Path | None:
    path = startup_script_path()
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        log_dir = workspace_dir / "service"
        log_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(
            [
                "@echo off",
                f"cd /d \"{project_root}\"",
                f"set PYTHONPATH={project_root / 'src'}",
                f"\"{sys.executable}\" -m du_research.cli --config \"{config_path}\" service start >> \"{log_dir / 'autostart.log'}\" 2>&1",
                "",
            ]
        )
        path.write_text(content, encoding="utf-8")
        return path
    except OSError:
        return None


def disable_autostart() -> bool:
    path = startup_script_path()
    if path and path.exists():
        path.unlink()
        return True
    return False


def autostart_enabled() -> bool:
    path = startup_script_path()
    return bool(path and path.exists())


def _prompt(input_fn: Callable[[str], str], label: str, default: str) -> str:
    try:
        raw = input_fn(f"{label} [{default}]: ").strip()
    except EOFError:
        return default
    return raw or default


def _prompt_bool(input_fn: Callable[[str], str], label: str, default: bool) -> bool:
    default_text = "Y/n" if default else "y/N"
    try:
        raw = input_fn(f"{label} [{default_text}]: ").strip().lower()
    except EOFError:
        return default
    if not raw:
        return default
    return raw in {"y", "yes", "true", "1"}


def default_user_settings(config: AppConfig) -> dict[str, Any]:
    return {
        "observation": {
            "enabled": True,
            "screenpipe_url": config.observation.screenpipe_url,
            "service_interval_minutes": config.observation.service_interval_minutes,
            "lookback_multiplier": config.observation.lookback_multiplier,
        },
        "daily": {
            "briefing_time": config.daily.briefing_time,
        },
        "idea": {
            "auto_research_enabled": True,
            "auto_research_top_k": config.idea.auto_research_top_k,
            "auto_research_dedupe_enabled": True,
            "auto_research_similarity_threshold": config.idea.auto_research_similarity_threshold,
            "auto_research_cooldown_days": config.idea.auto_research_cooldown_days,
        },
        "automation": {
            "enabled": True,
            "runner": "claude_code",
            "auto_execute": True,
            "headless": config.automation.headless,
            "timeout_seconds": config.automation.timeout_seconds,
        },
        "setup": {
            "autostart_enabled": True,
        },
    }


def ensure_first_run_setup(
    config: AppConfig,
    *,
    project_root: Path,
    force: bool = False,
    interactive: bool | None = None,
    input_fn: Callable[[str], str] = input,
) -> dict[str, Any]:
    workspace_dir = _workspace_dir(config)
    state_file = state_path(workspace_dir)
    if state_file.exists() and not force:
        return json.loads(state_file.read_text(encoding="utf-8"))

    settings = default_user_settings(config)
    if interactive is None:
        interactive = sys.stdin.isatty()

    if interactive:
        settings["setup"]["autostart_enabled"] = _prompt_bool(
            input_fn,
            "Enable automatic start on login",
            bool(settings["setup"]["autostart_enabled"]),
        )
        settings["automation"]["auto_execute"] = _prompt_bool(
            input_fn,
            "Let Claude Code browser automation run automatically when needed",
            bool(settings["automation"]["auto_execute"]),
        )
        settings["daily"]["briefing_time"] = _prompt(
            input_fn,
            "Daily briefing time (HH:MM)",
            str(settings["daily"]["briefing_time"]),
        )
        settings["observation"]["screenpipe_url"] = _prompt(
            input_fn,
            "Screenpipe URL",
            str(settings["observation"]["screenpipe_url"]),
        )
        fallback = _prompt(
            input_fn,
            "Fallback log file path (optional)",
            config.observation.fallback_log_path,
        )
        settings["observation"]["fallback_log_path"] = fallback

    save_user_settings(workspace_dir, settings)
    apply_user_settings(config)

    autostart_path = None
    config_path = config.config_path.resolve() if config.config_path else (project_root / "config" / "pipeline.toml")
    autostart_requested = bool(settings.get("setup", {}).get("autostart_enabled"))
    if autostart_requested:
        autostart_path = enable_autostart(
            project_root=project_root,
            config_path=config_path,
            workspace_dir=workspace_dir,
        )

    state = {
        "initialized_at": iso_now(),
        "interactive": interactive,
        "autostart_enabled": bool(autostart_path) if autostart_requested else False,
        "autostart_requested": autostart_requested,
        "autostart_path": str(autostart_path) if autostart_path else None,
        "autostart_error": None if autostart_path or not autostart_requested else "autostart_registration_failed",
        "settings_path": str(settings_path(workspace_dir)),
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state
