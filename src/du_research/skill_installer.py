"""Auto-installs research skills (Claude Code agents) on first run.

Copies the .claude/agents/ definitions into the user's project so that
Claude Code can use them as subagents for literature search, citation
analysis, peer review, etc.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# All skills that ship with Digital Unconscious
SKILLS = [
    "lit-search",
    "citation-network",
    "research-gap",
    "abstract",
    "journal-match",
    "peer-review",
    "cite-verify",
    "report-template",
    "literature-scout",
    "data-hunter",
    "analyst",
]


def install_skills(project_root: Path | None = None) -> dict[str, str]:
    """Copy agent definitions to the project's .claude/agents/ directory.

    Returns a dict of skill_name -> installed_path.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    source_dir = project_root / ".claude" / "agents"
    if not source_dir.exists():
        logger.warning("No agent definitions found at %s", source_dir)
        return {}

    # Also install to user's home .claude/agents/ so they work globally
    user_agents_dir = Path.home() / ".claude" / "agents"
    user_agents_dir.mkdir(parents=True, exist_ok=True)

    installed = {}
    for agent_file in sorted(source_dir.glob("*.md")):
        target = user_agents_dir / agent_file.name
        try:
            shutil.copy2(agent_file, target)
            skill_name = agent_file.stem
            installed[skill_name] = str(target)
            logger.info("Installed skill: %s -> %s", skill_name, target)
        except Exception as exc:
            logger.warning("Failed to install skill %s: %s", agent_file.stem, exc)

    return installed


def install_mcp_config(project_root: Path | None = None) -> Path | None:
    """Copy .mcp.json to the project root if it doesn't exist."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    source = project_root / ".mcp.json"
    if not source.exists():
        return None

    # Copy to user's home directory for global availability
    target = Path.home() / ".claude" / ".mcp.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, target)
        return target
    except Exception:
        return None


def ensure_skills_installed(project_root: Path | None = None) -> dict[str, str]:
    """Install skills if they haven't been installed yet."""
    user_agents_dir = Path.home() / ".claude" / "agents"
    marker = user_agents_dir / ".du_skills_installed"

    if marker.exists():
        # Already installed — check if we need to update
        try:
            installed_version = marker.read_text(encoding="utf-8").strip()
            current_version = _get_version()
            if installed_version == current_version:
                return {}
        except Exception:
            pass

    result = install_skills(project_root)
    install_mcp_config(project_root)

    # Write marker
    user_agents_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(_get_version(), encoding="utf-8")

    return result


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("digital-unconscious")
    except Exception:
        return "dev"


def list_installed_skills() -> list[str]:
    """List all installed skill names."""
    user_agents_dir = Path.home() / ".claude" / "agents"
    if not user_agents_dir.exists():
        return []
    return sorted(f.stem for f in user_agents_dir.glob("*.md"))
