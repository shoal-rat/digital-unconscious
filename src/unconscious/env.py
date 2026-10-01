"""The environment an app opened from Finder does not get.

A terminal hands `dun` your shell's PATH and variables; the Dock does not. Without them
the app cannot find `claude` or `codex`, the proxy settings the crew relies on, or the
API keys of the optional crew. So, once at start-up, the app asks your login shell for
them, and takes only what it needs and has not already been given.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

WANTED = (
    "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "ZAI_API_KEY", "GLM_API_KEY", "MOONSHOT_API_KEY",
    "KIMI_API_KEY", "CLAUDE_CONFIG_DIR", "CODEX_HOME", "DUN_HOME", "LANG", "LC_ALL", "LC_MESSAGES",
)
COMMON_BINS = ("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin", "~/.claude/local", "~/.npm-global/bin",
               "~/.bun/bin", "~/.volta/bin", "~/.cargo/bin")
MARK = "__digital_unconscious_env__"


def login_env(timeout: float = 6.0) -> dict[str, str]:
    shell = os.environ.get("SHELL") or "/bin/zsh"
    try:
        proc = subprocess.run([shell, "-ilc", f"printf '\\n{MARK}\\n'; env -0"], capture_output=True,
                              timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("could not ask the login shell for its environment: %s", exc)
        return {}
    _, _, tail = proc.stdout.partition(f"\n{MARK}\n".encode())
    found: dict[str, str] = {}
    for item in tail.split(b"\0"):
        key, sep, value = item.decode("utf-8", "replace").partition("=")
        if sep and key:
            found[key] = value
    return found


def adopt_login_env() -> None:
    """Fill in PATH, proxies and keys from the login shell, without overriding anything already set."""
    found = login_env()
    paths = [p for p in found.get("PATH", "").split(os.pathsep) if p]
    paths += [str(Path(p).expanduser()) for p in COMMON_BINS]
    current = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    os.environ["PATH"] = os.pathsep.join(dict.fromkeys(paths + current))
    for key in WANTED:
        if key in found and not os.environ.get(key):
            os.environ[key] = found[key]
