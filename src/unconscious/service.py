"""Open the app hidden in the tray at login: launchd on macOS, a systemd user unit on Linux."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from unconscious.config import app_home

LABEL = "com.digital-unconscious.dun"


def _command() -> list[str]:
    command = [sys.executable, "-m", "unconscious", "app", "--hidden"]
    if os.environ.get("DUN_HOME"):
        command[3:3] = ["--home", os.environ["DUN_HOME"]]
    return command


def _mac_plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _linux_unit() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "digital-unconscious.service"


def handle(action: str) -> int:
    system = platform.system()
    if system == "Darwin":
        return _mac(action)
    if system == "Linux":
        return _linux(action)
    print("On Windows, create a shortcut to the command below in shell:startup:")
    print("  " + " ".join(f'"{part}"' if " " in part else part for part in _command()))
    return 0


def _mac(action: str) -> int:
    path = _mac_plist()
    uid = os.getuid()
    if action == "status":
        result = subprocess.run(["launchctl", "print", f"gui/{uid}/{LABEL}"], capture_output=True, text=True)
        print("installed and loaded" if result.returncode == 0 else ("installed, not loaded" if path.exists() else "not installed"))
        return 0
    if action == "uninstall":
        subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True)
        path.unlink(missing_ok=True)
        print("Removed the login item.")
        return 0
    logs = app_home() / "logs"  # trimmed daily by housekeeping
    logs.mkdir(parents=True, exist_ok=True)
    args = "\n".join(f"    <string>{escape(part)}</string>" for part in _command())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{args}
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>{escape(str(logs / "dun.log"))}</string>
  <key>StandardErrorPath</key><string>{escape(str(logs / "dun.log"))}</string>
</dict>
</plist>
""",
        encoding="utf-8",
    )
    subprocess.run(["launchctl", "bootout", f"gui/{uid}", str(path)], capture_output=True)
    result = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Wrote {path}, but launchctl said: {result.stderr.strip()}")
        return 1
    print("Digital Unconscious will now start at login. macOS may ask for Accessibility permission for Python.")
    return 0


def _linux(action: str) -> int:
    path = _linux_unit()
    if action == "status":
        return subprocess.call(["systemctl", "--user", "status", path.name])
    if action == "uninstall":
        subprocess.run(["systemctl", "--user", "disable", "--now", path.name])
        path.unlink(missing_ok=True)
        print("Removed the user service.")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Unit]\nDescription=Digital Unconscious\n\n[Service]\n"
        f"ExecStart={' '.join(_command())}\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n",
        encoding="utf-8",
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"])
    return subprocess.call(["systemctl", "--user", "enable", "--now", path.name])
