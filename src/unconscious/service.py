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
    # Inside Digital Unconscious.app the executable is the app itself and takes dun's arguments.
    start = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "unconscious"]
    home = ["--home", os.environ["DUN_HOME"]] if os.environ.get("DUN_HOME") else []
    return [*start, *home, "app", "--hidden"]


def _mac_plist() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _linux_unit() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / "digital-unconscious.service"


def supported() -> bool:
    return platform.system() in {"Darwin", "Linux"}


def installed() -> bool:
    """Is the app set to set out at login (a file check: cheap enough for the interface)."""
    system = platform.system()
    if system == "Darwin":
        return _mac_plist().exists()
    if system == "Linux":
        return _linux_unit().exists()
    return False


def running_as_job() -> bool:
    """This process was started by the login item (launchd names the processes of its jobs)."""
    return os.environ.get("XPC_SERVICE_NAME") == LABEL


def set_at_login(on: bool, *, start_now: bool = False) -> None:
    """Add or remove the login item. From inside the app (start_now False) the running app is left
    alone: it is not stopped when the item goes, and a copy started by the item hands over and
    leaves quietly. Raises RuntimeError with the system's own words when it refuses."""
    system = platform.system()
    if system == "Darwin":
        _mac_set(on)
    elif system == "Linux":
        _linux_set(on, start_now)
    else:
        raise RuntimeError("not on this system")


def handle(action: str) -> int:
    system = platform.system()
    if system not in {"Darwin", "Linux"}:
        print("On Windows, create a shortcut to the command below in shell:startup:")
        print("  " + " ".join(f'"{part}"' if " " in part else part for part in _command()))
        return 0
    if action == "status":
        if system == "Linux":
            return subprocess.call(["systemctl", "--user", "status", _linux_unit().name])
        result = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"], capture_output=True, text=True)
        print("installed and loaded" if result.returncode == 0 else ("installed, not loaded" if installed() else "not installed"))
        return 0
    try:
        if action == "uninstall":  # from a terminal, the background app goes with its login item
            set_at_login(False, start_now=True)
            print("Removed the login item.")
            return 0
        set_at_login(True, start_now=True)
    except RuntimeError as exc:
        print(f"The login item was not changed: {exc}")
        return 1
    if getattr(sys, "frozen", False) or system != "Darwin":
        print("Digital Unconscious will now set out at every login, quietly in the menu bar.")
    else:
        print("Digital Unconscious will now start at login. macOS may ask for Accessibility permission for Python.")
    return 0


def _mac_set(on: bool) -> None:
    path = _mac_plist()
    domain = f"gui/{os.getuid()}"
    if not on:
        if not running_as_job():  # booting out the job that is this very app would quit it mid-click
            subprocess.run(["launchctl", "bootout", domain, str(path)], capture_output=True)
        path.unlink(missing_ok=True)
        return
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
    if running_as_job():
        return  # the item is already loaded: it is this app
    subprocess.run(["launchctl", "bootout", domain, str(path)], capture_output=True)
    result = subprocess.run(["launchctl", "bootstrap", domain, str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        path.unlink(missing_ok=True)  # not loaded: neither the switch nor the next login may think otherwise
        raise RuntimeError(result.stderr.strip() or f"launchctl bootstrap exited {result.returncode}")


def _linux_set(on: bool, start_now: bool) -> None:
    path = _linux_unit()
    now = ["--now"] if start_now else []
    if not on:
        try:
            subprocess.run(["systemctl", "--user", "disable", *now, path.name], capture_output=True)
        except OSError:
            pass  # no systemd: the unit file is all there is
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Unit]\nDescription=Digital Unconscious\n\n[Service]\n"
        f"ExecStart={' '.join(_command())}\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n",
        encoding="utf-8",
    )
    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        result = subprocess.run(["systemctl", "--user", "enable", *now, path.name], capture_output=True, text=True)
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(str(exc)) from exc
    if result.returncode != 0:
        path.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        raise RuntimeError(result.stderr.strip() or f"systemctl exited {result.returncode}")
