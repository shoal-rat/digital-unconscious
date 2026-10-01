"""Platform samplers: what is in front of the person right now?

Each sampler returns a :class:`Sample` with the foreground app, its window
title, the active browser URL when the platform allows it, and seconds since
the last keyboard/mouse input. Nothing here touches screenshots or OCR; window
metadata is a cheaper and far less invasive signal of attention.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field

SEP = "\x1f"

MAC_APP_NAMES = {
    "Code": "Visual Studio Code",
    "zoom.us": "Zoom",
    "WeChat": "WeChat",
    "iTerm2": "iTerm",
    "Electron": "Electron app",
}
CHROMIUM_BROWSERS = {
    "Google Chrome", "Google Chrome Beta", "Microsoft Edge", "Brave Browser", "Arc", "Chromium",
    "Vivaldi", "Opera", "Dia",
}
SAFARI_LIKE = {"Safari", "Safari Technology Preview", "Orion"}
WINDOWS_APP_NAMES = {
    "chrome.exe": "Google Chrome", "msedge.exe": "Microsoft Edge", "firefox.exe": "Firefox",
    "brave.exe": "Brave Browser", "opera.exe": "Opera", "code.exe": "Visual Studio Code",
    "cursor.exe": "Cursor", "winword.exe": "Microsoft Word", "excel.exe": "Microsoft Excel",
    "powerpnt.exe": "Microsoft PowerPoint", "outlook.exe": "Microsoft Outlook", "teams.exe": "Microsoft Teams",
    "ms-teams.exe": "Microsoft Teams", "slack.exe": "Slack", "wechat.exe": "WeChat", "weixin.exe": "WeChat",
    "obsidian.exe": "Obsidian", "notion.exe": "Notion", "zotero.exe": "Zotero", "acrobat.exe": "Acrobat",
    "acrord32.exe": "Acrobat", "stata-64.exe": "Stata", "statamp-64.exe": "Stata", "rstudio.exe": "RStudio",
    "windowsterminal.exe": "Terminal", "explorer.exe": "Explorer", "spotify.exe": "Spotify",
    "telegram.exe": "Telegram", "discord.exe": "Discord", "qq.exe": "QQ", "feishu.exe": "Lark",
}


class SensorError(RuntimeError):
    """A sampler cannot run; the message is shown to the person as-is."""


@dataclass
class Sample:
    app: str
    title: str = ""
    url: str = ""
    idle_seconds: float = 0.0


@dataclass
class Capabilities:
    platform: str = ""
    app: bool = False
    titles: bool = False
    urls: dict[str, str] = field(default_factory=dict)  # browser -> ok | denied | unsupported
    idle: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "platform": self.platform, "app": self.app, "titles": self.titles,
            "urls": dict(self.urls), "idle": self.idle, "note": self.note,
        }


def _run(cmd: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


class MacSensor:
    """osascript + ioreg. Needs Accessibility permission for window titles and,
    per browser, Automation permission for the active tab URL (macOS asks once)."""

    FRONT = (
        'tell application "System Events"\n'
        "  set fp to first application process whose frontmost is true\n"
        "  set appName to name of fp\n"
        '  set winTitle to ""\n'
        "  try\n"
        "    set winTitle to name of front window of fp\n"
        "  end try\n"
        "end tell\n"
        "return appName & (ASCII character 31) & winTitle"
    )

    def __init__(self, capture_urls: bool = True):
        self.capture_urls = capture_urls
        self.capabilities = Capabilities(platform="macOS")
        self._denied_until: dict[str, float] = {}

    def sample(self) -> Sample:
        try:
            result = _run(["osascript", "-e", self.FRONT])
        except (OSError, subprocess.SubprocessError) as exc:
            raise SensorError(f"osascript unavailable: {exc}") from exc
        if result.returncode != 0:
            message = result.stderr.strip()
            if "-1743" in message or "-1719" in message or "assistive" in message.lower():
                self.capabilities.note = (
                    "Grant Accessibility (and Automation for System Events) to the app running `dun` "
                    "in System Settings → Privacy & Security."
                )
            raise SensorError(self.capabilities.note or message or "osascript failed")
        app, _, title = result.stdout.rstrip("\n").partition(SEP)
        app = MAC_APP_NAMES.get(app.strip(), app.strip())
        self.capabilities.app = bool(app)
        if title:
            self.capabilities.titles = True
        url = self._browser_url(app) if self.capture_urls else ""
        return Sample(app=app, title=title.strip(), url=url, idle_seconds=self.idle_seconds())

    def _browser_url(self, app: str) -> str:
        if app in CHROMIUM_BROWSERS:
            script = f'tell application "{app}" to return URL of active tab of front window'
        elif app in SAFARI_LIKE:
            script = f'tell application "{app}" to return URL of front document'
        else:
            if app == "Firefox":
                self.capabilities.urls[app] = "unsupported"
            return ""
        if self._denied_until.get(app, 0) > time.time():
            return ""
        try:
            result = _run(["osascript", "-e", script])
        except (OSError, subprocess.SubprocessError):
            return ""
        if result.returncode != 0:
            if "-1743" in result.stderr:
                # Not authorised. Asking every 15 seconds would be hostile; retry hourly.
                self._denied_until[app] = time.time() + 3600
                self.capabilities.urls[app] = "denied"
            return ""
        self.capabilities.urls[app] = "ok"
        return result.stdout.strip()

    def idle_seconds(self) -> float:
        try:
            out = _run(["ioreg", "-c", "IOHIDSystem", "-r", "-d", "1", "-k", "HIDIdleTime"]).stdout
        except (OSError, subprocess.SubprocessError):
            return 0.0
        match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out)
        if not match:
            return 0.0
        self.capabilities.idle = True
        return int(match.group(1)) / 1_000_000_000


class WindowsSensor:
    """Win32 via ctypes: foreground window title, owning executable, and idle time."""

    def __init__(self, capture_urls: bool = True):
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        self.kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        self.capabilities = Capabilities(platform="Windows", note="Browser URLs are not available on Windows; titles are used.")

    def sample(self) -> Sample:
        ctypes, wintypes = self.ctypes, self.wintypes
        hwnd = self.user32.GetForegroundWindow()
        length = self.user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        self.user32.GetWindowTextW(hwnd, buffer, length + 1)
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        exe = ""
        handle = self.kernel32.OpenProcess(0x1000, False, pid.value)
        if handle:
            size = wintypes.DWORD(1024)
            path = ctypes.create_unicode_buffer(1024)
            if self.kernel32.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                exe = os.path.basename(path.value)
            self.kernel32.CloseHandle(handle)
        app = WINDOWS_APP_NAMES.get(exe.lower(), exe.rsplit(".", 1)[0] if exe else "")
        self.capabilities.app = self.capabilities.titles = True
        return Sample(app=app, title=buffer.value, idle_seconds=self.idle_seconds())

    def idle_seconds(self) -> float:
        ctypes = self.ctypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if not self.user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        self.capabilities.idle = True
        return max(0.0, (self.kernel32.GetTickCount() - info.dwTime) / 1000.0)


class LinuxSensor:
    """X11 via xdotool/xprop/xprintidle. Wayland compositors do not expose the
    focused window to other programs, so the sensor reports that plainly."""

    def __init__(self, capture_urls: bool = True):
        self.capabilities = Capabilities(platform="Linux")
        if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("DISPLAY"):
            self.capabilities.note = "Wayland does not expose the focused window. Use ActivityWatch import or jots."
        elif not shutil.which("xdotool"):
            self.capabilities.note = "Install xdotool (and optionally xprintidle) to enable the sensor."

    def sample(self) -> Sample:
        if self.capabilities.note.startswith(("Wayland", "Install")):
            raise SensorError(self.capabilities.note)
        try:
            window = _run(["xdotool", "getactivewindow"]).stdout.strip()
            title = _run(["xdotool", "getwindowname", window]).stdout.strip()
            klass = _run(["xprop", "-id", window, "WM_CLASS"]).stdout if shutil.which("xprop") else ""
        except (OSError, subprocess.SubprocessError) as exc:
            raise SensorError(str(exc)) from exc
        names = re.findall(r'"([^"]+)"', klass)
        app = names[-1] if names else ""
        self.capabilities.app = bool(app)
        self.capabilities.titles = bool(title)
        return Sample(app=app, title=title, idle_seconds=self.idle_seconds())

    def idle_seconds(self) -> float:
        if not shutil.which("xprintidle"):
            return 0.0
        try:
            self.capabilities.idle = True
            return int(_run(["xprintidle"]).stdout.strip() or 0) / 1000.0
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0.0


def make_sensor(capture_urls: bool = True):
    system = platform.system()
    if system == "Darwin":
        return MacSensor(capture_urls)
    if system == "Windows":
        return WindowsSensor(capture_urls)
    return LinuxSensor(capture_urls)
