"""The macOS tide watcher without spawning processes.

Every glance used to start `osascript` twice and `ioreg` once. Here the same
answers come from in-process calls:

  front app  CGWindowListCopyWindowInfo, front-to-back, first normal window
             (needs no permission)
  title      the Accessibility API on that app's focused window
             (needs Accessibility, exactly like System Events did)
  idle       CGEventSourceSecondsSinceLastEventType
  address    still AppleScript (browsers expose it no other way), but only
             when the app or title changed since the last ask

All Core Foundation objects created here are released before returning.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import subprocess
import time

from unconscious.sense.sensors import (
    CHROMIUM_BROWSERS,
    MAC_APP_NAMES,
    SAFARI_LIKE,
    Capabilities,
    Sample,
    SensorError,
)

c_void_p, c_int32, c_long, c_uint32, c_double = ctypes.c_void_p, ctypes.c_int32, ctypes.c_long, ctypes.c_uint32, ctypes.c_double
UTF8 = 0x08000100
NUMBER_SINT32 = 3
ON_SCREEN_ONLY, EXCLUDE_DESKTOP = 1 << 0, 1 << 4
SYSTEM_OWNERS = {"Window Server", "Dock", "Control Centre", "Control Center", "SystemUIServer", "Notification Center",
                 "Spotlight", "WindowManager", "TextInputMenuAgent"}


def _load(name: str):
    path = ctypes.util.find_library(name)
    if not path:
        raise OSError(f"{name} framework not found")
    return ctypes.cdll.LoadLibrary(path)


class _Frameworks:
    def __init__(self) -> None:
        cf = _load("CoreFoundation")
        cg = _load("CoreGraphics")
        ax = _load("ApplicationServices")
        cf.CFStringCreateWithCString.restype = c_void_p
        cf.CFStringCreateWithCString.argtypes = [c_void_p, ctypes.c_char_p, c_uint32]
        cf.CFStringGetCString.argtypes = [c_void_p, ctypes.c_char_p, c_long, c_uint32]
        cf.CFStringGetLength.restype = c_long
        cf.CFStringGetLength.argtypes = [c_void_p]
        cf.CFArrayGetCount.restype = c_long
        cf.CFArrayGetCount.argtypes = [c_void_p]
        cf.CFArrayGetValueAtIndex.restype = c_void_p
        cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, c_long]
        cf.CFDictionaryGetValue.restype = c_void_p
        cf.CFDictionaryGetValue.argtypes = [c_void_p, c_void_p]
        cf.CFNumberGetValue.argtypes = [c_void_p, ctypes.c_int, c_void_p]
        cf.CFGetTypeID.restype = c_long
        cf.CFGetTypeID.argtypes = [c_void_p]
        cf.CFStringGetTypeID.restype = c_long
        cf.CFRelease.argtypes = [c_void_p]
        cg.CGWindowListCopyWindowInfo.restype = c_void_p
        cg.CGWindowListCopyWindowInfo.argtypes = [c_uint32, c_uint32]
        cg.CGEventSourceSecondsSinceLastEventType.restype = c_double
        cg.CGEventSourceSecondsSinceLastEventType.argtypes = [c_int32, c_uint32]
        ax.AXIsProcessTrusted.restype = ctypes.c_bool
        ax.AXUIElementCreateApplication.restype = c_void_p
        ax.AXUIElementCreateApplication.argtypes = [c_int32]
        ax.AXUIElementCopyAttributeValue.restype = c_int32
        ax.AXUIElementCopyAttributeValue.argtypes = [c_void_p, c_void_p, ctypes.POINTER(c_void_p)]
        ax.AXUIElementSetMessagingTimeout.argtypes = [c_void_p, ctypes.c_float]
        self.cf, self.cg, self.ax = cf, cg, ax
        self.keys = {name: self.cfstr(name) for name in (
            "kCGWindowLayer", "kCGWindowOwnerName", "kCGWindowOwnerPID", "AXFocusedWindow", "AXTitle",
        )}
        self.string_type = cf.CFStringGetTypeID()

    def cfstr(self, text: str) -> int:
        return self.cf.CFStringCreateWithCString(None, text.encode("utf-8"), UTF8)

    def to_str(self, ref: int | None) -> str:
        if not ref or self.cf.CFGetTypeID(ref) != self.string_type:
            return ""
        length = self.cf.CFStringGetLength(ref)
        size = length * 4 + 1
        buffer = ctypes.create_string_buffer(size)
        if self.cf.CFStringGetCString(ref, buffer, size, UTF8):
            return buffer.value.decode("utf-8", "replace")
        return ""

    def to_int(self, ref: int | None) -> int:
        value = c_int32(0)
        if ref:
            self.cf.CFNumberGetValue(ref, NUMBER_SINT32, ctypes.byref(value))
        return value.value


def ask_for_accessibility() -> bool:
    """Show macOS's own "allow Accessibility" prompt if this app may not read window titles yet.
    Returns whether it already may. Only the bundled app asks: from a terminal, the prompt would
    name the terminal instead."""
    try:
        cf, ax = _load("CoreFoundation"), _load("ApplicationServices")
        ax.AXIsProcessTrusted.restype = ctypes.c_bool
        if ax.AXIsProcessTrusted():
            return True
        prompt = c_void_p.in_dll(ax, "kAXTrustedCheckOptionPrompt").value
        true = c_void_p.in_dll(cf, "kCFBooleanTrue").value
        cf.CFDictionaryCreate.restype = c_void_p
        cf.CFDictionaryCreate.argtypes = [c_void_p, ctypes.POINTER(c_void_p), ctypes.POINTER(c_void_p), c_long, c_void_p, c_void_p]
        keys, values = (c_void_p * 1)(prompt), (c_void_p * 1)(true)
        options = cf.CFDictionaryCreate(None, keys, values, 1, None, None)
        ax.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
        ax.AXIsProcessTrustedWithOptions.argtypes = [c_void_p]
        return bool(ax.AXIsProcessTrustedWithOptions(options))
    except (OSError, ValueError, AttributeError):
        return False


class NativeMacSensor:
    """Same contract as MacSensor, a few hundred times cheaper per glance."""

    def __init__(self, capture_urls: bool = True):
        self.fw = _Frameworks()
        self.capture_urls = capture_urls
        self.capabilities = Capabilities(platform="macOS")
        self._last_key: tuple[str, str] | None = None
        self._last_url = ""
        self._denied_until: dict[str, float] = {}

    def front_app(self) -> tuple[str, int]:
        fw = self.fw
        windows = fw.cg.CGWindowListCopyWindowInfo(ON_SCREEN_ONLY | EXCLUDE_DESKTOP, 0)
        if not windows:
            raise SensorError("The window server did not answer.")
        try:
            for index in range(fw.cf.CFArrayGetCount(windows)):
                info = fw.cf.CFArrayGetValueAtIndex(windows, index)
                if fw.to_int(fw.cf.CFDictionaryGetValue(info, fw.keys["kCGWindowLayer"])) != 0:
                    continue
                owner = fw.to_str(fw.cf.CFDictionaryGetValue(info, fw.keys["kCGWindowOwnerName"]))
                if not owner or owner in SYSTEM_OWNERS:
                    continue
                return owner, fw.to_int(fw.cf.CFDictionaryGetValue(info, fw.keys["kCGWindowOwnerPID"]))
        finally:
            fw.cf.CFRelease(windows)
        return "", 0

    def window_title(self, pid: int) -> str:
        fw = self.fw
        if not pid or not fw.ax.AXIsProcessTrusted():
            self.capabilities.note = (
                "Grant Accessibility to the app running Digital Unconscious in System Settings → "
                "Privacy & Security to read window titles."
            )
            return ""
        app = fw.ax.AXUIElementCreateApplication(pid)
        if not app:
            return ""
        window = c_void_p()
        title = c_void_p()
        try:
            fw.ax.AXUIElementSetMessagingTimeout(app, 0.5)  # a hung app must not stall the watcher
            if fw.ax.AXUIElementCopyAttributeValue(app, fw.keys["AXFocusedWindow"], ctypes.byref(window)) != 0 or not window.value:
                return ""
            if fw.ax.AXUIElementCopyAttributeValue(window.value, fw.keys["AXTitle"], ctypes.byref(title)) != 0:
                return ""
            return fw.to_str(title.value)
        finally:
            if title.value:
                fw.cf.CFRelease(title.value)
            if window.value:
                fw.cf.CFRelease(window.value)
            fw.cf.CFRelease(app)

    def idle_seconds(self) -> float:
        # kCGEventSourceStateHIDSystemState = 1, kCGAnyInputEventType = ~0
        self.capabilities.idle = True
        return float(self.fw.cg.CGEventSourceSecondsSinceLastEventType(1, 0xFFFFFFFF))

    def sample(self) -> Sample:
        owner, pid = self.front_app()
        app = MAC_APP_NAMES.get(owner, owner)
        self.capabilities.app = bool(app)
        title = self.window_title(pid)
        if title:
            self.capabilities.titles = True
            self.capabilities.note = ""
        url = ""
        if self.capture_urls and (app in CHROMIUM_BROWSERS or app in SAFARI_LIKE):
            key = (app, title)
            if key != self._last_key:  # same page, same address: no need to ask again
                self._last_url = self._browser_url(app)
                self._last_key = key
            url = self._last_url
        else:
            self._last_key = None
        return Sample(app=app, title=title.strip(), url=url, idle_seconds=self.idle_seconds())

    def _browser_url(self, app: str) -> str:
        if self._denied_until.get(app, 0) > time.time():
            return ""
        if app in CHROMIUM_BROWSERS:
            script = f'tell application "{app}" to return URL of active tab of front window'
        else:
            script = f'tell application "{app}" to return URL of front document'
        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            return ""
        if result.returncode != 0:
            if "-1743" in result.stderr:
                self._denied_until[app] = time.time() + 3600
                self.capabilities.urls[app] = "denied"
            return ""
        self.capabilities.urls[app] = "ok"
        return result.stdout.strip()
