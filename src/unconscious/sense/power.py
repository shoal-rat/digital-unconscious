"""Is the machine running on battery? Answered cheaply and cached.

The tide watcher glances less often on battery, and the shore calms its
animations. macOS asks IOKit in-process; Windows asks the kernel; Linux reads
sysfs. Anything unknown is treated as mains power.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import platform
import time

_cache: tuple[float, bool] = (0.0, False)
TTL = 120.0


def on_battery() -> bool:
    global _cache
    checked, value = _cache
    if time.monotonic() - checked < TTL:
        return value
    try:
        value = _probe()
    except Exception:
        value = False
    _cache = (time.monotonic(), value)
    return value


def _probe() -> bool:
    system = platform.system()
    if system == "Darwin":
        return _mac()
    if system == "Windows":
        return _windows()
    return _linux()


def _mac() -> bool:
    iokit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit"))
    cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
    iokit.IOPSCopyPowerSourcesInfo.restype = ctypes.c_void_p
    iokit.IOPSGetProvidingPowerSourceType.restype = ctypes.c_void_p
    iokit.IOPSGetProvidingPowerSourceType.argtypes = [ctypes.c_void_p]
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    snapshot = iokit.IOPSCopyPowerSourcesInfo()
    if not snapshot:
        return False
    try:
        kind = iokit.IOPSGetProvidingPowerSourceType(snapshot)
        buffer = ctypes.create_string_buffer(64)
        if kind and cf.CFStringGetCString(kind, buffer, 64, 0x08000100):
            return buffer.value.decode() == "Battery Power"
        return False
    finally:
        cf.CFRelease(snapshot)


def _windows() -> bool:
    class Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):  # type: ignore[attr-defined]
        return False
    return status.ACLineStatus == 0


def _linux() -> bool:
    supplies = glob.glob("/sys/class/power_supply/*/online")
    if not supplies:
        return False
    for path in supplies:
        try:
            with open(path) as handle:
                if handle.read().strip() == "1":
                    return False
        except OSError:
            continue
    return True
