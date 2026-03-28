"""Screen observation layer — integrates with screenpipe for passive capture.

The observation layer is the sensory input of Digital Unconscious.  It reads
from screenpipe's local database (SQLite) via its HTTP API, extracts raw
behaviour frames, and filters/deduplicates them before handing off to the
compression layer.

If screenpipe is not running or not installed, the layer falls back to
reading plain-text daily-log files so the rest of the pipeline still works.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from du_research.net import fetch_json
from du_research.utils import iso_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BehaviorFrame:
    """One unit of observed screen behaviour."""

    timestamp: str
    app_name: str
    window_title: str
    text_content: str
    dwell_seconds: float = 0.0
    frame_type: str = "screen"  # "screen" | "search" | "audio"
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "app_name": self.app_name,
            "window_title": self.window_title,
            "text_content": self.text_content[:500],
            "dwell_seconds": self.dwell_seconds,
            "frame_type": self.frame_type,
            "url": self.url,
        }


# ---------------------------------------------------------------------------
# Screenpipe client
# ---------------------------------------------------------------------------

_SCREENPIPE_BASE = "http://localhost:3030"

# Apps / titles that are filtered out by default
_BLACKLIST_APPS = {
    "screensaver", "lock screen", "loginwindow", "systemuiserver",
}
_BLACKLIST_PATTERNS = [
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"bank|payment|checkout", re.IGNORECASE),
]


def _is_filtered(frame: BehaviorFrame) -> bool:
    """Return True if the frame should be dropped (privacy / noise)."""
    if frame.app_name.lower() in _BLACKLIST_APPS:
        return True
    combined = f"{frame.window_title} {frame.text_content}"
    return any(pat.search(combined) for pat in _BLACKLIST_PATTERNS)


@dataclass
class ScreenpipeObserver:
    """Reads behaviour frames from screenpipe's local HTTP API."""

    base_url: str = _SCREENPIPE_BASE
    timeout: int = 10

    def fetch_recent(
        self,
        minutes: int = 30,
        limit: int = 100,
        content_type: str = "ocr",
    ) -> list[BehaviorFrame]:
        """Fetch frames from the last *minutes* minutes."""
        now = datetime.now(timezone.utc)
        start = (now - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        try:
            data = fetch_json(
                f"{self.base_url}/search",
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        except Exception:
            # Try query-param style
            try:
                url = (
                    f"{self.base_url}/search"
                    f"?content_type={content_type}"
                    f"&start_time={start}"
                    f"&end_time={end}"
                    f"&limit={limit}"
                )
                data = fetch_json(url, timeout=self.timeout)
            except Exception as exc:
                logger.warning("screenpipe not reachable: %s", exc)
                return []

        frames: list[BehaviorFrame] = []
        for item in data.get("data", []):
            content = item.get("content", {})
            frame = BehaviorFrame(
                timestamp=content.get("timestamp", iso_now()),
                app_name=content.get("app_name", "unknown"),
                window_title=content.get("window_name", ""),
                text_content=content.get("text", "")[:1000],
                frame_type=item.get("type", "OCR").lower(),
            )
            if not _is_filtered(frame):
                frames.append(frame)
        return frames

    def is_available(self) -> bool:
        """Check whether screenpipe is running."""
        try:
            fetch_json(f"{self.base_url}/health", timeout=3)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Fallback: file-based observation
# ---------------------------------------------------------------------------


@dataclass
class FileObserver:
    """Reads behaviour frames from a plain-text or JSONL log file.

    Each line becomes a frame.  This is the fallback when screenpipe is
    unavailable and the user supplies a daily log.
    """

    def read(self, path: Path) -> list[BehaviorFrame]:
        if not path.exists():
            return []

        frames: list[BehaviorFrame] = []
        text = path.read_text(encoding="utf-8", errors="replace")

        if path.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frames.append(BehaviorFrame(
                    timestamp=obj.get("timestamp", iso_now()),
                    app_name=obj.get("app_name", obj.get("app", "unknown")),
                    window_title=obj.get("window_title", obj.get("title", "")),
                    text_content=obj.get("text", obj.get("content", ""))[:1000],
                    dwell_seconds=float(obj.get("dwell_seconds", 0)),
                    frame_type=obj.get("type", "log"),
                    url=obj.get("url"),
                ))
        else:
            # Plain-text: each non-empty line is a frame
            for line in text.splitlines():
                line = line.strip()
                if len(line) < 10:
                    continue
                frames.append(BehaviorFrame(
                    timestamp=iso_now(),
                    app_name="text_log",
                    window_title="",
                    text_content=line[:1000],
                    frame_type="log",
                ))

        return [f for f in frames if not _is_filtered(f)]


# ---------------------------------------------------------------------------
# Sliding-window grouper
# ---------------------------------------------------------------------------


def group_into_windows(
    frames: list[BehaviorFrame],
    window_minutes: int = 30,
) -> list[list[BehaviorFrame]]:
    """Group *frames* into time-based windows of *window_minutes*."""
    if not frames:
        return []

    # Sort by timestamp string (ISO 8601 sorts lexicographically)
    sorted_frames = sorted(frames, key=lambda f: f.timestamp)
    windows: list[list[BehaviorFrame]] = []
    current_window: list[BehaviorFrame] = [sorted_frames[0]]

    for frame in sorted_frames[1:]:
        # Simple heuristic: if timestamps are parseable, check gap
        try:
            prev_ts = datetime.fromisoformat(current_window[0].timestamp.replace("Z", "+00:00"))
            curr_ts = datetime.fromisoformat(frame.timestamp.replace("Z", "+00:00"))
            if (curr_ts - prev_ts).total_seconds() > window_minutes * 60:
                windows.append(current_window)
                current_window = []
        except (ValueError, TypeError):
            pass
        current_window.append(frame)

    if current_window:
        windows.append(current_window)
    return windows


def deduplicate_frames(frames: list[BehaviorFrame]) -> list[BehaviorFrame]:
    """Remove consecutive duplicate frames (same app + title + content)."""
    if not frames:
        return []
    result = [frames[0]]
    for frame in frames[1:]:
        prev = result[-1]
        if (
            frame.app_name == prev.app_name
            and frame.window_title == prev.window_title
            and frame.text_content == prev.text_content
        ):
            # Merge dwell time
            result[-1] = BehaviorFrame(
                timestamp=prev.timestamp,
                app_name=prev.app_name,
                window_title=prev.window_title,
                text_content=prev.text_content,
                dwell_seconds=prev.dwell_seconds + frame.dwell_seconds,
                frame_type=prev.frame_type,
                url=prev.url or frame.url,
            )
        else:
            result.append(frame)
    return result
