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
# Pure-noise apps that never carry useful signal. This is noise reduction, not a
# privacy filter — the observation layer trusts the model with whatever is on screen.
_BLACKLIST_APPS = {
    "screensaver", "lock screen", "loginwindow", "systemuiserver",
}


def _is_filtered(
    frame: BehaviorFrame,
    extra_blacklist_apps: set[str] | None = None,
) -> bool:
    """Return True if the frame is pure noise or in the user's optional app blacklist."""
    app_lower = frame.app_name.lower()
    if app_lower in _BLACKLIST_APPS:
        return True
    if extra_blacklist_apps and app_lower in extra_blacklist_apps:
        return True
    return False


@dataclass
class ScreenpipeObserver:
    """Reads behaviour frames from screenpipe's local HTTP API."""

    base_url: str = _SCREENPIPE_BASE
    timeout: int = 10
    blacklist_apps: set[str] = field(default_factory=set)

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
            if not _is_filtered(frame, extra_blacklist_apps=self.blacklist_apps):
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

    blacklist_apps: set[str] = field(default_factory=set)

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

        return [f for f in frames if not _is_filtered(f, extra_blacklist_apps=self.blacklist_apps)]


# ---------------------------------------------------------------------------
# Vision observer — screenshot → multimodal model (no OCR dependency)
# ---------------------------------------------------------------------------


VISION_PROMPT = (
    "Look at this screenshot of a knowledge worker's screen and report what they "
    "are working on, as JSON only:\n"
    "- app: the foreground application or website (short)\n"
    "- window: the document/page title or main subject (short)\n"
    "- topics: 3-6 specific topics or keywords visible\n"
    "- intent: one sentence on what the user is trying to do\n"
    "- cross_domain_hints: 0-3 connections to other fields, if any\n"
    'Output ONLY JSON: {"app":"","window":"","topics":[],"intent":"","cross_domain_hints":[]}'
)

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "app": {"type": "string"},
        "window": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "intent": {"type": "string"},
        "cross_domain_hints": {"type": "array", "items": {"type": "string"}},
    },
}


def capture_available() -> bool:
    """Whether a screenshot backend (mss or Pillow) is importable."""
    try:
        import mss  # noqa: F401
        return True
    except ImportError:
        try:
            from PIL import ImageGrab  # noqa: F401
            return True
        except ImportError:
            return False


def _maybe_downscale_png(png_bytes: bytes, max_dim: int) -> bytes:
    """Downscale a PNG so its largest side is <= max_dim (keeps vision cost sane)."""
    try:
        import io
        from PIL import Image
    except ImportError:
        return png_bytes
    try:
        image = Image.open(io.BytesIO(png_bytes))
        width, height = image.size
        scale = min(1.0, max_dim / max(width, height)) if max(width, height) else 1.0
        if scale < 1.0:
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
        out = io.BytesIO()
        image.convert("RGB").save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png_bytes


def capture_screenshot(max_dimension: int = 1568) -> bytes | None:
    """Capture the primary screen as downscaled PNG bytes, or None if unavailable."""
    try:
        import mss
        import mss.tools
        with mss.mss() as sct:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            raw = sct.grab(monitor)
            png = mss.tools.to_png(raw.rgb, raw.size)
            return _maybe_downscale_png(png, max_dimension)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("mss screenshot failed (%s); trying Pillow", exc)
    try:
        import io
        from PIL import ImageGrab
        image = ImageGrab.grab()
        out = io.BytesIO()
        image.convert("RGB").save(out, format="PNG")
        return _maybe_downscale_png(out.getvalue(), max_dimension)
    except Exception as exc:
        logger.warning("Pillow screenshot failed: %s", exc)
        return None


def _vision_frame_text(data: dict[str, Any]) -> str:
    topics = ", ".join(str(t) for t in (data.get("topics") or [])[:6])
    hints = ", ".join(str(h) for h in (data.get("cross_domain_hints") or [])[:3])
    parts: list[str] = []
    if data.get("intent"):
        parts.append(str(data["intent"]))
    if topics:
        parts.append(f"Topics: {topics}")
    if hints:
        parts.append(f"Cross-domain: {hints}")
    return " | ".join(parts) or str(data.get("window", ""))


@dataclass
class VisionObserver:
    """Reads the screen by sending a screenshot to a multimodal model.

    Replaces OCR/screenpipe: each capture grabs the current screen and asks a
    vision-capable model what the user is working on, returning a normal
    BehaviorFrame so the rest of the pipeline is unchanged.
    """

    backend: Any
    model: str = "sonnet"
    max_dimension: int = 1568

    def is_available(self) -> bool:
        return capture_available()

    def capture(self) -> list[BehaviorFrame]:
        image = capture_screenshot(self.max_dimension)
        if image is None:
            logger.warning("Vision observer: no screenshot backend (pip install digital-unconscious[vision])")
            return []
        response = self.backend.call(
            VISION_PROMPT,
            mode="strict",
            model=self.model,
            max_tokens=600,
            images=[image],
            json_schema=VISION_SCHEMA,
        )
        if not response.ok:
            logger.warning("Vision model call failed: %s", response.raw.get("error"))
            return []
        data = response.structured
        if not isinstance(data, dict):
            try:
                text = response.text.strip()
                start, end = text.find("{"), text.rfind("}") + 1
                data = json.loads(text[start:end]) if start >= 0 and end > start else {}
            except (json.JSONDecodeError, ValueError):
                data = {}
        if not isinstance(data, dict) or not data:
            return []
        return [BehaviorFrame(
            timestamp=iso_now(),
            app_name=str(data.get("app", "screen"))[:80] or "screen",
            window_title=str(data.get("window", ""))[:200],
            text_content=_vision_frame_text(data)[:1000],
            frame_type="vision",
        )]


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
