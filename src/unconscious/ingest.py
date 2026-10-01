"""Other ways in: jots, fed documents, plain activity logs, and ActivityWatch.

Jots and fed documents are *conscious* signals (the person chose to record
them), so the digest treats them as the strongest evidence of intent.
Everything imported from another tracker passes through the same privacy
filters as the live sensor.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import statistics
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unconscious.sense.privacy import redact
from unconscious.sense.subjects import describe, is_browser
from unconscious.text import clip

if TYPE_CHECKING:
    from unconscious.app import App

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".tex", ".org", ".log", ".text"}
MAX_FEED_BYTES = 40 * 1024 * 1024


class IngestError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now().astimezone()


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:12]


def jot(app: App, text: str, when: datetime | None = None) -> int:
    text = redact((text or "").strip())
    if not text:
        raise IngestError("Empty jot.")
    when = when or _now()
    stamp = when.isoformat(timespec="seconds")
    return app.store.add_trace(
        day=when.date().isoformat(),
        started_at=stamp,
        ended_at=stamp,
        seconds=0.0,
        kind="jot",
        source="jot",
        category="note",
        body=clip(text, 4000),
        subject_key=f"jot:{_digest(text)}",
        subject=clip(text, 140),
    )


def _excerpt(text: str, limit: int = 1800) -> str:
    """Prefer the abstract of a paper; otherwise the opening."""
    head = text[:6000]
    match = re.search(r"\babstract\b[:.\s]*", head, re.I)
    start = match.end() if match and match.start() < 3000 else 0
    return clip(text[start:], limit)


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines()[:40]:
        line = line.strip().lstrip("#").strip()
        if 8 <= len(line) <= 200 and not line.lower().startswith(("arxiv:", "http", "doi")):
            return clip(line, 140)
    return fallback


def _looks_like_log(lines: list[str]) -> bool:
    """Short independent lines (a hand-written day log) rather than prose."""
    if len(lines) < 2:
        return False
    lengths = [len(line) for line in lines]
    headings = sum(1 for line in lines if line.startswith("#"))
    return statistics.median(lengths) < 240 and headings <= 1 and len(lines) <= 400


def read_pdf(data: bytes) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise IngestError("PDF support needs the optional extra: pip install 'digital-unconscious[pdf]'") from exc
    reader = PdfReader(io.BytesIO(data))
    title = ""
    try:
        title = str((reader.metadata or {}).get("/Title") or "").strip()
    except Exception:
        title = ""
    pages = []
    for page in reader.pages[:6]:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n".join(pages).strip()
    if not text:
        raise IngestError("This PDF has no extractable text (it may be a scan). Run OCR first.")
    return title, text


def feed_bytes(app: App, filename: str, data: bytes, when: datetime | None = None) -> dict[str, Any]:
    if len(data) > MAX_FEED_BYTES:
        raise IngestError("File is larger than 40 MB.")
    when = when or _now()
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem
    if suffix == ".pdf":
        title, text = read_pdf(data)
        return {"kind": "reading", "traces": [_reading(app, title or _title_from_text(text, stem), text, when)]}
    if suffix in {".json", ".jsonl"}:
        return import_records(app, data.decode("utf-8", "replace"))
    if suffix in TEXT_SUFFIXES or not suffix:
        text = data.decode("utf-8", "replace")
        lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 8]
        if suffix in {".txt", ".log", ""} and _looks_like_log(lines):
            ids = [_note(app, line, when) for line in lines]
            return {"kind": "notes", "traces": ids}
        return {"kind": "reading", "traces": [_reading(app, _title_from_text(text, stem), text, when)]}
    raise IngestError(f"Unsupported file type {suffix!r}. Use PDF, text, Markdown, or an ActivityWatch JSON export.")


def feed_path(app: App, path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise IngestError(f"No such file: {path}")
    return feed_bytes(app, path.name, path.read_bytes())


def _reading(app: App, title: str, text: str, when: datetime) -> int:
    stamp = when.isoformat(timespec="seconds")
    title = redact(title)
    return app.store.add_trace(
        day=when.date().isoformat(),
        started_at=stamp,
        ended_at=stamp,
        seconds=0.0,
        kind="reading",
        source="feed",
        category="reading",
        title=title,
        body=redact(_excerpt(text)),
        subject_key=f"doc:{_digest(text[:4000])}",
        subject=title,
    )


def _note(app: App, line: str, when: datetime) -> int:
    stamp = when.isoformat(timespec="seconds")
    line = redact(line)
    return app.store.add_trace(
        day=when.date().isoformat(),
        started_at=stamp,
        ended_at=stamp,
        seconds=0.0,
        kind="note",
        source="log",
        category="note",
        body=clip(line, 1000),
        subject_key=f"note:{_digest(line)}",
        subject=clip(line, 140),
    )


# ------------------------------------------------------------------ ActivityWatch


def _parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    except (AttributeError, ValueError):
        return None


def _records_from_buckets(buckets: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ActivityWatch buckets into generic attention records."""
    window, web, afk = [], [], []
    for bucket_id, bucket in buckets.items():
        kind = bucket.get("type", "")
        events = bucket.get("events") or []
        if kind == "currentwindow" or bucket_id.startswith("aw-watcher-window"):
            window.extend(events)
        elif kind == "web.tab.current" or bucket_id.startswith("aw-watcher-web"):
            web.extend(events)
        elif kind == "afkstatus" or bucket_id.startswith("aw-watcher-afk"):
            afk.extend(events)
    away: list[tuple[datetime, datetime]] = []
    for event in afk:
        if (event.get("data") or {}).get("status") == "afk":
            start = _parse_time(event.get("timestamp", ""))
            if start:
                away.append((start, start + timedelta(seconds=float(event.get("duration") or 0))))

    def present(start: datetime, seconds: float) -> bool:
        middle = start + timedelta(seconds=seconds / 2)
        return not any(a <= middle <= b for a, b in away)

    records = []
    for event in web:
        data = event.get("data") or {}
        if data.get("incognito"):
            continue
        start = _parse_time(event.get("timestamp", ""))
        seconds = float(event.get("duration") or 0)
        if start and seconds >= 1 and present(start, seconds):
            records.append({"timestamp": start, "duration": seconds, "app": "Browser",
                            "title": data.get("title", ""), "url": data.get("url", "")})
    for event in window:
        data = event.get("data") or {}
        app = data.get("app", "")
        if web and is_browser(app):
            continue  # the web watcher already covered browser time, with URLs
        start = _parse_time(event.get("timestamp", ""))
        seconds = float(event.get("duration") or 0)
        if start and seconds >= 1 and present(start, seconds):
            records.append({"timestamp": start, "duration": seconds, "app": app,
                            "title": data.get("title", ""), "url": ""})
    return records


def import_records(app: App, text: str, source: str = "import") -> dict[str, Any]:
    """Accept an ActivityWatch export ({"buckets": …}) or JSON/JSONL records
    with timestamp, duration (seconds), app, title and optional url."""
    text = text.strip()
    records: list[dict[str, Any]] = []
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "buckets" in data:
            records = _records_from_buckets(data["buckets"])
            source = "activitywatch"
        elif isinstance(data, list):
            records = data
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]
    return _store_records(app, records, source)


def _store_records(app: App, records: list[dict[str, Any]], source: str) -> dict[str, Any]:
    s = app.settings.sense
    timed = []
    for record in records:
        start = record["timestamp"] if isinstance(record.get("timestamp"), datetime) else _parse_time(str(record.get("timestamp", "")))
        if start is not None:
            timed.append((start, record))
    timed.sort(key=lambda pair: pair[0])
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for start, record in timed:
        seconds = min(float(record.get("duration") or record.get("seconds") or 0), 3600.0)
        subject = describe(
            str(record.get("app") or ""), str(record.get("title") or ""), str(record.get("url") or ""),
            quiet_apps=s.quiet_apps, quiet_domains=s.quiet_domains, private_apps=s.private_apps,
            capture_titles=s.capture_titles, capture_urls=s.capture_urls,
        )
        if subject is None or seconds <= 0:
            continue
        day = start.date().isoformat()
        end = start + timedelta(seconds=seconds)
        slot = merged.get((day, subject.key))
        # Collapse an event stream into one trace per subject per contiguous run.
        if slot and (start - slot["end"]).total_seconds() <= 120:
            slot["end"] = max(slot["end"], end)
            slot["seconds"] += seconds
            continue
        if slot:
            _flush(app, slot, source)
        merged[(day, subject.key)] = {"subject": subject, "start": start, "end": end, "seconds": seconds, "day": day}
    for slot in merged.values():
        _flush(app, slot, source)
    days = sorted({slot["day"] for slot in merged.values()})
    return {"kind": source, "traces": len(merged), "days": days}


def _flush(app: App, slot: dict[str, Any], source: str) -> None:
    subject = slot["subject"]
    app.store.add_trace(
        day=slot["day"],
        started_at=slot["start"].isoformat(timespec="seconds"),
        ended_at=slot["end"].isoformat(timespec="seconds"),
        seconds=round(slot["seconds"], 1),
        kind=subject.kind,
        source=source,
        app=subject.app,
        category=subject.category,
        title=subject.title,
        url=subject.url,
        domain=subject.domain,
        subject_key=subject.key,
        subject=subject.label,
    )


def import_activitywatch(app: App, day: str, base_url: str = "http://localhost:5600") -> dict[str, Any]:
    """Pull one day from a running ActivityWatch server, replacing any earlier import of that day."""
    start = datetime.combine(date.fromisoformat(day), datetime.min.time()).astimezone()
    end = start + timedelta(days=1)
    try:
        with urllib.request.urlopen(f"{base_url}/api/0/buckets/", timeout=5) as resp:
            buckets = json.loads(resp.read())
        full: dict[str, Any] = {}
        for bucket_id, meta in buckets.items():
            query = urllib.parse.urlencode({"start": start.isoformat(), "end": end.isoformat(), "limit": -1})
            with urllib.request.urlopen(f"{base_url}/api/0/buckets/{urllib.parse.quote(bucket_id)}/events?{query}", timeout=15) as resp:
                full[bucket_id] = {**meta, "events": json.loads(resp.read())}
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise IngestError(f"ActivityWatch is not reachable at {base_url}: {exc}") from exc
    app.store.delete_traces(day, "activitywatch")
    return _store_records(app, _records_from_buckets(full), "activitywatch")
