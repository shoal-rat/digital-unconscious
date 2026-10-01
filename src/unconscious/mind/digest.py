"""Digest: one day's subjects → topics → long-running threads.

The model groups and names; code owns identity. It checks every ref, refuses
to create a thread whose name duplicates an existing one, and writes the
day's thread assignments atomically so a re-digest never double counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from unconscious.llm.base import LLMRequest
from unconscious.mind.prompts import DIGEST_SCHEMA, DIGEST_SYSTEM, DIGEST_TASK, LANGUAGE_LINE
from unconscious.text import clip, clock, duration, similar, tokens

if TYPE_CHECKING:
    from unconscious.app import App

INTENT_KINDS = {"search", "jot", "reading", "note"}


class DigestError(RuntimeError):
    pass


@dataclass
class DayDigest:
    day: str
    topics: list[dict[str, Any]]
    subjects: list[dict[str, Any]]
    model: str = ""
    cached: bool = False
    by_ref: dict[str, dict[str, Any]] = field(default_factory=dict)


def prepare_subjects(app: App, day: str, limit: int = 90) -> list[dict[str, Any]]:
    rows = app.store.subjects(day)
    titles: dict[str, list[str]] = {}
    for trace in app.store.traces(day):
        title = trace.get("title") or ""
        if title and title != trace["subject"]:
            bucket = titles.setdefault(trace["subject_key"], [])
            if title not in bucket and len(bucket) < 4:
                bucket.append(title)
    intent = [r for r in rows if r["kind"] in INTENT_KINDS][:40]
    focus = [
        r for r in rows
        if r["kind"] not in INTENT_KINDS and r["category"] != "system" and (r["seconds"] or 0) >= 20
    ]
    chosen = intent + focus[: max(0, limit - len(intent))]
    chosen.sort(key=lambda r: (r["kind"] not in INTENT_KINDS, -(r["seconds"] or 0)))
    for index, row in enumerate(chosen, 1):
        row["ref"] = f"S{index}"
        row["examples"] = titles.get(row["subject_key"], [])
    return chosen


def subject_line(row: dict[str, Any]) -> str:
    kind = row["kind"]
    label = clip(row["subject"], 140)
    if kind == "jot":
        return f'{row["ref"]} · jot · "{clip(row.get("body") or label, 320)}"'
    if kind == "reading" and row.get("body"):
        return f'{row["ref"]} · reading (document) · "{label}" — excerpt: "{clip(row["body"], 420)}"'
    if kind == "note":
        return f'{row["ref"]} · note · "{clip(row.get("body") or label, 240)}"'
    parts = [row["ref"]]
    seconds = row.get("seconds") or 0
    if seconds:
        parts.append(duration(seconds))
    if (row.get("visits") or 0) > 1:
        parts.append(f'{row["visits"]} visits')
    span = f"{clock(row.get('first_at', ''))}–{clock(row.get('last_at', ''))}"
    if span != "–":
        parts.append(span)
    parts.append(row.get("category") or kind)
    parts.append(f'"{label}"' if kind != "search" else f'searched "{label}"')
    if row.get("domain"):
        parts.append(row["domain"])
    line = " · ".join(parts)
    if row.get("examples"):
        line += " — e.g. " + "; ".join(clip(t, 70) for t in row["examples"][:3])
    return line


def thread_catalog(app: App, day: str, limit: int = 40) -> list[dict[str, Any]]:
    since = (date.fromisoformat(day) - timedelta(days=120)).isoformat()
    active_days: dict[int, int] = {}
    for row in app.store.thread_days(since):
        if row["day"] != day:
            active_days[row["thread_id"]] = active_days.get(row["thread_id"], 0) + 1
    threads = [t for t in app.store.threads() if t["state"] != "merged"]
    threads.sort(key=lambda t: t["last_day"] or "", reverse=True)
    threads.sort(key=lambda t: t["state"] != "pinned")  # stable: pinned first, then most recent
    out = []
    for thread in threads[:limit]:
        out.append({**thread, "ref": f"T{thread['id']}", "days": active_days.get(thread["id"], 0)})
    return out


def thread_line(thread: dict[str, Any]) -> str:
    keywords = ", ".join((thread.get("keywords") or [])[:8])
    line = f'{thread["ref"]} · "{thread["name"]}"'
    if thread.get("gist"):
        line += f" — {clip(thread['gist'], 140)}"
    if keywords:
        line += f" · keywords: {keywords}"
    line += f" · last seen {thread.get('last_day') or 'never'} · {thread.get('days', 0)} earlier days"
    return line


def digest_day(app: App, day: str, *, force: bool = False) -> DayDigest:
    subjects = prepare_subjects(app, day)
    by_ref = {s["ref"]: s for s in subjects}
    trace_count = app.store.trace_count(day)
    if not subjects:
        return DayDigest(day, [], [], cached=True)

    existing = app.store.digest(day)
    if existing and not force and existing.get("trace_count") == trace_count:
        payload = existing["payload"] or {}
        return DayDigest(day, payload.get("topics", []), subjects, existing.get("model", ""), True, by_ref)

    catalog = thread_catalog(app, day)
    settings = app.settings
    language = LANGUAGE_LINE[settings.language]
    subject_block = "SUBJECTS\n" + "\n".join(subject_line(s) for s in subjects)
    thread_block = "\n".join(thread_line(t) for t in catalog) or "(none yet: this is an early day)"
    request = LLMRequest(
        role="digest",
        system=DIGEST_SYSTEM,
        prompt=DIGEST_TASK.format(subjects=subject_block, threads=thread_block, language=language),
        schema=DIGEST_SCHEMA,
        max_tokens=4000,
        payload={
            "subjects": [
                {"ref": s["ref"], "label": s["subject"], "kind": s["kind"], "seconds": s.get("seconds") or 0,
                 "body": s.get("body") or "", "category": s.get("category") or ""}
                for s in subjects
            ],
            "threads": [{"ref": t["ref"], "name": t["name"], "keywords": t.get("keywords") or []} for t in catalog],
        },
    )
    result = app.router.call(request)
    if not result.ok or not result.data:
        raise DigestError(result.error or "the digest model returned nothing")

    topics = apply_digest(app, day, result.data, by_ref, catalog)
    app.store.save_digest(day, {"topics": topics}, trace_count, result.label)
    return DayDigest(day, topics, subjects, result.label, False, by_ref)


def _thread_ref(value: str) -> int | None:
    match = re.fullmatch(r"\s*T(\d+)\s*", value or "", re.I)
    return int(match.group(1)) if match else None


def apply_digest(
    app: App,
    day: str,
    data: dict[str, Any],
    by_ref: dict[str, dict[str, Any]],
    catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    known = {t["id"]: t for t in app.store.threads()}
    claimed: set[str] = set()
    topics: list[dict[str, Any]] = []
    per_thread: dict[int, dict[str, Any]] = {}
    subject_map: dict[str, int] = {}

    for raw in data.get("topics") or []:
        refs = [r for r in dict.fromkeys(str(x).strip() for x in raw.get("subjects") or []) if r in by_ref and r not in claimed]
        if not refs:
            continue
        claimed.update(refs)
        keywords = [str(k).strip().lower() for k in raw.get("keywords") or [] if str(k).strip()][:6]
        label = clip(str(raw.get("label") or "").strip() or by_ref[refs[0]]["subject"], 80)
        thread_value = str(raw.get("thread") or "none").strip().lower()
        thread_id = _thread_ref(thread_value)
        if thread_id is not None:
            thread = known.get(thread_id)
            if thread is None:
                thread_id = None
            elif thread["state"] == "merged" and thread.get("merged_into"):
                thread_id = thread["merged_into"]
        if thread_id is None and thread_value == "new":
            name = clip(str(raw.get("thread_name") or label).strip(), 60)
            thread_id = _find_duplicate(name, keywords, known.values())
            if thread_id is None:
                thread_id = app.store.create_thread(name, clip(str(raw.get("thread_gist") or raw.get("gist") or ""), 240), keywords, day)
                known[thread_id] = app.store.thread(thread_id) or {"id": thread_id, "name": name, "keywords": keywords, "state": "active"}
        seconds = sum(by_ref[r].get("seconds") or 0 for r in refs)
        visits = sum(by_ref[r].get("visits") or 0 for r in refs)
        topic = {
            "label": label,
            "gist": clip(str(raw.get("gist") or ""), 300),
            "thread_id": thread_id,
            "keywords": keywords,
            "subject_keys": [by_ref[r]["subject_key"] for r in refs],
            "seconds": round(seconds, 1),
            "visits": visits,
        }
        topics.append(topic)
        if thread_id is not None:
            bucket = per_thread.setdefault(thread_id, {"thread_id": thread_id, "seconds": 0.0, "visits": 0, "subjects": [], "note": ""})
            bucket["seconds"] += seconds
            bucket["visits"] += visits
            bucket["subjects"].extend(by_ref[r]["subject"] for r in refs)
            bucket["note"] = bucket["note"] or topic["gist"]
            for r in refs:
                subject_map[by_ref[r]["subject_key"]] = thread_id
            thread = known.get(thread_id)
            if thread is not None and keywords:
                merged = list(dict.fromkeys(keywords + list(thread.get("keywords") or [])))[:12]
                if merged != thread.get("keywords"):
                    app.store.update_thread(thread_id, keywords=merged)
                    thread["keywords"] = merged

    for bucket in per_thread.values():
        bucket["subjects"] = list(dict.fromkeys(bucket["subjects"]))[:12]
        bucket["seconds"] = round(bucket["seconds"], 1)
    app.store.replace_day_assignments(day, list(per_thread.values()), subject_map)
    topics.sort(key=lambda t: -t["seconds"])
    return topics


def _find_duplicate(name: str, keywords: list[str], threads) -> int | None:
    name_tokens = tokens(name)
    for thread in threads:
        if thread.get("state") == "merged":
            continue
        if similar(name, thread["name"]) >= 0.6 or (
            name_tokens and name_tokens == tokens(thread["name"])
        ):
            return thread["id"]
    return None
