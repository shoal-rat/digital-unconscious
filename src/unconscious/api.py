"""View models for the dashboard: plain functions from the store to JSON-ready dicts."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from unconscious import __version__
from unconscious.mind.signals import compute_signals, thread_stats
from unconscious.mind.taste import load_taste
from unconscious.sense.watcher import pause_state
from unconscious.store import today
from unconscious.text import clip

if TYPE_CHECKING:
    from unconscious.app import App


def _minutes(iso: str) -> float:
    try:
        moment = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return 0.0
    return moment.hour * 60 + moment.minute + moment.second / 60


def sensor_status(app: App) -> dict[str, Any]:
    status = dict(app.store.get("sensor", {}) or {})
    heartbeat = status.get("heartbeat")
    stale = True
    if heartbeat:
        try:
            age = (datetime.now().astimezone() - datetime.fromisoformat(heartbeat)).total_seconds()
            stale = age > max(90, 4 * int(status.get("interval") or 15))
        except ValueError:
            pass
    if stale and status.get("state") not in {None, "stopped"}:
        status["state"] = "stopped"
    status.setdefault("state", "never")
    status.update(pause_state(app))
    return status


def state(app: App) -> dict[str, Any]:
    day = today()
    settings = app.settings
    latest = app.store.latest_dream()
    routes = app.router.describe()["routes"]
    return {
        "version": __version__,
        "today": day,
        "language": settings.language,
        "name": settings.you.name,
        "personalised": bool(settings.you.persona or settings.you.focus),
        "sensor": sensor_status(app),
        "today_stats": {
            "seconds": round(app.store.day_seconds(day)),
            "traces": app.store.trace_count(day),
        },
        "dreamt_today": app.store.dream(day) is not None,
        "latest_dream": latest["day"] if latest else None,
        "has_memory": app.store.trace_count() > 0,
        "jobs": app.store.active_jobs(),
        "models_ready": all(routes[r]["chain"] for r in ("digest", "dream")),
        "dream_time": settings.dream.time,
        "counts": {
            "threads": len([t for t in app.store.threads() if t["state"] not in {"merged"}]),
            "new_sparks": len(app.store.sparks(status="new")),
            "dreams": len(app.store.dreams(limit=1000)),
        },
    }


def day_view(app: App, day: str) -> dict[str, Any]:
    threads = {t["id"]: t for t in app.store.threads()}
    mapping = app.store.subject_threads(day)
    subjects = []
    for row in app.store.subjects(day)[:60]:
        thread = threads.get(mapping.get(row["subject_key"], -1))
        subjects.append({
            "key": row["subject_key"], "label": row["subject"], "kind": row["kind"], "category": row["category"],
            "seconds": round(row["seconds"] or 0), "visits": row["visits"], "domain": row["domain"],
            "url": row["url"], "body": clip(row.get("body") or "", 400),
            "first": row["first_at"], "last": row["last_at"],
            "thread_id": thread["id"] if thread else None, "thread": thread["name"] if thread else None,
            "hue": thread["hue"] if thread else None,
        })
    segments: list[dict[str, Any]] = []
    marks: list[dict[str, Any]] = []
    for trace in app.store.traces(day):
        thread = threads.get(mapping.get(trace["subject_key"], -1))
        start, end = _minutes(trace["started_at"]), _minutes(trace["ended_at"])
        if trace["kind"] in {"jot", "reading", "note"} and (trace["seconds"] or 0) == 0:
            marks.append({"at": round(start, 1), "kind": trace["kind"], "label": clip(trace["subject"], 90),
                          "hue": thread["hue"] if thread else None})
            continue
        if end - start < 0.2:
            continue
        hue = thread["hue"] if thread else None
        last = segments[-1] if segments else None
        if last and last["hue"] == hue and last["category"] == trace["category"] and start - last["end"] < 2:
            last["end"] = round(end, 1)
            if trace["subject"] not in last["labels"] and len(last["labels"]) < 4:
                last["labels"].append(clip(trace["subject"], 70))
            continue
        segments.append({
            "start": round(start, 1), "end": round(end, 1), "hue": hue, "category": trace["category"],
            "thread": thread["name"] if thread else None, "labels": [clip(trace["subject"], 70)],
        })
    digest = app.store.digest(day)
    return {
        "day": day,
        "seconds": round(app.store.day_seconds(day)),
        "subjects": subjects,
        "segments": segments,
        "marks": marks,
        "topics": (digest or {}).get("payload", {}).get("topics", []) if digest else [],
        "digested": digest is not None,
    }


def _thread_refs(app: App, ids: list[int]) -> list[dict[str, Any]]:
    out = []
    for tid in ids or []:
        thread = app.store.thread(tid)
        if thread:
            out.append({"id": thread["id"], "name": thread["name"], "hue": thread["hue"], "state": thread["state"]})
    return out


def spark_card(app: App, spark: dict[str, Any]) -> dict[str, Any]:
    return {**spark, "threads": _thread_refs(app, spark.get("thread_ids") or [])}


def dream_view(app: App, day: str) -> dict[str, Any] | None:
    dream = app.store.dream(day)
    if not dream:
        return None
    sparks = [spark_card(app, s) for s in app.store.sparks(dream_id=dream["id"])]
    sparks.sort(key=lambda s: -s["score"])
    return {**dream, "number": app.store.dream_number(day), "sparks": sparks}


def dreams_list(app: App) -> list[dict[str, Any]]:
    return [
        {k: d[k] for k in ("id", "day", "title", "undercurrent", "spark_count", "created_at")}
        for d in app.store.dreams(limit=365)
    ]


def sparks_list(app: App, status: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
    return [spark_card(app, s) for s in app.store.sparks(status=status or None, query=query or None, limit=300)]


def spark_view(app: App, spark_id: int) -> dict[str, Any] | None:
    spark = app.store.spark(spark_id)
    if not spark:
        return None
    dream = app.store.dream(spark["day"])
    return {
        **spark_card(app, spark),
        "dream": {"day": dream["day"], "title": dream["title"]} if dream else None,
        "dives": app.store.dives(spark_id),
    }


def threads_view(app: App, day: str | None = None) -> dict[str, Any]:
    day = day or today()
    signals, stats = compute_signals(app, day, app.settings.language)
    rows = [s.to_dict() for s in stats if s.thread["state"] != "merged"]
    rows = [r for r in rows if r["total"] > 0 or r["state"] == "pinned" or r["active_28"]]
    rows.sort(key=lambda r: (r["state"] != "pinned", r["state"] == "muted", -(r["active_14"] * 3600 + r["total"] / 10)))
    days = stats[0].days if stats else [
        (date.fromisoformat(day) - timedelta(days=o)).isoformat() for o in range(27, -1, -1)
    ]
    return {"day": day, "days": days, "threads": rows, "signals": [s.to_dict() for s in signals]}


def thread_view(app: App, thread_id: int) -> dict[str, Any] | None:
    thread = app.store.thread(thread_id)
    if not thread:
        return None
    day = today()
    stat = next((s for s in thread_stats(app, day, window=56) if s.id == thread_id), None)
    _signals, recent = compute_signals(app, day, app.settings.language)
    current = next((s for s in recent if s.id == thread_id), None)
    if stat is not None and current is not None:
        stat.signals = list(current.signals)  # patterns are judged on four weeks; the chart shows eight
    since = (date.fromisoformat(day) - timedelta(days=120)).isoformat()
    history = app.store.thread_days(since, thread_id)
    subjects: dict[str, int] = {}
    for row in history:
        for label in row.get("subjects") or []:
            subjects[label] = subjects.get(label, 0) + 1
    others = [
        {"id": t["id"], "name": t["name"]} for t in app.store.threads()
        if t["id"] != thread_id and t["state"] not in {"merged"}
    ]
    return {
        **(stat.to_dict() if stat else thread),
        "days": stat.days if stat else [],
        "history": [{"day": r["day"], "seconds": round(r["seconds"]), "note": r.get("note", ""),
                     "subjects": r.get("subjects") or []} for r in reversed(history)][:40],
        "subjects": sorted(subjects.items(), key=lambda kv: -kv[1])[:24],
        "sparks": [spark_card(app, s) for s in app.store.sparks(thread_id=thread_id)],
        "merged_into": thread.get("merged_into"),
        "others": others,
    }


def settings_view(app: App) -> dict[str, Any]:
    from unconscious.llm.router import ROLES

    described = app.router.describe()
    return {
        "settings": app.settings.to_dict(),
        "models": described,
        "roles": list(ROLES),
        "sensor": sensor_status(app),
        "home": str(app.home),
        "taste": load_taste(app).to_dict(),
        "usage": app.store.llm_usage((date.today() - timedelta(days=30)).isoformat()),
        "last_error": app.store.last_llm_error(),
        "memory": {
            "traces": app.store.trace_count(),
            "days": len(app.store.days_with_traces(limit=10000)),
        },
    }
