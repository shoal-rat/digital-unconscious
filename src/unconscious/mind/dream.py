"""The nightly pass: digest the day, compute signals, dream, critique, keep the best.

Generation is deliberately wider than what is shown. The dream proposes
``dream.candidates`` sparks; a separate model scores them against a rubric;
code applies the weights, the person's taste and a diversity rule, then keeps
``dream.sparks``. Anything that cites evidence it was not given is discarded
before a human ever sees it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from unconscious.llm.base import LLMRequest
from unconscious.mind.digest import DayDigest, digest_day
from unconscious.mind.prompts import (
    CRITIQUE_SCHEMA,
    CRITIQUE_SYSTEM,
    CRITIQUE_TASK,
    DREAM_SCHEMA,
    DREAM_SYSTEM,
    DREAM_TASK,
    LANGUAGE_LINE,
    MECHANISMS,
    RESEARCH_DREAM,
)
from unconscious.mind.signals import Signal, ThreadStats, compute_signals
from unconscious.mind.taste import Taste, load_taste
from unconscious.store import now_iso
from unconscious.text import clip, duration, similar

if TYPE_CHECKING:
    from unconscious.app import App

WEIGHTS = {"grounded": 0.25, "sharp": 0.20, "fresh": 0.25, "doable": 0.15, "fit": 0.15}
MIN_SCORE = 40.0
STEPS = ("gather", "digest", "signals", "dream", "critique", "save")


class DreamError(RuntimeError):
    pass


@dataclass
class Context:
    day: str
    digest: DayDigest
    signals: list[Signal]
    stats: list[ThreadStats]
    taste: Taste
    threads: dict[int, dict[str, Any]]
    valid_s: dict[str, dict[str, Any]]
    valid_t: dict[str, int]


def person_block(app: App) -> str:
    you = app.settings.you
    lines = []
    if you.name:
        lines.append(f"Name: {you.name}")
    if you.persona:
        lines.append(f"About them: {you.persona}")
    if you.focus:
        lines.append("Focus fields (sparks should land here): " + ", ".join(you.focus))
    if not lines:
        lines.append(
            "Not specified. Infer their field from the evidence. Prefer research-grade questions "
            "(something that could be studied, measured or tested) over product pitches."
        )
    return "\n".join(lines)


def build_context(app: App, day: str, digest: DayDigest, language: str) -> Context:
    signals, stats = compute_signals(app, day, language)
    threads = {t["id"]: t for t in app.store.threads()}
    key_to_ref = {s["subject_key"]: s["ref"] for s in digest.subjects}
    valid_s = {s["ref"]: s for s in digest.subjects if s["subject_key"] in key_to_ref}
    mentioned = {t["thread_id"] for t in digest.topics if t.get("thread_id")}
    for signal in signals:
        mentioned.update(signal.threads)
    valid_t = {
        f"T{tid}": tid for tid in mentioned
        if tid in threads and threads[tid]["state"] not in {"muted", "merged"}
    }
    return Context(day, digest, signals, stats, load_taste(app), threads, valid_s, valid_t)


def today_block(ctx: Context) -> str:
    key_to_subject = {s["subject_key"]: s for s in ctx.digest.subjects}
    lines = []
    topics = [t for t in ctx.digest.topics if not t.get("thread_id") or f"T{t['thread_id']}" in ctx.valid_t]
    topics.sort(key=lambda t: (not any(key_to_subject.get(k, {}).get("kind") in {"jot", "search", "reading"}
                                       for k in t["subject_keys"]), -t["seconds"]))
    for topic in topics[:12]:
        thread_id = topic.get("thread_id")
        thread = ctx.threads.get(thread_id) if thread_id else None
        head = f"• {topic['label']}"
        if topic.get("gist"):
            head += f" — {topic['gist']}"
        tag = f"T{thread_id} “{thread['name']}”" if thread else "no thread"
        head += f" [{tag}]"
        if topic.get("seconds"):
            head += f" ({duration(topic['seconds'])})"
        lines.append(head)
        for key in topic["subject_keys"][:6]:
            subject = key_to_subject.get(key)
            if not subject:
                continue
            detail = subject["kind"]
            if subject.get("seconds"):
                detail += f", {duration(subject['seconds'])}"
            text = subject.get("body") if subject["kind"] in {"jot", "note"} else subject["subject"]
            lines.append(f'    {subject["ref"]} ({detail}) “{clip(text or subject["subject"], 160)}”')
    return "\n".join(lines) or "(little was observed today)"


def undercurrent_block(ctx: Context) -> str:
    lines = []
    for signal in ctx.signals:
        refs = " × ".join(f"T{tid}" for tid in signal.threads)
        names = " × ".join(f"“{ctx.threads[tid]['name']}”" for tid in signal.threads if tid in ctx.threads)
        lines.append(f"{signal.kind.upper()} {refs} {names} — {signal.text}")
    return "\n".join(lines) or "(no long-term patterns yet: there is not enough history)"


def shown_block(app: App, day: str) -> str:
    """Fish of the last three weeks, the same day's earlier dives first: a second dive of a day
    must not bring back what the first one caught, kept or threw back."""
    since = (date.fromisoformat(day) - timedelta(days=21)).isoformat()
    same_day = app.store.sparks(day=day, limit=60)
    recent = [s for s in app.store.sparks(limit=60) if since <= s["day"] < day or s["day"] > day]
    return "\n".join(f"- {s['title']}" for s in (same_day + recent)[:24]) or "(nothing yet)"


def _refs(values: Any, valid: dict[str, Any]) -> list[str]:
    out = []
    for value in values or []:
        ref = str(value).strip().upper()
        if re.fullmatch(r"[ST]\d+", ref) and ref in valid and ref not in out:
            out.append(ref)
    return out


def validate_sparks(raw: list[dict[str, Any]], ctx: Context, shown: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for item in raw or []:
        title = clip(str(item.get("title") or "").strip(), 120)
        question = str(item.get("question") or "").strip()
        mechanism = str(item.get("mechanism") or "").strip().lower()
        evidence = _refs(item.get("evidence"), ctx.valid_s)
        threads = _refs(item.get("threads"), ctx.valid_t)
        reason = ""
        if not title or not question:
            reason = "missing title or question"
        elif mechanism not in MECHANISMS:
            reason = f"unknown mechanism {mechanism!r}"
        elif not evidence and not threads:
            reason = "cites no valid evidence"
        elif any(similar(title, old) >= 0.55 for old in shown):
            reason = "repeats a recent spark"
        elif any(similar(title, other["title"]) >= 0.55 for other in kept):
            reason = "duplicate of another candidate"
        if reason:
            rejected.append({"title": title or "(untitled)", "reason": reason})
            continue
        kept.append({
            "title": title,
            "mechanism": mechanism,
            "evidence_refs": evidence,
            "thread_refs": threads,
            "question": clip(question, 400),
            "insight": clip(str(item.get("insight") or ""), 900),
            "first_step": clip(str(item.get("first_step") or ""), 500),
            "kill": clip(str(item.get("kill") or ""), 400),
            "field": clip(str(item.get("field") or ""), 80),
            "search_terms": [clip(str(t), 120) for t in (item.get("search_terms") or []) if str(t).strip()][:4],
        })
    return kept, rejected


def critique_cards(app: App, ctx: Context, cards: list[dict[str, Any]], language: str) -> tuple[list[dict[str, Any]] | None, str]:
    lines, cited = [], []
    for index, card in enumerate(cards, 1):
        thread_names = ", ".join(
            f"{ref} “{ctx.threads[ctx.valid_t[ref]]['name']}”" for ref in card["thread_refs"]
        ) or "none"
        evidence = ", ".join(f"{ref} “{clip(ctx.valid_s[ref]['subject'], 70)}”" for ref in card["evidence_refs"]) or "none"
        cited.extend(card["evidence_refs"])
        lines.append(
            f"CARD {index} · mechanism: {card['mechanism']} · threads: {thread_names}\n"
            f"Title: {card['title']}\nQuestion: {card['question']}\nInsight: {card['insight']}\n"
            f"First step: {card['first_step']}\nKill condition: {card['kill']}\nEvidence: {evidence}"
        )
    evidence_lines = []
    for ref in dict.fromkeys(cited):
        s = ctx.valid_s[ref]
        text = s.get("body") if s["kind"] in {"jot", "note"} else s["subject"]
        evidence_lines.append(f'{ref} · {s["kind"]}{", " + duration(s["seconds"]) if s.get("seconds") else ""} · “{clip(text or s["subject"], 200)}”')
    request = LLMRequest(
        role="critique",
        system=CRITIQUE_SYSTEM,
        prompt=CRITIQUE_TASK.format(
            person=person_block(app),
            evidence="\n".join(evidence_lines) or "(none)",
            cards="\n\n".join(lines),
            language=LANGUAGE_LINE[language],
        ),
        schema=CRITIQUE_SCHEMA,
        max_tokens=3000,
        payload={"cards": cards},
    )
    result = app.router.call(request)
    if not result.ok or not result.data:
        return None, ""
    return list(result.data.get("reviews") or []), result.label


def score_cards(cards: list[dict[str, Any]], reviews: list[dict[str, Any]] | None, ctx: Context) -> None:
    by_card: dict[int, dict[str, Any]] = {}
    for review in reviews or []:
        try:
            by_card[int(review.get("card"))] = review
        except (TypeError, ValueError):
            continue
    signal_kinds: dict[int, set[str]] = {}
    for signal in ctx.signals:
        for tid in signal.threads:
            signal_kinds.setdefault(tid, set()).add(signal.kind)
    for index, card in enumerate(cards, 1):
        review = by_card.get(index)
        if review:
            dims = {k: max(1, min(5, int(review.get(k) or 3))) for k in WEIGHTS}
            base = (sum(dims[k] * w for k, w in WEIGHTS.items()) - 1) / 4 * 100
            if review.get("generic"):
                base *= 0.6
            card["scores"] = {**dims, "generic": bool(review.get("generic"))}
            card["objection"] = clip(str(review.get("objection") or ""), 300)
        else:
            base = 60.0
            card["scores"] = {}
            card["objection"] = ""
        thread_ids = [ctx.valid_t[r] for r in card["thread_refs"]]
        if any(card["mechanism"] in signal_kinds.get(tid, set()) for tid in thread_ids):
            base *= 1.05  # grown from a pattern code actually measured
        card["score"] = round(min(100.0, base * ctx.taste.multiplier(card["mechanism"])), 1)


def select(cards: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(cards, key=lambda c: -c["score"])
    chosen: list[dict[str, Any]] = []
    for card in ranked:
        if len(chosen) >= limit:
            break
        if card["score"] < MIN_SCORE and chosen:
            continue
        signature = (card["mechanism"], tuple(sorted(card["thread_refs"])))
        if card["thread_refs"] and any(
            (c["mechanism"], tuple(sorted(c["thread_refs"]))) == signature for c in chosen
        ):
            continue
        chosen.append(card)
    return chosen


def run_dream(app: App, day: str, progress: Callable[[str], None] | None = None, *, force_digest: bool = False) -> dict[str, Any]:
    step = progress or (lambda _name: None)
    settings = app.settings
    language = settings.language

    step("gather")
    if app.store.trace_count(day) == 0:
        raise DreamError(f"Nothing was observed on {day}. Leave `dun up` running, jot a thought, or feed a document.")
    # what this dive reads, on the scale the night watch measures growth by (see jobs.grown_since_dive)
    read_at, read_seconds = now_iso(), app.store.day_seconds(day)

    step("digest")
    digest = digest_day(app, day, force=force_digest)
    if not digest.topics:
        raise DreamError("The day was too thin to dream about: no topics survived the digest.")

    step("signals")
    ctx = build_context(app, day, digest, language)

    step("dream")
    shown = [line[2:] for line in shown_block(app, day).splitlines() if line.startswith("- ")]
    candidates_wanted = settings.dream.candidates
    total = sum(s.get("seconds") or 0 for s in digest.subjects)
    request = LLMRequest(
        role="dream",
        system=DREAM_SYSTEM.replace("{candidates}", str(candidates_wanted)) + (RESEARCH_DREAM if settings.models.research else ""),
        prompt=DREAM_TASK.format(
            person=person_block(app),
            day=day,
            total=duration(total),
            today=today_block(ctx),
            undercurrents=undercurrent_block(ctx),
            taste=ctx.taste.to_prompt(),
            shown=shown_block(app, day),
            candidates=candidates_wanted,
            language=LANGUAGE_LINE[language],
        ),
        schema=DREAM_SCHEMA,
        max_tokens=6000,
        payload={
            "day": day,
            "topics": digest.topics,
            "subjects": [{k: s.get(k) for k in ("ref", "subject", "kind", "seconds", "body", "subject_key")} for s in digest.subjects],
            "threads": {ref: ctx.threads[tid]["name"] for ref, tid in ctx.valid_t.items()},
            "signals": [s.to_dict() for s in ctx.signals],
            "candidates": candidates_wanted,
            "language": language,
            "focus": settings.you.focus,
        },
        research=settings.models.research,
    )
    result = app.router.call(request)
    if not result.ok or not result.data:
        raise DreamError(result.error or "The dream model returned nothing.")
    data = result.data
    cards, rejected = validate_sparks(data.get("sparks") or [], ctx, shown)
    if not cards:
        raise DreamError("Every candidate spark was rejected (" + "; ".join(r["reason"] for r in rejected[:3]) + ").")

    critic_label = ""
    reviews = None
    if settings.dream.critique:
        step("critique")
        reviews, critic_label = critique_cards(app, ctx, cards, language)
    score_cards(cards, reviews, ctx)
    chosen = select(cards, settings.dream.sparks)

    step("save")
    topics_out = []
    for topic in digest.topics:
        thread = ctx.threads.get(topic.get("thread_id") or -1)
        topics_out.append({**topic, "thread_name": thread["name"] if thread else None, "hue": thread["hue"] if thread else None})
    payload = {
        "topics": topics_out,
        "signals": [s.to_dict() for s in ctx.signals],
        "stats": {
            "seconds": round(total),
            "subjects": len(digest.subjects),
            "threads": len({t["thread_id"] for t in digest.topics if t.get("thread_id")}),
            "candidates": len(cards) + len(rejected),
            "kept": len(chosen),
            "day_seconds": round(read_seconds),
            "read_at": read_at,
        },
        "rejected": rejected + [{"title": c["title"], "reason": f"ranked out ({c['score']})"} for c in cards if c not in chosen],
    }
    fish = []
    for card in chosen:
        evidence = []
        for ref in card["evidence_refs"]:
            s = ctx.valid_s[ref]
            evidence.append({
                "ref": ref, "label": s["subject"], "kind": s["kind"], "seconds": round(s.get("seconds") or 0),
                "visits": s.get("visits") or 0, "url": s.get("url") or "", "domain": s.get("domain") or "",
                "body": clip(s.get("body") or "", 400), "day": day,
            })
        fish.append({
            "title": card["title"],
            "mechanism": card["mechanism"],
            "question": card["question"],
            "insight": card["insight"],
            "first_step": card["first_step"],
            "kill": card["kill"],
            "field": card["field"],
            "thread_ids": [ctx.valid_t[r] for r in card["thread_refs"]],
            "evidence": evidence,
            "search_terms": card["search_terms"],
            "scores": card["scores"],
            "score": card["score"],
            "objection": card["objection"],
        })
    # one transaction: the dive and its fish arrive together, and an earlier dive of the day stays whole
    dream_id = app.store.save_dream(
        day,
        title=clip(str(data.get("title") or ""), 140),
        reflection=clip(str(data.get("reflection") or ""), 1400),
        undercurrent=clip(str(data.get("undercurrent") or ""), 400),
        payload=payload,
        models={"digest": digest.model, "dream": result.label, "critique": critic_label},
        sparks=fish,
    )
    spark_ids = [s["id"] for s in sorted(app.store.sparks(dream_id=dream_id), key=lambda s: s["id"])]
    app.store.log_event("dream", day, {"sparks": spark_ids, "models": {"dream": result.label, "critique": critic_label}})
    return {"day": day, "dream_id": dream_id, "title": data.get("title", ""), "sparks": spark_ids,
            "candidates": len(cards), "rejected": len(rejected)}
