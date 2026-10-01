"""Signals: patterns in the threads over time, computed by plain arithmetic.

This is where the "unconscious" becomes concrete and explainable. A model can
interpret an orbit; it should not be trusted to count one.

  orbit      frequent but shallow: back again and again, never for long
  return     resurfaced after a long absence
  surge      far more time today than its own baseline
  seed       first appearance today, with real time or intent behind it
  steady     the person's known main work (an anchor, not an undercurrent)
  fade       a thread that used to recur and has gone quiet
  collision  two distant threads active close together today
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from unconscious.text import tokens

if TYPE_CHECKING:
    from unconscious.app import App

ORBIT_MAX_MEDIAN = 600
ORBIT_MAX_PEAK = 1500
COLLISION_DISTANCE = 0.85
COLLISION_WINDOW_MIN = 90


@dataclass
class ThreadStats:
    thread: dict[str, Any]
    days: list[str]
    series: list[float]
    today: float = 0.0
    active_today: bool = False
    active_14: int = 0
    active_28: int = 0
    median: float = 0.0
    mean_prior: float = 0.0
    peak: float = 0.0
    gap: int | None = None
    prior_active: int = 0
    last_active: str | None = None
    total: float = 0.0
    signals: list[str] = field(default_factory=list)

    @property
    def id(self) -> int:
        return int(self.thread["id"])

    @property
    def name(self) -> str:
        return str(self.thread["name"])

    def to_dict(self) -> dict[str, Any]:
        t = self.thread
        return {
            "id": t["id"], "name": t["name"], "gist": t.get("gist", ""), "keywords": t.get("keywords") or [],
            "hue": t.get("hue", 0), "state": t.get("state", "active"), "first_day": t.get("first_day"),
            "last_day": t.get("last_day"), "series": [round(x) for x in self.series], "today": round(self.today),
            "active_14": self.active_14, "active_28": self.active_28, "median": round(self.median),
            "peak": round(self.peak), "gap": self.gap, "total": round(self.total), "signals": self.signals,
        }


@dataclass
class Signal:
    kind: str
    threads: list[int]
    strength: float
    facts: dict[str, Any]
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "threads": self.threads, "strength": round(self.strength, 3),
                "facts": self.facts, "text": self.text}


def _active(row: dict[str, Any]) -> bool:
    seconds = row.get("seconds") or 0
    # seconds == 0 marks pure intent (a jot or a fed document): conscious, so it counts.
    return seconds >= 45 or (row.get("visits") or 0) >= 2 or seconds == 0


def thread_stats(app: App, day: str, window: int = 28) -> list[ThreadStats]:
    end = date.fromisoformat(day)
    days = [(end - timedelta(days=offset)).isoformat() for offset in range(window - 1, -1, -1)]
    lookback = (end - timedelta(days=window + 60)).isoformat()
    rows = app.store.thread_days(lookback)
    by_thread: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row["day"] <= day:
            by_thread.setdefault(row["thread_id"], {})[row["day"]] = row
    stats: list[ThreadStats] = []
    for thread in app.store.threads():
        if thread["state"] == "merged":
            continue
        history = by_thread.get(thread["id"], {})
        series = [float((history.get(d) or {}).get("seconds") or 0) for d in days]
        active_days = sorted(d for d, r in history.items() if _active(r))
        prior = [d for d in active_days if d < day]
        prior_window = [d for d in prior if d >= days[0]]
        st = ThreadStats(thread=thread, days=days, series=series)
        today_row = history.get(day)
        st.today = float((today_row or {}).get("seconds") or 0)
        st.active_today = bool(today_row and _active(today_row))
        cutoff14 = (end - timedelta(days=13)).isoformat()
        st.active_14 = sum(1 for d in active_days if d >= cutoff14)
        st.active_28 = sum(1 for d in active_days if d >= days[0])
        durations = [float(history[d].get("seconds") or 0) for d in active_days if d >= days[0]]
        st.median = statistics.median(durations) if durations else 0.0
        st.peak = max(durations) if durations else 0.0
        prior_durations = [float(history[d].get("seconds") or 0) for d in prior_window]
        st.mean_prior = statistics.mean(prior_durations) if prior_durations else 0.0
        st.total = sum(series)
        st.last_active = active_days[-1] if active_days else None
        if prior:
            st.gap = (end - date.fromisoformat(prior[-1])).days
            # count active days in the run before the gap (up to 60 days back)
            st.prior_active = len(prior)
        stats.append(st)
    return stats


def _thread_tokens(thread: dict[str, Any]) -> set[str]:
    return tokens(thread["name"] + " " + " ".join(thread.get("keywords") or []))


def _intervals(app: App, day: str) -> dict[int, list[tuple[datetime, datetime]]]:
    mapping = app.store.subject_threads(day)
    out: dict[int, list[tuple[datetime, datetime]]] = {}
    for trace in app.store.traces(day):
        thread_id = mapping.get(trace["subject_key"])
        if thread_id is None:
            continue
        try:
            start = datetime.fromisoformat(trace["started_at"])
            stop = datetime.fromisoformat(trace["ended_at"])
        except ValueError:
            continue
        out.setdefault(thread_id, []).append((start, stop))
    return out


def _closest_minutes(a: list[tuple[datetime, datetime]], b: list[tuple[datetime, datetime]]) -> float | None:
    best: float | None = None
    for s1, e1 in a:
        for s2, e2 in b:
            gap = max(0.0, (max(s1, s2) - min(e1, e2)).total_seconds() / 60)
            best = gap if best is None else min(best, gap)
    return best


def compute_signals(app: App, day: str, language: str = "en") -> tuple[list[Signal], list[ThreadStats]]:
    stats = thread_stats(app, day)
    live = [s for s in stats if s.thread["state"] != "muted"]
    end = date.fromisoformat(day)
    signals: list[Signal] = []

    for st in live:
        recent = st.last_active is not None and (end - date.fromisoformat(st.last_active)).days <= 2
        facts = {"name": st.name, "active": st.active_14, "today": st.today, "median": st.median,
                 "peak": st.peak, "gap": st.gap}
        if recent and st.active_14 >= 4 and st.median <= ORBIT_MAX_MEDIAN and st.peak <= ORBIT_MAX_PEAK:
            strength = min(1.0, st.active_14 / 10) * (1 - st.median / 900)
            signals.append(Signal("orbit", [st.id], strength, facts))
            st.signals.append("orbit")
        if st.active_today and st.gap is not None and st.gap >= 5 and st.prior_active >= 2:
            strength = min(1.0, st.gap / 21) * min(1.0, st.prior_active / 4)
            signals.append(Signal("return", [st.id], strength, facts))
            st.signals.append("return")
        prior_days = sum(1 for x in st.series[:-1] if x > 0)
        if st.today >= 1200 and prior_days >= 3 and st.mean_prior > 0 and st.today >= 3 * st.mean_prior:
            ratio = st.today / st.mean_prior
            signals.append(Signal("surge", [st.id], min(1.0, ratio / 6), {**facts, "ratio": round(ratio, 1)}))
            st.signals.append("surge")
        if st.thread.get("first_day") == day and st.active_today and (st.today >= 600 or st.today == 0):
            signals.append(Signal("seed", [st.id], min(1.0, max(st.today, 900) / 2400), facts))
            st.signals.append("seed")
        if st.active_14 >= 7 and st.median >= 1200:
            signals.append(Signal("steady", [st.id], 0.3, facts))
            st.signals.append("steady")
        if (
            not st.active_today and st.last_active
            and 7 <= (end - date.fromisoformat(st.last_active)).days <= 21
            and st.prior_active >= 4
        ):
            gap = (end - date.fromisoformat(st.last_active)).days
            signals.append(Signal("fade", [st.id], min(1.0, st.prior_active / 10), {**facts, "gap": gap, "active": st.prior_active}))
            st.signals.append("fade")

    # Collisions: distant threads that met today.
    today_threads = [s for s in live if s.active_today and (s.today >= 180 or s.today == 0)]
    intervals = _intervals(app, day) if len(today_threads) > 1 else {}
    pairs: list[Signal] = []
    for i, a in enumerate(today_threads):
        for b in today_threads[i + 1 :]:
            if "steady" in a.signals and "steady" in b.signals:
                continue
            ta, tb = _thread_tokens(a.thread), _thread_tokens(b.thread)
            union = ta | tb
            distance = 1 - (len(ta & tb) / len(union) if union else 0)
            if distance < COLLISION_DISTANCE:
                continue
            minutes = _closest_minutes(intervals.get(a.id, []), intervals.get(b.id, []))
            adjacent = minutes is not None and minutes <= COLLISION_WINDOW_MIN
            # Two hours of main work next to six minutes of something else is not a meeting of
            # equals; prefer balanced pairs, pairs that met in time, and pairs involving an
            # undercurrent rather than the person's known work.
            small, large = sorted((max(a.today, 60.0), max(b.today, 60.0)))
            balance = (small / large) ** 0.3
            weight = min(1.0, (a.today + b.today) / 1800) ** 0.5 if (a.today + b.today) else 0.6
            proximity = 1.0 if adjacent else (0.7 if minutes is None else 0.55)
            flags = set(a.signals) | set(b.signals)
            if flags & {"orbit", "return", "seed", "surge"}:
                novelty = 1.0
            elif "steady" in flags:
                novelty = 0.6
            else:
                novelty = 0.75
            strength = min(1.0, distance * balance * weight * proximity * novelty)
            pairs.append(Signal("collision", [a.id, b.id], strength, {
                "a": a.name, "b": b.name, "minutes": None if minutes is None else round(minutes),
                "adjacent": adjacent,
            }))
    pairs.sort(key=lambda s: -s.strength)
    for pair in pairs[:3]:
        signals.append(pair)
        for st in live:
            if st.id in pair.threads and "collision" not in st.signals:
                st.signals.append("collision")

    for signal in signals:
        signal.text = describe(signal, language)
    order = {"collision": 0, "orbit": 1, "return": 2, "surge": 3, "seed": 4, "fade": 5, "steady": 6}
    signals.sort(key=lambda s: (order.get(s.kind, 9), -s.strength))
    return signals, stats


def human_duration(seconds: float, language: str) -> str:
    minutes = int(round((seconds or 0) / 60))
    if language == "zh":
        if minutes < 60:
            return f"{max(minutes, 1)} 分钟"
        hours, rest = divmod(minutes, 60)
        return f"{hours} 小时" + (f" {rest} 分" if rest else "")
    if minutes < 60:
        return f"{max(minutes, 1)} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h" + (f" {rest} min" if rest else "")


TEMPLATES = {
    "en": {
        "orbit": "An eddy: you came back to “{name}” on {active} of the last 14 days, never staying longer than {peak}.",
        "return": "A return tide: “{name}” drifted back today after {gap} days away.",
        "surge": "A swell: “{name}” carried {today} today, about {ratio}× its usual.",
        "seed": "Driftwood: “{name}” washed in for the first time today.",
        "steady": "A main current: “{name}” runs on {active} of the last 14 days, usually {median}.",
        "fade": "An ebb: “{name}” has gone quiet, last seen {gap} days ago after {active} active days.",
        "collision": "A confluence: “{a}” and “{b}” both ran today",
        "collision_near": "A confluence: “{a}” and “{b}” ran today, {minutes} minutes apart at the closest.",
        "collision_same": "A confluence: “{a}” and “{b}” ran at the same time today.",
    },
    "zh": {
        "orbit": "涡流：过去 14 天里你有 {active} 天回到「{name}」，但每次都没超过 {peak}。",
        "return": "回潮：「{name}」离开 {gap} 天后，今天又漂了回来。",
        "surge": "涌浪：「{name}」今天带来了 {today}，约为平时的 {ratio} 倍。",
        "seed": "漂流木：「{name}」今天第一次被冲上岸。",
        "steady": "主流：「{name}」过去 14 天有 {active} 天在流动，通常每天 {median}。",
        "fade": "退潮：「{name}」渐渐平息，上次出现在 {gap} 天前，此前活跃过 {active} 天。",
        "collision": "交汇：「{a}」和「{b}」今天都在流动",
        "collision_near": "交汇：「{a}」和「{b}」今天都在流动，最近时只相隔 {minutes} 分钟。",
        "collision_same": "交汇：「{a}」和「{b}」今天几乎同时流过。",
    },
}


def describe(signal: Signal, language: str = "en") -> str:
    t = TEMPLATES.get(language, TEMPLATES["en"])
    f = signal.facts
    if signal.kind == "collision":
        minutes = f.get("minutes")
        if minutes is None:
            key = "collision"
        elif minutes == 0:
            key = "collision_same"
        else:
            key = "collision_near"
        text = t[key].format(a=f["a"], b=f["b"], minutes=minutes)
        return text if key != "collision" else text + ("." if language == "en" else "。")
    return t[signal.kind].format(
        name=f.get("name", ""),
        active=f.get("active", 0),
        peak=human_duration(f.get("peak", 0), language),
        median=human_duration(f.get("median", 0), language),
        today=human_duration(f.get("today", 0), language),
        gap=f.get("gap", 0),
        ratio=f.get("ratio", ""),
    )
