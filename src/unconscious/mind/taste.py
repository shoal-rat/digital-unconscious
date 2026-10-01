"""Taste: what the person has kept and rejected, in a form both the prompt and
the ranking can use. Counted, not inferred, so it is transparent and cannot
drift into a story the person never told."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unconscious.app import App

POSITIVE = {"kept", "pursuing", "done"}
REASONS = {
    "generic": "too generic",
    "known": "already knew it",
    "off_field": "not my field",
    "infeasible": "not feasible",
    "wrong": "misread my day",
}


@dataclass
class Taste:
    kept: int = 0
    dismissed: int = 0
    mechanism_up: Counter = field(default_factory=Counter)
    mechanism_down: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    loved_threads: list[str] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)
    kept_titles: list[str] = field(default_factory=list)
    dismissed_titles: list[str] = field(default_factory=list)

    def multiplier(self, mechanism: str) -> float:
        """Nudge ranking toward mechanisms the person responds to: at most ±15%."""
        up, down = self.mechanism_up[mechanism], self.mechanism_down[mechanism]
        total = up + down
        if total < 2:
            return 1.0
        return 1.0 + 0.15 * (up - down) / total

    def to_prompt(self) -> str:
        if not (self.kept or self.dismissed or self.pinned):
            return "No feedback yet."
        lines = []
        if self.kept:
            mechanisms = ", ".join(f"{m} ×{n}" for m, n in self.mechanism_up.most_common(4))
            lines.append(f"Kept or pursued {self.kept} sparks ({mechanisms}). Examples: " + "; ".join(self.kept_titles[:4]))
        if self.dismissed:
            reasons = ", ".join(f"{REASONS.get(r, r)} ×{n}" for r, n in self.reasons.most_common(4)) or "no reason given"
            lines.append(f"Dismissed {self.dismissed} sparks ({reasons}). Examples: " + "; ".join(self.dismissed_titles[:3]))
        if self.pinned:
            lines.append("Threads they explicitly care about: " + ", ".join(self.pinned[:6]))
        if self.loved_threads:
            lines.append("Threads behind sparks they liked: " + ", ".join(self.loved_threads[:6]))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kept": self.kept, "dismissed": self.dismissed,
            "mechanisms": {m: {"up": self.mechanism_up[m], "down": self.mechanism_down[m]}
                           for m in set(self.mechanism_up) | set(self.mechanism_down)},
            "reasons": dict(self.reasons), "pinned": self.pinned, "loved_threads": self.loved_threads,
        }


def load_taste(app: App) -> Taste:
    taste = Taste()
    threads = {t["id"]: t for t in app.store.threads()}
    taste.pinned = [t["name"] for t in threads.values() if t["state"] == "pinned"]
    loved: Counter = Counter()
    for spark in app.store.sparks(limit=400):
        status = spark["status"]
        if status in POSITIVE:
            taste.kept += 1
            taste.mechanism_up[spark["mechanism"]] += 1
            taste.kept_titles.append(spark["title"])
            for tid in spark.get("thread_ids") or []:
                if tid in threads:
                    loved[threads[tid]["name"]] += 1
        elif status == "dismissed":
            taste.dismissed += 1
            taste.mechanism_down[spark["mechanism"]] += 1
            taste.dismissed_titles.append(spark["title"])
            if spark.get("reason"):
                taste.reasons[spark["reason"]] += 1
    taste.loved_threads = [name for name, _ in loved.most_common(6)]
    return taste
