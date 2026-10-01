"""Small text helpers: tokens for overlap tests, durations for prompts."""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = set(
    """a an and are as at be by for from has have how in into is it its of on or that the this to
    was were what when where which who why will with you your vs via using use new about after
    before over under than then them they their there these those not no can could should would
    may might do does did done just also more most very much many some any each other""".split()
)

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.-]*[a-z0-9+#]|[a-z0-9]")
_CJK = re.compile(r"[㐀-鿿豈-﫿]+")


def tokens(text: str) -> set[str]:
    """Lowercase word tokens plus CJK character bigrams, minus stopwords."""
    text = unicodedata.normalize("NFKC", text or "").casefold()
    out = {w for w in _WORD.findall(text) if w not in STOPWORDS and len(w) > 1}
    for run in _CJK.findall(text):
        if len(run) == 1:
            out.add(run)
        out.update(run[i : i + 2] for i in range(len(run) - 1))
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similar(a: str, b: str) -> float:
    return jaccard(tokens(a), tokens(b))


def duration(seconds: float) -> str:
    seconds = int(round(seconds or 0))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"


def clock(iso: str) -> str:
    return iso[11:16] if iso and len(iso) >= 16 else ""


def clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"[^\w]+", " ", text).strip()[:160]
