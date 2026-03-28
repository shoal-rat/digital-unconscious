from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from html import unescape
from statistics import mean
from typing import Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_for_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def slugify(value: str, max_length: int = 48) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized[:max_length] or "idea"


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 2]


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(without_tags)).strip()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0


def overlap_score(query: str, candidate: str) -> float:
    query_tokens = set(tokenize(query))
    candidate_tokens = set(tokenize(candidate))
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    return overlap / len(query_tokens)


def top_keywords(texts: Iterable[str], limit: int = 10) -> list[dict[str, float | str]]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    total = sum(counter.values()) or 1
    return [
        {"keyword": keyword, "weight": round(count / total, 4)}
        for keyword, count in counter.most_common(limit)
    ]


def infer_domain(texts: Iterable[str]) -> str:
    domain_rules = {
        "ai": {"ai", "llm", "model", "machine", "learning", "automation"},
        "product": {"pricing", "saas", "product", "churn", "growth", "ux"},
        "health": {"clinical", "health", "medical", "patient", "disease"},
        "education": {"student", "teaching", "education", "learning"},
        "economics": {"market", "economics", "consumer", "pricing", "demand"},
        "behavior": {"behavior", "cognitive", "attention", "motivation"},
    }
    scores = Counter()
    tokens = set(tokenize(" ".join(texts)))
    for domain, keywords in domain_rules.items():
        scores[domain] = len(tokens & keywords)
    best_domain, best_score = "general", 0
    for domain, score in scores.items():
        if score > best_score:
            best_domain, best_score = domain, score
    return best_domain


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def safe_year(value: object) -> int | None:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return int(match.group(0))
    return None


def recentness_score(year: int | None, current_year: int | None = None) -> float:
    if year is None:
        return 0.2
    current_year = current_year or datetime.now(timezone.utc).year
    age = max(0, current_year - year)
    return clamp(1.0 - (age / 15.0), 0.1, 1.0)


def citation_score(count: int | None) -> float:
    if not count:
        return 0.0
    return clamp(math.log10(count + 1) / 3.0, 0.0, 1.0)

