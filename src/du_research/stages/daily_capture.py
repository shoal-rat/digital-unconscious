from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from du_research.utils import infer_domain, iso_now, overlap_score, slugify, tokenize, top_keywords


IDEA_HINTS = {
    "idea",
    "question",
    "hypothesis",
    "experiment",
    "study",
    "test",
    "measure",
    "pricing",
    "behavior",
    "churn",
    "interface",
    "could",
    "should",
    "why",
    "how",
}


def _read_entries(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        entries = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            entries.append(payload)
        return entries
    return [{"text": input_path.read_text(encoding="utf-8")}]


def _candidate_snippets(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    snippets = []
    for entry in entries:
        text = " ".join(
            str(entry.get(key, ""))
            for key in ["timestamp", "title", "url", "text", "content", "note"]
            if entry.get(key)
        ).strip()
        if not text:
            continue
        chunks = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
        for chunk in chunks:
            cleaned = re.sub(r"\s+", " ", chunk).strip(" -")
            if len(cleaned) < 25:
                continue
            snippets.append({"text": cleaned})
    return snippets


def _score_snippet(snippet: str) -> float:
    tokens = tokenize(snippet)
    if not tokens:
        return 0.0
    unique_ratio = min(len(set(tokens)) / max(len(tokens), 1), 1.0)
    idea_signal = len(set(tokens) & IDEA_HINTS) / 6.0
    question_signal = 0.15 if "?" in snippet else 0.0
    bridge_signal = 0.15 if any(word in tokens for word in ["as", "vs", "between", "across"]) else 0.0
    return round(min((unique_ratio * 0.4) + idea_signal + question_signal + bridge_signal, 1.0), 4)


def _rewrite_as_idea(snippet: str) -> str:
    text = snippet.strip()
    text = re.sub(r"^\d+[\).\s-]+", "", text)
    if text.endswith("?"):
        return text
    if not re.search(r"\b(how|why|what|could|should|test|measure|study)\b", text.lower()):
        return f"Research whether {text[0].lower() + text[1:]}" if text else text
    return text


def capture_daily_ideas(
    input_path: Path,
    output_dir: Path,
    max_ideas: int,
    min_idea_score: float,
    backlog_path: Path | None = None,
) -> dict[str, Any]:
    entries = _read_entries(input_path)
    snippets = _candidate_snippets(entries)
    ranked = []
    for snippet in snippets:
        score = _score_snippet(snippet["text"])
        if score < min_idea_score:
            continue
        idea_text = _rewrite_as_idea(snippet["text"])
        ranked.append(
            {
                "idea_id": f"idea_{slugify(idea_text, max_length=24)}",
                "idea_text": idea_text,
                "score": score,
                "domain": infer_domain([idea_text]),
                "keywords": top_keywords([idea_text], limit=5),
                "source_excerpt": snippet["text"][:300],
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for item in sorted(ranked, key=lambda value: value["score"], reverse=True):
        key = item["idea_text"].lower()
        existing = deduped.get(key)
        if existing is None or item["score"] > existing["score"]:
            deduped[key] = item
    ideas = list(deduped.values())[:max_ideas]

    domain_counts = Counter(item["domain"] for item in ideas)
    payload = {
        "created_at": iso_now(),
        "input_path": str(input_path),
        "entry_count": len(entries),
        "candidate_count": len(snippets),
        "idea_count": len(ideas),
        "top_domains": [{"domain": key, "count": value} for key, value in domain_counts.most_common(5)],
        "ideas": ideas,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ideas.json"
    md_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Daily Idea Capture",
        "",
        f"- Input: `{input_path}`",
        f"- Entries parsed: {len(entries)}",
        f"- Candidate snippets: {len(snippets)}",
        f"- Ideas captured: {len(ideas)}",
        "",
        "## Top Ideas",
        "",
    ]
    for index, idea in enumerate(ideas, start=1):
        lines.append(f"{index}. **{idea['idea_text']}** (score {idea['score']:.2f}, domain {idea['domain']})")
        lines.append(f"   - Source excerpt: {idea['source_excerpt']}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if backlog_path is not None and ideas:
        backlog_path.parent.mkdir(parents=True, exist_ok=True)
        with backlog_path.open("a", encoding="utf-8") as handle:
            for idea in ideas:
                handle.write(json.dumps(idea, ensure_ascii=False) + "\n")
    return payload
