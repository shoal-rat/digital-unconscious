"""Daily idea capture — extracts research ideas from daily logs using LLM.

LLM-only: Claude scores snippets and rewrites them as research questions.
When LLM is unavailable, raw snippets are saved for later processing.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from du_research.utils import iso_now, slugify


def _read_entries(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        entries = []
        for line in input_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
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


def _llm_score_and_rewrite(
    snippets: list[dict[str, str]],
    backend: object | None,
    max_ideas: int = 10,
) -> list[dict[str, Any]]:
    """Use LLM to score snippets and rewrite them as research ideas."""
    if backend is None or not snippets:
        # No LLM — return raw snippets with zero scores (will be queued)
        return [
            {
                "idea_id": f"idea_{slugify(s['text'], max_length=24)}",
                "idea_text": s["text"],
                "score": 0.0,
                "domain": "pending",
                "keywords": [],
                "source_excerpt": s["text"][:300],
                "status": "pending_llm",
            }
            for s in snippets[:max_ideas]
        ]

    # Batch snippets for LLM evaluation
    snippet_texts = [s["text"][:200] for s in snippets[:30]]
    prompt = (
        f"From these {len(snippet_texts)} text snippets from a user's daily activity, "
        f"extract the top {max_ideas} most promising research ideas.\n\n"
        f"Snippets:\n" + "\n".join(f"{i+1}. {t}" for i, t in enumerate(snippet_texts)) + "\n\n"
        f"For each idea:\n"
        f"1. Rewrite it as a clear research question or hypothesis\n"
        f"2. Score it 0-100 on research potential\n"
        f"3. Identify the domain\n"
        f"4. Extract 3-5 keywords\n\n"
        f"Return ONLY JSON array:\n"
        f'[{{"idea_text": "...", "score": 75, "domain": "...", "keywords": ["..."], "source_index": 0}}]'
    )

    try:
        response = backend.call(prompt, mode="strict", model="sonnet", max_tokens=2000)
        if response.ok:
            text = response.text
            try:
                ideas = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    ideas = json.loads(text[start:end])
                else:
                    ideas = []

            result = []
            for idea in ideas[:max_ideas]:
                src_idx = idea.get("source_index", 0)
                source = snippets[src_idx]["text"] if src_idx < len(snippets) else ""
                result.append({
                    "idea_id": f"idea_{slugify(idea.get('idea_text', ''), max_length=24)}",
                    "idea_text": idea.get("idea_text", ""),
                    "score": round(idea.get("score", 0) / 100, 4),
                    "domain": idea.get("domain", "general"),
                    "keywords": idea.get("keywords", []),
                    "source_excerpt": source[:300],
                })
            return result
    except Exception:
        pass

    # LLM failed — return raw
    return [
        {
            "idea_id": f"idea_{slugify(s['text'], max_length=24)}",
            "idea_text": s["text"],
            "score": 0.0,
            "domain": "pending",
            "keywords": [],
            "source_excerpt": s["text"][:300],
            "status": "pending_llm",
        }
        for s in snippets[:max_ideas]
    ]


def capture_daily_ideas(
    input_path: Path,
    output_dir: Path,
    max_ideas: int,
    min_idea_score: float,
    backlog_path: Path | None = None,
    backend: object | None = None,
) -> dict[str, Any]:
    entries = _read_entries(input_path)
    snippets = _candidate_snippets(entries)

    ideas = _llm_score_and_rewrite(snippets, backend, max_ideas=max_ideas)
    # Filter by minimum score (LLM scores are 0-1)
    ideas = [i for i in ideas if i.get("score", 0) >= min_idea_score or i.get("status") == "pending_llm"]

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
