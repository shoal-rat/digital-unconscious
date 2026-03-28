"""Daily Idea Briefing generator — assembles the final markdown output.

Takes the day's behaviour summaries, scored ideas, and optional research
results and produces a single reader-friendly markdown briefing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from du_research.ai_backend import AIBackend

logger = logging.getLogger(__name__)


BRIEFING_SYSTEM_PROMPT = """\
You are the Briefing Writer for Digital Unconscious.

You receive structured data about a user's day — their screen behaviour
summaries, AI-generated ideas, and evaluation scores — and you produce a
single engaging daily briefing in Markdown.

The briefing has these sections (in order):
1. **Today's Focus** — What the user actually spent deep time on (from
   behaviour summaries). 2-3 sentences.
2. **Hidden Problem** — A question or challenge the user seems to be
   circling without directly addressing. 1-2 sentences.
3. **Today's Ideas (TOP 3-5)** — Each idea with its score, source
   behaviour, and a one-line research hook.
4. **Overlooked Signals** — Low-dwell content that might be worth
   exploring deeper. 1-2 items.
5. **Work Persona Insight** — One observation about the user's current
   thinking pattern or domain focus. 1-2 sentences.

Style: conversational but substantive. No fluff. Direct second-person
("you spent time on…"). Use emoji sparingly (one per section header max).

Output ONLY the markdown text. No JSON wrapping.
"""


@dataclass
class BriefingAgent:
    """Generates the daily idea briefing."""

    backend: AIBackend
    model: str = "opus"
    system_prompt: str | None = None

    def generate(
        self,
        behaviour_summaries: list[dict[str, Any]],
        scored_ideas: list[dict[str, Any]],
        *,
        idea_evaluations: list[dict[str, Any]] | None = None,
        human_idea_model: dict[str, Any] | None = None,
        date_str: str | None = None,
    ) -> str:
        """Generate the daily markdown briefing."""
        date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Build structured input
        parts = [f"## Date: {date_str}"]

        parts.append("\n## Behaviour Summaries")
        for i, summary in enumerate(behaviour_summaries[:6], 1):
            parts.append(f"\n### Window {i}")
            parts.append(json.dumps(summary, indent=2, ensure_ascii=False))

        # Merge ideas with their evaluations
        eval_map = {}
        if idea_evaluations:
            for ev in idea_evaluations:
                eval_map[ev.get("idea_id", "")] = ev

        top_ideas = []
        for idea in scored_ideas:
            idea_id = idea.get("id", idea.get("idea_id", ""))
            ev = eval_map.get(idea_id, {})
            merged = {**idea, **ev}
            top_ideas.append(merged)

        # Sort by score, take top included/held ones
        top_ideas.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        included = [i for i in top_ideas if i.get("verdict") != "discard"][:10]

        parts.append(f"\n## Scored Ideas ({len(included)} above threshold)")
        parts.append(json.dumps(included, indent=2, ensure_ascii=False))

        if human_idea_model:
            parts.append("\n## User's Idea Model Summary")
            model_brief = {
                k: human_idea_model[k]
                for k in ["top_domains", "top_keywords", "average_quality_score"]
                if k in human_idea_model
            }
            parts.append(json.dumps(model_brief, indent=2, ensure_ascii=False))

        parts.append(
            "\nWrite the daily briefing as described. "
            "Output ONLY markdown, starting with a date header."
        )

        prompt = "\n".join(parts)
        response = self.backend.call(
            prompt,
            mode="balanced",
            system=self.system_prompt or BRIEFING_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=2000,
        )

        if response.ok:
            return response.text

        # Fallback: generate a simple briefing without AI
        return _heuristic_briefing(date_str, behaviour_summaries, included)


def _heuristic_briefing(
    date_str: str,
    summaries: list[dict[str, Any]],
    ideas: list[dict[str, Any]],
) -> str:
    """Generate a simple briefing without AI."""
    lines = [
        f"# Daily Idea Briefing — {date_str}",
        "",
        "## Today's Focus",
        "",
    ]

    # Extract dominant topics
    all_topics: list[str] = []
    for s in summaries:
        all_topics.extend(s.get("dominant_topics", []))
    if all_topics:
        lines.append(f"Your screen time was dominated by: {', '.join(all_topics[:6])}.")
    else:
        lines.append("No behaviour data captured for today.")

    lines.extend(["", "## Hidden Problem", ""])
    intent_signals = []
    for s in summaries:
        intent_signals.extend(s.get("intent_signals", []))
    if intent_signals:
        lines.append(
            f"You seem to be circling around: {', '.join(intent_signals[:3])}. "
            "Consider addressing this directly."
        )
    else:
        lines.append("No strong hidden signals detected today.")

    lines.extend(["", "## Today's Ideas", ""])
    if ideas:
        for i, idea in enumerate(ideas[:5], 1):
            title = idea.get("title", idea.get("idea_text", "Untitled"))
            score = idea.get("total_score", idea.get("score", 0))
            desc = idea.get("description", "")
            reason = idea.get("one_line_reason", "")
            lines.append(f"### #{i} · {score:.0f}/100 — {title}")
            if desc:
                lines.append(f"\n{desc}")
            if reason:
                lines.append(f"\n*{reason}*")
            lines.append("")
    else:
        lines.append("No ideas passed the quality threshold today.")

    lines.extend(["", "## Overlooked Signals", ""])
    cross_hints = []
    for s in summaries:
        cross_hints.extend(s.get("cross_domain_hints", []))
    if cross_hints:
        for hint in cross_hints[:3]:
            lines.append(f"- {hint}")
    else:
        lines.append("- No cross-domain signals detected.")

    lines.extend([
        "",
        "## Work Persona Insight",
        "",
        f"Today's behaviour covered {len(summaries)} observation window(s) "
        f"and generated {len(ideas)} above-threshold idea(s).",
        "",
        "---",
        f"*Generated by Digital Unconscious · {date_str}*",
    ])

    return "\n".join(lines) + "\n"


def save_briefing(
    briefing_text: str,
    output_dir: Path,
    date_str: str | None = None,
) -> Path:
    """Save a briefing to disk and return the path."""
    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"briefing_{date_str}.md"
    path.write_text(briefing_text, encoding="utf-8")
    return path
