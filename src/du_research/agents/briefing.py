"""Daily Idea Briefing generator — assembles the final markdown output.

Takes the day's behaviour summaries, scored ideas, and optional research
results and produces a single reader-friendly markdown briefing.

LLM-only: no heuristic fallback. If the AI is unreachable, returns None
so the engine can queue the task for later.
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
    ) -> str | None:
        """Generate the daily markdown briefing.

        Returns None if the LLM is unavailable (caller should queue).
        """
        date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        parts = [f"## Date: {date_str}"]

        parts.append("\n## Behaviour Summaries")
        for i, summary in enumerate(behaviour_summaries[:6], 1):
            parts.append(f"\n### Window {i}")
            parts.append(json.dumps(summary, indent=2, ensure_ascii=False))

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

        top_ideas.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        included = [i for i in top_ideas if i.get("verdict") != "discard"][:10]

        parts.append(f"\n## Scored Ideas ({len(included)} above threshold)")
        parts.append(json.dumps(included, indent=2, ensure_ascii=False))

        if human_idea_model:
            parts.append("\n## User's Idea Model Summary")
            model_brief = {
                k: human_idea_model[k]
                for k in ["top_domains", "top_keywords", "average_quality_score",
                           "core_obsessions", "recurring_blind_spots"]
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

        logger.warning("Briefing generation failed — LLM unavailable")
        return None


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
