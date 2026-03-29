"""Creative Idea Generation agent — transforms behaviour summaries into ideas.

Uses Claude Opus at high temperature to generate novel, cross-domain
research/product ideas from compressed screen behaviour summaries combined
with RAG knowledge context.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from du_research.ai_backend import AIBackend

logger = logging.getLogger(__name__)


IDEA_GENERATOR_SYSTEM_PROMPT = """\
You are the Creative Engine of Digital Unconscious — a system that transforms
a human's passive daily screen behaviour into actionable research and product
ideas.

You receive:
1. A compressed behaviour summary (what the human browsed, read, searched)
2. Domain configuration (primary, secondary, open domains)
3. Optional: RAG knowledge context from the user's personal knowledge base
4. Optional: Human Idea Model (the user's intellectual fingerprint)

Your job:
- Generate 5-8 distinct, specific, actionable ideas
- Each idea MUST connect the human's observed behaviour to a concrete
  research question or product concept
- Favour CROSS-DOMAIN connections — the most valuable ideas come from
  connecting the user's primary domain with unexpected adjacent fields
- Be BOLD — prefer novel over safe
- Every idea must be testable with real data or a concrete experiment

Output ONLY valid JSON — no markdown, no explanation:
{
  "ideas": [
    {
      "id": "idea_001",
      "title": "Short compelling title (max 80 chars)",
      "description": "2-3 sentence description of the idea, the hypothesis, and how to test it",
      "source_behaviour": "Which observed behaviour triggered this idea",
      "domains": ["primary_domain", "crossing_domain"],
      "research_question": "The specific question this idea answers",
      "data_hint": "What kind of data would be needed to test this",
      "novelty_signal": "Why this hasn't been done before (or why the angle is fresh)"
    }
  ]
}

Think divergently. Break patterns. Connect the unexpected.
"""


@dataclass
class IdeaGeneratorAgent:
    """Generates creative ideas from behaviour summaries."""

    backend: AIBackend
    model: str = "opus"
    system_prompt: str | None = None
    primary_domains: list[str] = field(default_factory=lambda: ["AI tools"])
    secondary_domains: list[str] = field(default_factory=lambda: ["cognitive science"])
    focus_fields: list[str] = field(default_factory=list)

    def generate(
        self,
        behaviour_summary: dict[str, Any],
        *,
        rag_context: str | None = None,
        human_idea_model: dict[str, Any] | None = None,
        idea_count: int = 6,
    ) -> list[dict[str, Any]]:
        """Generate ideas from a compressed behaviour summary."""
        parts = [
            f"## Today's Behaviour Summary\n{json.dumps(behaviour_summary, indent=2, ensure_ascii=False)}",
            f"\n## Domain Configuration\n"
            f"- Primary domains: {', '.join(self.primary_domains)}\n"
            f"- Secondary domains: {', '.join(self.secondary_domains)}\n"
            f"- Open domain: any cross-domain connection is welcome",
        ]

        if self.focus_fields:
            parts.append(
                f"\n## Focus Fields (IMPORTANT)\n"
                f"The user's research/work focus: {', '.join(self.focus_fields)}\n"
                f"Every idea MUST be applicable to at least one of these fields.\n"
                f"Cross-domain inspiration is encouraged — draw from ANY field — but the\n"
                f"idea's application and research question must land in the focus fields.\n"
                f"Add an 'application_field' key to each idea indicating which focus field it serves."
            )

        if rag_context:
            parts.append(f"\n## Knowledge Base Context\n{rag_context[:3000]}")

        if human_idea_model:
            model_summary = _format_idea_model(human_idea_model)
            parts.append(f"\n## Human Idea Model (intellectual fingerprint)\n{model_summary}")

        parts.append(f"\nGenerate {idea_count} ideas. Output ONLY valid JSON.")

        prompt = "\n".join(parts)

        response = self.backend.call(
            prompt,
            mode="creative",
            system=self.system_prompt or IDEA_GENERATOR_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=3000,
        )

        if not response.ok:
            logger.warning("Idea generation failed: %s", response.raw.get("error"))
            return []

        return _parse_ideas(response.text)


def _format_idea_model(model: dict[str, Any]) -> str:
    """Format the human idea model for inclusion in the prompt."""
    lines = []
    obsessions = model.get("core_obsessions", model.get("top_domains", []))
    if obsessions:
        lines.append("Core obsessions: " + ", ".join(
            item.get("theme", item.get("domain", str(item)))
            for item in obsessions[:5]
        ))

    crossings = model.get("productive_crossings", model.get("preferred_analogy_domains", []))
    if crossings:
        if isinstance(crossings[0], dict):
            lines.append("Productive crossings: " + ", ".join(
                f"{c.get('from', '')}→{c.get('to', '')}" for c in crossings[:4]
            ))
        else:
            lines.append("Analogy domains: " + ", ".join(str(c) for c in crossings[:4]))

    blind_spots = model.get("recurring_blind_spots", [])
    if blind_spots:
        lines.append("Known blind spots: " + "; ".join(str(b) for b in blind_spots[:3]))

    good_idea = model.get("what_makes_a_good_idea_for_this_user", [])
    if good_idea:
        lines.append("What makes a good idea: " + "; ".join(str(g) for g in good_idea[:3]))

    return "\n".join(lines) if lines else "No model data yet."


def _parse_ideas(text: str) -> list[dict[str, Any]]:
    """Extract ideas list from potentially messy AI output."""
    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "ideas" in data:
            return data["ideas"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, dict) and "ideas" in data:
                return data["ideas"]
        except json.JSONDecodeError:
            pass

    # Try array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse ideas from AI output")
    return []
