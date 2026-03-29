"""Judge agent — evaluates and scores generated ideas.

Uses Claude Sonnet at low temperature for consistent, rigorous evaluation
across four dimensions: novelty, feasibility, domain relevance, and
timeliness.

LLM-only: no heuristic fallback. If the AI is unreachable, returns None
so the engine can queue the task for later.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from du_research.ai_backend import AIBackend

logger = logging.getLogger(__name__)


JUDGE_SYSTEM_PROMPT = """\
You are the Judge Agent of Digital Unconscious — a strict, adversarial
evaluator of research and product ideas.

You evaluate each idea on exactly 4 dimensions (each 0-100):

1. **Novelty (25%)**: How different is this from the user's existing ideas
   and from obvious/generic suggestions? Penalise heavily for ideas that
   are common knowledge or have been extensively studied.

2. **Feasibility (30%)**: Can this realistically be turned into a study or
   product? Is the data likely available? Are the methods tractable?
   Penalise moonshots with no clear execution path.

3. **Domain Relevance (25%)**: Does this connect to the user's configured
   primary or secondary domains? Cross-domain ideas get a bonus if the
   bridge is plausible; a penalty if forced.

4. **Timeliness (20%)**: Is this connected to the user's CURRENT behaviour
   and interests (today's observation)? Penalise ideas that seem generic
   and disconnected from today's context.

Scoring rules:
- Be HARSH. Most ideas should score 50-70. Reserve 80+ for genuinely
  compelling ideas.
- Total score = weighted sum of dimensions.
- Score < 60: discard
- Score 60-75: hold in observation pool (not in daily briefing)
- Score > 75: include in daily briefing

Output ONLY valid JSON:
{
  "evaluations": [
    {
      "idea_id": "idea_001",
      "novelty": 72,
      "feasibility": 65,
      "domain_relevance": 80,
      "timeliness": 58,
      "total_score": 68.9,
      "verdict": "hold",
      "one_line_reason": "Interesting angle but data availability is unclear"
    }
  ]
}

Verdicts: "discard" (<60), "hold" (60-75), "include" (>75).
"""


WEIGHTS = {
    "novelty": 0.25,
    "feasibility": 0.30,
    "domain_relevance": 0.25,
    "timeliness": 0.20,
}


@dataclass
class JudgeAgent:
    """Evaluates and scores a batch of generated ideas."""

    backend: AIBackend
    model: str = "sonnet"
    system_prompt: str | None = None

    def evaluate(
        self,
        ideas: list[dict[str, Any]],
        *,
        behaviour_summary: dict[str, Any] | None = None,
        primary_domains: list[str] | None = None,
        existing_ideas: list[str] | None = None,
        human_idea_model: dict[str, Any] | None = None,
        focus_fields: list[str] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Score each idea and return evaluations sorted by total_score.

        Returns None if the LLM is unavailable (caller should queue).
        """
        if not ideas:
            return []

        parts = [f"## Ideas to Evaluate\n{json.dumps(ideas, indent=2, ensure_ascii=False)}"]

        if behaviour_summary:
            parts.append(
                f"\n## Today's Behaviour Context\n"
                f"{json.dumps(behaviour_summary, indent=2, ensure_ascii=False)}"
            )

        if primary_domains:
            parts.append(f"\n## User's Domains\n{', '.join(primary_domains)}")

        if existing_ideas:
            parts.append(
                f"\n## Recent Existing Ideas (for novelty comparison)\n"
                + "\n".join(f"- {idea}" for idea in existing_ideas[:20])
            )

        if focus_fields:
            parts.append(
                f"\n## Focus Fields (CRITICAL)\n"
                f"The user ONLY cares about ideas applicable to: {', '.join(focus_fields)}\n"
                f"Ideas whose application_field does NOT match any focus field should get\n"
                f"domain_relevance < 30 regardless of other qualities.\n"
                f"Cross-domain INSPIRATION is fine — but the idea must LAND in a focus field."
            )

        if human_idea_model:
            model_hints = _format_judge_model_context(human_idea_model)
            if model_hints:
                parts.append(f"\n## Personalized Evaluation Context\n{model_hints}")

        parts.append(f"\nEvaluate all {len(ideas)} ideas. Output ONLY valid JSON.")
        prompt = "\n".join(parts)

        response = self.backend.call(
            prompt,
            mode="strict",
            system=self.system_prompt or JUDGE_SYSTEM_PROMPT,
            model=self.model,
            max_tokens=2000,
        )

        if not response.ok:
            logger.warning("Judge evaluation failed — LLM unavailable")
            return None

        evaluations = _parse_evaluations(response.text)
        if not evaluations:
            logger.warning("Could not parse judge output")
            return None

        # Ensure total_score is computed consistently
        for ev in evaluations:
            ev["total_score"] = round(
                sum(ev.get(dim, 50) * w for dim, w in WEIGHTS.items()), 1
            )
            if ev["total_score"] >= 75:
                ev["verdict"] = "include"
            elif ev["total_score"] >= 60:
                ev["verdict"] = "hold"
            else:
                ev["verdict"] = "discard"

        return sorted(evaluations, key=lambda e: e["total_score"], reverse=True)


def _format_judge_model_context(model: dict[str, Any]) -> str:
    """Extract relevant hints from the human idea model for personalized judging."""
    lines = []
    good_idea = model.get("what_makes_a_good_idea_for_this_user", [])
    if good_idea:
        lines.append("What historically makes a good idea for this user:")
        for criterion in good_idea[:4]:
            lines.append(f"  - {criterion}")

    blind_spots = model.get("recurring_blind_spots", [])
    if blind_spots:
        lines.append("Known blind spots (be lenient on ideas that counterbalance these):")
        for spot in blind_spots[:3]:
            lines.append(f"  - {spot}")

    lifecycle = model.get("idea_lifecycle", {})
    conversion = lifecycle.get("conversion_rate", 0)
    if conversion > 0:
        lines.append(f"Historical idea-to-paper conversion rate: {conversion:.1%}")

    return "\n".join(lines)


def _parse_evaluations(text: str) -> list[dict[str, Any]]:
    """Parse evaluations from AI response."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "evaluations" in data:
            return data["evaluations"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end])
            if isinstance(data, dict) and "evaluations" in data:
                return data["evaluations"]
        except json.JSONDecodeError:
            pass
    return []
