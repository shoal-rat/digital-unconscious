"""Judge agent — evaluates and scores generated ideas.

Uses Claude Sonnet at low temperature for consistent, rigorous evaluation
across four dimensions: novelty, feasibility, domain relevance, and
timeliness.
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
    ) -> list[dict[str, Any]]:
        """Score each idea and return evaluations sorted by total_score."""
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
            logger.warning("Judge evaluation failed, using heuristic scoring")
            return _heuristic_evaluate(ideas)

        evaluations = _parse_evaluations(response.text)
        if not evaluations:
            return _heuristic_evaluate(ideas)

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


def _heuristic_evaluate(ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fallback scoring when AI is unavailable."""
    evaluations = []
    for idea in ideas:
        desc = idea.get("description", idea.get("title", ""))
        # Simple length-based heuristic as proxy for specificity
        specificity = min(len(desc) / 200, 1.0)
        has_data_hint = 1.0 if idea.get("data_hint") else 0.5
        has_question = 1.0 if idea.get("research_question") else 0.6
        domains = idea.get("domains", [])
        cross_domain = 1.0 if len(domains) >= 2 else 0.7

        novelty = round(55 + specificity * 25, 1)
        feasibility = round(50 + has_data_hint * 20 + has_question * 10, 1)
        domain_relevance = round(55 + cross_domain * 20, 1)
        timeliness = 60.0  # can't assess without behaviour context

        total = round(
            novelty * WEIGHTS["novelty"]
            + feasibility * WEIGHTS["feasibility"]
            + domain_relevance * WEIGHTS["domain_relevance"]
            + timeliness * WEIGHTS["timeliness"],
            1,
        )

        if total >= 75:
            verdict = "include"
        elif total >= 60:
            verdict = "hold"
        else:
            verdict = "discard"

        evaluations.append({
            "idea_id": idea.get("id", idea.get("idea_id", "")),
            "novelty": novelty,
            "feasibility": feasibility,
            "domain_relevance": domain_relevance,
            "timeliness": timeliness,
            "total_score": total,
            "verdict": verdict,
            "one_line_reason": "Scored by heuristic (AI unavailable)",
        })

    return sorted(evaluations, key=lambda e: e["total_score"], reverse=True)


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
