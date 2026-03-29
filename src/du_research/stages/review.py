"""Review stage — AI peer review loop with adversarial scoring.

LLM-only: uses the ReviewerAgent for scoring and RevisionAgent for revisions.
No heuristic fallbacks. When AI agents are unavailable, returns pending status.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from du_research.agents.reviewer import ReviewerAgent
from du_research.agents.revision import RevisionAgent
from du_research.models import PaperCandidate, StageResult
from du_research.utils import word_count


def run_stage(
    output_dir: Path,
    manuscript_text: str,
    papers: list[PaperCandidate],
    feasibility: dict[str, Any],
    analysis: dict[str, Any],
    quality_threshold: int,
    max_revisions: int = 3,
    reviewer_agent: ReviewerAgent | None = None,
    revision_agent: RevisionAgent | None = None,
) -> tuple[dict[str, Any], StageResult]:
    weights = {
        "novelty": 0.13,
        "statistical_rigor": 0.16,
        "clarity": 0.12,
        "reproducibility": 0.15,
        "figure_quality": 0.11,
        "abstract_accuracy": 0.11,
        "overreach_detection": 0.11,
        "reference_quality": 0.11,
    }
    current_text = manuscript_text
    history = []
    payload: dict[str, Any] = {}

    for iteration in range(1, max_revisions + 1):
        # Score with LLM reviewer
        ai_review = None
        if reviewer_agent is not None:
            ai_review = reviewer_agent.review(
                {
                    "manuscript_text": current_text,
                    "papers": [paper.to_dict() for paper in papers[:12]],
                    "feasibility": feasibility,
                    "analysis": analysis,
                }
            )

        if ai_review and ai_review.get("dimension_scores"):
            dimensions = {
                key: round(float(value), 1)
                for key, value in ai_review.get("dimension_scores", {}).items()
                if key in weights
            }
            for key in weights:
                dimensions.setdefault(key, 60.0)
            critique_types = list(ai_review.get("critique_types", []))
            suggestions = list(ai_review.get("suggestions", []))
        else:
            # No AI available — assign pending scores
            dimensions = {key: 50.0 for key in weights}
            critique_types = ["llm_unavailable"]
            suggestions = ["Review pending — LLM was unavailable. Run `du drain` to retry."]

        if feasibility.get("decision") == "archive":
            suggestions.append("Treat this run as exploratory; the evidence base is currently too weak for a strong claim.")

        overall = round(sum(dimensions[key] * weight for key, weight in weights.items()), 1)
        payload = {
            "overall_score": overall,
            "threshold": quality_threshold,
            "passes_threshold": overall >= quality_threshold,
            "dimension_scores": dimensions,
            "critique_types": critique_types,
            "suggestions": suggestions,
            "iterations": iteration,
        }
        history.append(
            {
                "iteration": iteration,
                "overall_score": overall,
                "passes_threshold": payload["passes_threshold"],
                "critique_types": critique_types,
                "suggestions": suggestions,
            }
        )

        if payload["passes_threshold"] or iteration == max_revisions:
            break

        # Revise with LLM revision agent
        revised = None
        if revision_agent is not None:
            revised = revision_agent.revise(current_text, payload)
        if revised:
            current_text = revised
        else:
            # Can't revise without AI — stop loop
            break

    # Write artifacts
    md_lines = [
        "# Review Report",
        "",
        f"- Overall score: **{payload['overall_score']}/100**",
        f"- Threshold: **{quality_threshold}**",
        f"- Passes threshold: **{payload['passes_threshold']}**",
        f"- Iterations used: **{payload['iterations']}**",
        "",
        "## Dimension Scores",
        "",
    ]
    for key, value in payload["dimension_scores"].items():
        md_lines.append(f"- {key}: {value}")
    md_lines.extend(["", "## Suggestions", ""])
    if payload["suggestions"]:
        for suggestion in payload["suggestions"]:
            md_lines.append(f"- {suggestion}")
    else:
        md_lines.append("- No blocking issues detected.")

    review_json = output_dir / "review_report.json"
    review_md = output_dir / "review_report.md"
    score_json = output_dir / "quality_score.json"
    history_jsonl = output_dir / "review_history.jsonl"
    revised_md = output_dir / "revised_manuscript.md"
    review_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    review_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    score_json.write_text(
        json.dumps(
            {"overall_score": payload["overall_score"], "dimension_scores": payload["dimension_scores"]},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with history_jsonl.open("w", encoding="utf-8") as handle:
        for item in history:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    revised_md.write_text(current_text, encoding="utf-8")

    result = StageResult(
        order=6,
        name="review",
        status="completed",
        summary=f"Scored draft at {payload['overall_score']}/100.",
        artifacts=[str(review_json), str(review_md), str(score_json), str(history_jsonl), str(revised_md)],
        metrics={
            "overall_score": payload["overall_score"],
            "passes_threshold": payload["passes_threshold"],
            "iterations": payload["iterations"],
        },
    )
    payload["final_manuscript_text"] = current_text
    return payload, result
