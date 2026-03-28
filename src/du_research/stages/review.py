from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from du_research.agents.reviewer import ReviewerAgent
from du_research.agents.revision import RevisionAgent
from du_research.models import PaperCandidate, StageResult
from du_research.utils import clamp, word_count


REQUIRED_SECTIONS = [
    "## Abstract",
    "## Research Question",
    "## Literature Snapshot",
    "## Data Plan",
    "## Methods",
    "## Results Or Planned Analysis",
    "## Limitations",
    "## References",
]


def _reference_count(manuscript_text: str) -> int:
    in_references = False
    count = 0
    for line in manuscript_text.splitlines():
        if line.strip() == "## References":
            in_references = True
            continue
        if in_references and line.strip():
            count += 1
    return count


def _score_dimensions(
    manuscript_text: str,
    papers: list[PaperCandidate],
    feasibility: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, float]:
    section_presence = sum(1 for section in REQUIRED_SECTIONS if section in manuscript_text) / len(REQUIRED_SECTIONS)
    novelty = clamp(50 + feasibility.get("confidence", 0) * 0.35, 0, 100)
    statistical_rigor = 90 if analysis.get("analysis_executed") and analysis.get("numeric_summary") else 45
    reproducibility = 92 if analysis.get("reproducibility_check", {}).get("passed") else (65 if analysis.get("analysis_executed") else 40)
    figure_quality = 90 if analysis.get("figure_path") else 35
    abstract_accuracy = 88 if "confidence" in manuscript_text.lower() and "idea" in manuscript_text.lower() else 60
    overreach_detection = 88 if "provisional" in manuscript_text.lower() or "limitations" in manuscript_text.lower() else 55
    clarity = clamp((section_presence * 70) + (min(word_count(manuscript_text) / 450.0, 1.0) * 30), 0, 100)
    reference_quality = clamp(min(_reference_count(manuscript_text) / 5.0, 1.0) * 100, 0, 100)
    return {
        "novelty": round(novelty, 1),
        "statistical_rigor": round(statistical_rigor, 1),
        "clarity": round(clarity, 1),
        "reproducibility": round(reproducibility, 1),
        "figure_quality": round(figure_quality, 1),
        "abstract_accuracy": round(abstract_accuracy, 1),
        "overreach_detection": round(overreach_detection, 1),
        "reference_quality": round(reference_quality, 1),
    }


def _apply_revision(
    manuscript_text: str,
    critique_types: list[str],
    papers: list[PaperCandidate],
    analysis: dict[str, Any],
) -> str:
    revised = manuscript_text
    additions = []
    if "evidence" in critique_types and papers:
        evidence_lines = [
            "## Evidence Strengthening",
            "",
            "The revision highlights the strongest literature anchors identified in the scout stage:",
        ]
        for paper in papers[:3]:
            evidence_lines.append(f"- {paper.title} ({paper.year or 'n.d.'}, {paper.source})")
        additions.append("\n".join(evidence_lines))
    if "reference_quality" in critique_types and papers:
        evidence_lines = [
            "## Reference Quality Improvements",
            "",
            "The revision adds and foregrounds stronger citations from the ranked core references:",
        ]
        for paper in papers[:3]:
            evidence_lines.append(f"- {paper.title} ({paper.year or 'n.d.'}, {paper.source})")
        additions.append("\n".join(evidence_lines))
    if any(item in critique_types for item in ["statistical_rigor", "reproducibility", "figure_quality"]):
        repro = analysis.get("reproducibility_check", {})
        repro_lines = [
            "## Reproducibility Note",
            "",
            f"- Analysis executed: {analysis.get('analysis_executed', False)}",
            f"- Reproducibility check passed: {repro.get('passed', False)}",
            f"- Processed data artifact: {analysis.get('processed_file', 'n/a')}",
        ]
        additions.append("\n".join(repro_lines))
    if "clarity" in critique_types or "abstract_accuracy" in critique_types or "overreach_detection" in critique_types:
        additions.append(
            "\n".join(
                [
                    "## Clarity Improvements",
                    "",
                    "This revised draft makes the outcome variable, candidate data source, and next analytical step explicit.",
                ]
            )
        )
    if additions:
        revised = revised.rstrip() + "\n\n" + "\n\n".join(additions) + "\n"
    return revised


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
            dimensions = _score_dimensions(current_text, papers, feasibility, analysis)
            suggestions = []
            critique_types = []
            for key, score in dimensions.items():
                if score < 70:
                    critique_types.append(key)
                    suggestions.append(f"Raise `{key}` by adding stronger evidence or clearer execution details.")
        if feasibility.get("decision") == "archive":
            suggestions.append("Treat this run as exploratory; the evidence base is currently too weak for a strong claim.")
        overall = round(sum(dimensions[key] * weight for key, weight in weights.items()), 1)
        payload = {
            "overall_score": overall,
            "threshold": quality_threshold,
            "passes_threshold": overall >= quality_threshold,
            "dimension_scores": dimensions,
            "reference_count": _reference_count(current_text),
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
        revised = None
        if revision_agent is not None:
            revised = revision_agent.revise(current_text, payload)
        current_text = revised or _apply_revision(current_text, critique_types, papers, analysis)
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
        md_lines.append("- No blocking issues detected in the MVP rubric.")
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
