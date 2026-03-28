from __future__ import annotations

import json
from pathlib import Path

from du_research.models import PaperCandidate, StageResult
from du_research.utils import clamp, infer_domain, mean_or_zero, overlap_score, tokenize


METHOD_RULES = {
    "regression": ["regression", "predict", "pricing", "effect"],
    "survey": ["survey", "attitude", "perception", "self-report"],
    "experiment": ["experiment", "randomized", "intervention", "trial"],
    "content-analysis": ["text", "language", "content", "document"],
    "observational": ["behavior", "usage", "event", "log"],
}


def _recommended_methods(idea_text: str, papers: list[PaperCandidate]) -> list[str]:
    combined_text = " ".join([idea_text] + [paper.title + " " + paper.summary for paper in papers[:8]])
    tokens = set(tokenize(combined_text))
    methods = []
    for method, hints in METHOD_RULES.items():
        if tokens & set(hints):
            methods.append(method)
    return methods or ["observational", "descriptive-statistics"]


def _distinctive_angle(idea_text: str, papers: list[PaperCandidate]) -> str:
    idea_tokens = tokenize(idea_text)
    literature_tokens = set(tokenize(" ".join(paper.title for paper in papers[:10])))
    unique = [token for token in idea_tokens if token not in literature_tokens]
    if unique:
        return "Focus the study around: " + ", ".join(unique[:3])
    return "Differentiate via clearer variable definition, target population, or context."


def run_stage(
    idea_text: str,
    papers: list[PaperCandidate],
    output_dir: Path,
) -> tuple[dict, StageResult]:
    paper_count = len(papers)
    avg_score = mean_or_zero(paper.score for paper in papers[:8])
    recent_ratio = mean_or_zero((1.0 if (paper.year or 0) >= 2020 else 0.4) for paper in papers[:8])
    empirical_signal = mean_or_zero(
        overlap_score(idea_text, f"{paper.title} {paper.summary}") for paper in papers[:8]
    )
    specificity = min(len(tokenize(idea_text)) / 10.0, 1.0)
    confidence = round(
        clamp(
            (min(paper_count / 8.0, 1.0) * 35)
            + (avg_score * 25)
            + (recent_ratio * 15)
            + (empirical_signal * 15)
            + (specificity * 10),
            0,
            100,
        ),
        1,
    )
    if confidence >= 70:
        decision = "proceed"
    elif confidence >= 50:
        decision = "review"
    else:
        decision = "archive"

    recommended_methods = _recommended_methods(idea_text, papers)
    output = {
        "decision": decision,
        "confidence": confidence,
        "domain": infer_domain([idea_text] + [paper.title for paper in papers[:10]]),
        "novel_angle": _distinctive_angle(idea_text, papers),
        "recommended_methods": recommended_methods,
        "required_data": [
            "entity-level observations",
            "outcome variable",
            "time or cohort metadata",
        ],
        "signals": {
            "paper_count": paper_count,
            "average_relevance": round(avg_score, 3),
            "recent_ratio": round(recent_ratio, 3),
            "specificity": round(specificity, 3),
        },
    }
    report = [
        "# Feasibility Memo",
        "",
        f"- Decision: **{decision}**",
        f"- Confidence: **{confidence}/100**",
        f"- Domain: `{output['domain']}`",
        f"- Novel angle: {output['novel_angle']}",
        "",
        "## Recommended Methods",
        "",
    ]
    for method in recommended_methods:
        report.append(f"- {method}")
    report.extend(["", "## Required Data", ""])
    for item in output["required_data"]:
        report.append(f"- {item}")
    json_path = output_dir / "feasibility.json"
    md_path = output_dir / "memo.md"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    result = StageResult(
        order=2,
        name="feasibility",
        status="completed",
        summary=f"Feasibility decision `{decision}` with confidence {confidence}/100.",
        artifacts=[str(json_path), str(md_path)],
        metrics={"confidence": confidence, "decision": decision},
    )
    return output, result
