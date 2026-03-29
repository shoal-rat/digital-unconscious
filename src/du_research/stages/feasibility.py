from __future__ import annotations

import json
from pathlib import Path

import json
import logging

from du_research.models import PaperCandidate, StageResult
from du_research.utils import clamp, infer_domain, mean_or_zero, overlap_score, tokenize

logger = logging.getLogger(__name__)


FEASIBILITY_SYSTEM_PROMPT = """\
You are the Feasibility Assessor for an autonomous research pipeline.

Given a research idea and related literature, synthesize a go/no-go decision.

Evaluate:
1. Is there enough existing literature to ground this research?
2. Are the required datasets likely available (open or acquirable)?
3. Are the methods tractable with standard tools?
4. What is the novel angle that differentiates this from existing work?

Output ONLY valid JSON:
{
  "decision": "proceed" | "review" | "archive",
  "confidence": 0-100,
  "novel_angle": "What makes this idea fresh",
  "recommended_methods": ["method1", "method2"],
  "required_data": ["data type 1", "data type 2"],
  "estimated_effort": "low" | "medium" | "high",
  "key_risks": ["risk1", "risk2"],
  "reasoning": "2-3 sentence explanation of the decision"
}

Be rigorous. Most ideas should get confidence 40-70. Reserve 80+ for ideas with
clear data availability and a strong novel angle."""


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


def _ai_feasibility(
    idea_text: str,
    papers: list[PaperCandidate],
    backend: object,
) -> dict | None:
    """Use AI to synthesize literature into a feasibility assessment."""
    paper_summaries = []
    for p in papers[:8]:
        paper_summaries.append({
            "title": p.title,
            "summary": (p.summary or "")[:200],
            "year": p.year,
            "methods": getattr(p, "methods", []),
            "datasets_used": getattr(p, "datasets_used", []),
        })

    prompt = (
        f"## Research Idea\n{idea_text}\n\n"
        f"## Related Literature ({len(paper_summaries)} papers)\n"
        f"{json.dumps(paper_summaries, indent=2, ensure_ascii=False)}\n\n"
        f"Assess feasibility. Output ONLY valid JSON."
    )

    try:
        response = backend.call(
            prompt,
            mode="strict",
            system=FEASIBILITY_SYSTEM_PROMPT,
            model="opus",
            max_tokens=1500,
        )
        if not response.ok:
            return None

        text = response.text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
    except Exception as exc:
        logger.warning("AI feasibility assessment failed: %s", exc)
    return None


def run_stage(
    idea_text: str,
    papers: list[PaperCandidate],
    output_dir: Path,
    backend: object | None = None,
) -> tuple[dict, StageResult]:
    # Try AI-powered assessment first
    if backend is not None:
        ai_result = _ai_feasibility(idea_text, papers, backend)
        if ai_result and "decision" in ai_result and "confidence" in ai_result:
            output = {
                "decision": ai_result["decision"],
                "confidence": ai_result["confidence"],
                "domain": infer_domain([idea_text] + [paper.title for paper in papers[:10]]),
                "novel_angle": ai_result.get("novel_angle", ""),
                "recommended_methods": ai_result.get("recommended_methods", []),
                "required_data": ai_result.get("required_data", []),
                "estimated_effort": ai_result.get("estimated_effort", "medium"),
                "key_risks": ai_result.get("key_risks", []),
                "reasoning": ai_result.get("reasoning", ""),
                "assessment_mode": "ai",
                "signals": {
                    "paper_count": len(papers),
                    "average_relevance": round(mean_or_zero(paper.score for paper in papers[:8]), 3),
                },
            }
            report = [
                "# Feasibility Memo (AI-assessed)",
                "",
                f"- Decision: **{output['decision']}**",
                f"- Confidence: **{output['confidence']}/100**",
                f"- Domain: `{output['domain']}`",
                f"- Novel angle: {output['novel_angle']}",
                f"- Estimated effort: {output.get('estimated_effort', 'medium')}",
                "",
                "## Reasoning",
                "",
                output.get("reasoning", ""),
                "",
                "## Recommended Methods",
                "",
            ]
            for method in output.get("recommended_methods", []):
                report.append(f"- {method}")
            report.extend(["", "## Required Data", ""])
            for item in output.get("required_data", []):
                report.append(f"- {item}")
            if output.get("key_risks"):
                report.extend(["", "## Key Risks", ""])
                for risk in output["key_risks"]:
                    report.append(f"- {risk}")
            json_path = output_dir / "feasibility.json"
            md_path = output_dir / "memo.md"
            json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
            md_path.write_text("\n".join(report) + "\n", encoding="utf-8")
            result = StageResult(
                order=2,
                name="feasibility",
                status="completed",
                summary=f"AI feasibility: `{output['decision']}` with confidence {output['confidence']}/100.",
                artifacts=[str(json_path), str(md_path)],
                metrics={"confidence": output["confidence"], "decision": output["decision"], "mode": "ai"},
            )
            return output, result

    # No backend available — return None so engine can queue
    logger.warning("Feasibility assessment requires LLM — no backend provided")
    output = {
        "decision": "review",
        "confidence": 50,
        "domain": "",
        "novel_angle": "Pending LLM assessment",
        "recommended_methods": [],
        "required_data": [],
        "status": "queued_for_llm",
    }
    json_path = output_dir / "feasibility.json"
    md_path = output_dir / "memo.md"
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text("# Feasibility Memo\n\nPending — waiting for LLM.\n", encoding="utf-8")
    result = StageResult(
        order=2,
        name="feasibility",
        status="pending",
        summary="Feasibility assessment queued — LLM unavailable.",
        artifacts=[str(json_path), str(md_path)],
        metrics={"confidence": 0, "decision": "pending"},
    )
    return output, result
