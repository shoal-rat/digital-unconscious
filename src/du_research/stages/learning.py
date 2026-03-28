from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from du_research.models import DatasetCandidate, PaperCandidate
from du_research.utils import infer_domain, iso_now, top_keywords


def build_learning_signal(
    run_id: str,
    idea_text: str,
    papers: list[PaperCandidate],
    datasets: list[DatasetCandidate],
    feasibility: dict[str, Any],
    analysis: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    critique_types = review.get("critique_types", [])
    domain = infer_domain([idea_text] + [paper.title for paper in papers[:10]])
    return {
        "run_id": run_id,
        "timestamp": iso_now(),
        "idea_text": idea_text,
        "domain": domain,
        "keywords": top_keywords([idea_text] + [paper.title for paper in papers[:5]], limit=8),
        "literature": {
            "paper_count": len(papers),
            "top_sources": [paper.source for paper in papers[:5]],
        },
        "datasets": {
            "dataset_count": len(datasets),
            "top_sources": [dataset.source for dataset in datasets[:5]],
        },
        "feasibility": feasibility,
        "analysis": {
            "analysis_executed": analysis.get("analysis_executed", False),
            "row_count": analysis.get("row_count"),
        },
        "review": {
            "overall_score": review.get("overall_score"),
            "critique_types": critique_types,
        },
        "blockers": [
            blocker
            for blocker, active in {
                "low_evidence": len(papers) < 3,
                "no_dataset_candidates": len(datasets) == 0,
                "no_local_data": not analysis.get("analysis_executed", False),
                "quality_below_threshold": not review.get("passes_threshold", False),
            }.items()
            if active
        ],
    }


def _load_signals(workspace_dir: Path) -> list[dict[str, Any]]:
    signals = []
    runs_dir = workspace_dir / "runs"
    if not runs_dir.exists():
        return signals
    for run_dir in sorted(runs_dir.iterdir()):
        signal_path = run_dir / "learning_signal.json"
        if signal_path.exists():
            signals.append(json.loads(signal_path.read_text(encoding="utf-8")))
    return signals


def update_learning_model(workspace_dir: Path, min_runs_before_update: int = 1) -> dict[str, Any] | None:
    signals = _load_signals(workspace_dir)
    if len(signals) < min_runs_before_update:
        return None

    learning_dir = workspace_dir / "learning"
    learning_dir.mkdir(parents=True, exist_ok=True)
    current_path = learning_dir / "human_idea_model.json"
    current_version = 0
    if current_path.exists():
        current_version = json.loads(current_path.read_text(encoding="utf-8")).get("model_version", 0)

    domain_counts = Counter(signal.get("domain", "general") for signal in signals)
    keyword_counter = Counter()
    source_counter = Counter()
    blocker_counter = Counter()
    quality_scores = []
    analyses_executed = 0
    for signal in signals:
        for item in signal.get("keywords", []):
            keyword_counter[item["keyword"]] += item.get("weight", 0)
        source_counter.update(signal.get("literature", {}).get("top_sources", []))
        source_counter.update(signal.get("datasets", {}).get("top_sources", []))
        blocker_counter.update(signal.get("blockers", []))
        score = signal.get("review", {}).get("overall_score")
        if score is not None:
            quality_scores.append(float(score))
        if signal.get("analysis", {}).get("analysis_executed"):
            analyses_executed += 1

    model = {
        "model_version": current_version + 1,
        "last_updated": iso_now(),
        "run_count": len(signals),
        "average_quality_score": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0,
        "top_domains": [
            {"domain": domain, "runs": count}
            for domain, count in domain_counts.most_common(5)
        ],
        "top_keywords": [
            {"keyword": keyword, "weight": round(weight, 4)}
            for keyword, weight in keyword_counter.most_common(10)
        ],
        "preferred_sources": [
            {"source": source, "count": count}
            for source, count in source_counter.most_common(8)
        ],
        "common_blockers": [
            {"blocker": blocker, "count": count}
            for blocker, count in blocker_counter.most_common(8)
        ],
        "analysis_execution_rate": round(analyses_executed / len(signals), 3) if signals else 0.0,
    }
    current_path.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")

    changes_md = learning_dir / "learning_changes.md"
    lines = [
        "# Learning Update",
        "",
        f"- Timestamp: {model['last_updated']}",
        f"- Model version: {model['model_version']}",
        f"- Runs analyzed: {model['run_count']}",
        f"- Average quality score: {model['average_quality_score']}",
        f"- Analysis execution rate: {model['analysis_execution_rate']}",
        "",
        "## Top Domains",
        "",
    ]
    for item in model["top_domains"]:
        lines.append(f"- {item['domain']}: {item['runs']} run(s)")
    lines.extend(["", "## Preferred Sources", ""])
    for item in model["preferred_sources"]:
        lines.append(f"- {item['source']}: {item['count']} appearances")
    lines.extend(["", "## Common Blockers", ""])
    if model["common_blockers"]:
        for item in model["common_blockers"]:
            lines.append(f"- {item['blocker']}: {item['count']}")
    else:
        lines.append("- No recurring blockers detected.")
    changes_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model
