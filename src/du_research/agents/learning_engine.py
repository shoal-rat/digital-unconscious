"""Enhanced Learning Engine — prompt evolution, human idea model, domain knowledge.

Every completed pipeline run and daily idea cycle produces a learning_signal.
The Learning Engine studies accumulated signals and improves the system:

1. **Run Outcome Analyzer** — identifies patterns across runs.
2. **Human Idea Model Builder** — builds a persistent model of the user's
   intellectual fingerprint.
3. **Prompt Evolution Engine** — proposes incremental improvements to agent
   prompts based on performance data.
4. **Domain Knowledge Expander** — decides which knowledge base areas to grow.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from du_research.ai_backend import AIBackend
from du_research.utils import iso_now, top_keywords

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Run Outcome Analyzer
# ---------------------------------------------------------------------------


def analyze_run_outcomes(signals: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate learning signals and identify actionable patterns."""
    if not signals:
        return {"patterns": [], "run_count": 0}

    domain_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    blocker_counter: Counter[str] = Counter()
    quality_scores: list[float] = []
    critique_counter: Counter[str] = Counter()

    for signal in signals:
        domain_counter[signal.get("domain", "general")] += 1
        for kw in signal.get("keywords", []):
            keyword_counter[kw.get("keyword", "")] += 1
        for src in signal.get("literature", {}).get("top_sources", []):
            source_counter[src] += 1
        blocker_counter.update(signal.get("blockers", []))
        score = signal.get("review", {}).get("overall_score")
        if score is not None:
            quality_scores.append(float(score))
        for ct in signal.get("review", {}).get("critique_types", []):
            critique_counter[ct] += 1

    avg_quality = round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0
    recent_scores = quality_scores[-3:]
    earlier_scores = quality_scores[:-3]
    recent_average = round(sum(recent_scores) / len(recent_scores), 2) if recent_scores else avg_quality
    earlier_average = round(sum(earlier_scores) / len(earlier_scores), 2) if earlier_scores else avg_quality

    # Identify patterns
    patterns: list[dict[str, str]] = []

    # Most common critique type
    if critique_counter:
        top_critique, count = critique_counter.most_common(1)[0]
        ratio = count / len(signals)
        if ratio > 0.5:
            patterns.append({
                "type": "review_bottleneck",
                "insight": f"`{top_critique}` appears in {ratio:.0%} of reviews",
                "action": f"Add pre-check for {top_critique} before the main review loop",
            })

    # Database effectiveness
    if source_counter:
        best_src, best_count = source_counter.most_common(1)[0]
        patterns.append({
            "type": "database_effectiveness",
            "insight": f"`{best_src}` is the most productive source ({best_count} appearances)",
            "action": f"Prioritise {best_src} in literature search order",
        })

    # Common blockers
    if blocker_counter:
        top_blocker, b_count = blocker_counter.most_common(1)[0]
        if b_count >= 2:
            patterns.append({
                "type": "recurring_blocker",
                "insight": f"`{top_blocker}` blocked {b_count} runs",
                "action": f"Address {top_blocker} systemically",
            })

    return {
        "run_count": len(signals),
        "average_quality": avg_quality,
        "recent_average_quality": recent_average,
        "historical_average_quality": earlier_average,
        "top_domains": [{"domain": d, "count": c} for d, c in domain_counter.most_common(5)],
        "top_keywords": [{"keyword": k, "count": c} for k, c in keyword_counter.most_common(10)],
        "top_sources": [{"source": s, "count": c} for s, c in source_counter.most_common(8)],
        "common_blockers": [{"blocker": b, "count": c} for b, c in blocker_counter.most_common(5)],
        "common_critiques": [{"type": t, "count": c} for t, c in critique_counter.most_common(5)],
        "patterns": patterns,
    }


# ---------------------------------------------------------------------------
# 2. Human Idea Model Builder
# ---------------------------------------------------------------------------


def build_human_idea_model(
    signals: list[dict[str, Any]],
    daily_ideas: list[dict[str, Any]] | None = None,
    existing_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build or update the persistent human idea model."""
    existing_model = existing_model or {}
    model_version = existing_model.get("model_version", 0) + 1

    # Aggregate domains, keywords, sources from all signals
    domain_counter: Counter[str] = Counter()
    keyword_counter: Counter[str] = Counter()
    quality_by_domain: dict[str, list[float]] = {}

    for signal in signals:
        domain = signal.get("domain", "general")
        domain_counter[domain] += 1
        for kw in signal.get("keywords", []):
            keyword_counter[kw.get("keyword", "")] += 1
        score = signal.get("review", {}).get("overall_score")
        if score is not None:
            quality_by_domain.setdefault(domain, []).append(float(score))

    # Build domain attention map from daily ideas
    ideas_total = 0
    ideas_by_domain: Counter[str] = Counter()
    if daily_ideas:
        for idea in daily_ideas:
            domain = idea.get("domain", "general")
            ideas_by_domain[domain] += 1
            ideas_total += 1

    # Core obsessions
    core_obsessions = []
    for domain, count in domain_counter.most_common(5):
        scores = quality_by_domain.get(domain, [])
        avg = round(sum(scores) / len(scores), 2) if scores else 0
        trend = "stable"
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            older_avg = sum(scores[:-3]) / max(len(scores) - 3, 1)
            trend = "growing" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
        core_obsessions.append({
            "theme": domain,
            "strength": round(count / max(len(signals), 1), 2),
            "trend": trend,
        })

    # Idea lifecycle
    ideas_reaching_research = sum(1 for s in signals if s.get("review", {}).get("overall_score", 0) > 0)
    ideas_completing = sum(
        1 for s in signals
        if s.get("review", {}).get("overall_score", 0) >= 75
    )

    # What makes a good idea for this user
    good_idea_criteria = existing_model.get("what_makes_a_good_idea_for_this_user", [
        "Combines an observable human behavior pattern with a measurable outcome",
        "Has a clear 'why hasn't this been studied yet' angle",
        "Can be tested with publicly available behavioral or usage data",
    ])

    # Blind spots
    blind_spots = existing_model.get("recurring_blind_spots", [])
    blocker_counter: Counter[str] = Counter()
    for signal in signals:
        blocker_counter.update(signal.get("blockers", []))
    if blocker_counter.get("no_dataset_candidates", 0) >= 2:
        spot = "Underestimates data availability challenges"
        if spot not in blind_spots:
            blind_spots.append(spot)

    model = {
        "model_version": model_version,
        "last_updated": iso_now(),
        "core_obsessions": core_obsessions,
        "top_keywords": [
            {"keyword": k, "weight": round(c / max(sum(keyword_counter.values()), 1), 4)}
            for k, c in keyword_counter.most_common(15)
        ],
        "domain_attention": {
            domain: {
                "idea_count": count,
                "idea_fraction": round(count / max(ideas_total, 1), 2),
            }
            for domain, count in ideas_by_domain.most_common(6)
        },
        "idea_lifecycle": {
            "ideas_generated_total": ideas_total,
            "ideas_reaching_research": ideas_reaching_research,
            "ideas_completing_paper": ideas_completing,
            "conversion_rate": round(
                ideas_completing / max(ideas_reaching_research, 1), 3
            ),
        },
        "what_makes_a_good_idea_for_this_user": good_idea_criteria,
        "recurring_blind_spots": blind_spots[:5],
    }
    return model


# ---------------------------------------------------------------------------
# 3. Prompt Evolution Engine
# ---------------------------------------------------------------------------


@dataclass
class PromptEvolutionEngine:
    """Proposes incremental improvements to agent prompts."""

    backend: AIBackend
    prompts_dir: Path
    min_runs: int = 3

    def evolve(
        self,
        agent_name: str,
        current_prompt: str,
        run_outcomes: dict[str, Any],
    ) -> dict[str, Any]:
        """Propose an evolution of the prompt for *agent_name*.

        Returns a dict with ``proposed_prompt``, ``reasoning``, and
        ``version``.  Returns ``None`` if evolution is skipped.
        """
        run_count = run_outcomes.get("run_count", 0)
        if run_count < self.min_runs:
            return {
                "evolved": False,
                "reason": f"Not enough runs ({run_count} < {self.min_runs})",
            }

        patterns = run_outcomes.get("patterns", [])
        if not patterns:
            return {"evolved": False, "reason": "No actionable patterns found"}

        # Read prompt version history
        agent_dir = self.prompts_dir / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)
        evolution_log = agent_dir / "evolution_log.jsonl"
        current_version = _count_versions(agent_dir)

        prompt = (
            f"You are a prompt engineer analyzing agent performance data.\n\n"
            f"Current prompt for {agent_name}:\n```\n{current_prompt[:2000]}\n```\n\n"
            f"Performance patterns:\n{json.dumps(patterns, indent=2)}\n\n"
            f"Identify the single most impactful change to this prompt "
            f"that would reduce the most common failure pattern.\n"
            f"Return a minimal, surgical edit — not a rewrite.\n"
            f"Explain your reasoning.\n\n"
            f"Output JSON: {{\"edit\": \"the text to add/change\", "
            f"\"location\": \"where in the prompt\", "
            f"\"reasoning\": \"why this helps\"}}"
        )

        response = self.backend.call(
            prompt,
            mode="balanced",
            system="You are a prompt engineer. Be precise and conservative.",
            model="sonnet",
            max_tokens=1000,
        )

        if not response.ok:
            return {"evolved": False, "reason": "AI call failed"}

        try:
            edit_data = json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON
            text = response.text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    edit_data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    return {"evolved": False, "reason": "Could not parse edit proposal"}
            else:
                return {"evolved": False, "reason": "Could not parse edit proposal"}

        new_version = current_version + 1
        version_name = f"v{new_version}.0.0"

        # Save the proposed prompt
        prompt_file = agent_dir / f"{version_name}_proposed.txt"
        proposed_prompt = current_prompt.rstrip() + "\n\nLearned refinement:\n" + edit_data.get("edit", "").strip() + "\n"
        prompt_file.write_text(
            f"# Edit: {edit_data.get('edit', '')}\n"
            f"# Location: {edit_data.get('location', '')}\n"
            f"# Reasoning: {edit_data.get('reasoning', '')}\n\n"
            f"{proposed_prompt}",
            encoding="utf-8",
        )

        # Log the evolution
        log_entry = {
            "timestamp": iso_now(),
            "agent": agent_name,
            "version": version_name,
            "edit": edit_data.get("edit", ""),
            "reasoning": edit_data.get("reasoning", ""),
            "status": "proposed",
        }
        with evolution_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return {
            "evolved": True,
            "version": version_name,
            "edit": edit_data,
            "proposed_prompt": proposed_prompt,
            "prompt_file": str(prompt_file),
        }


def _count_versions(agent_dir: Path) -> int:
    return sum(1 for f in agent_dir.glob("v*.txt"))


# ---------------------------------------------------------------------------
# 4. Domain Knowledge Expander
# ---------------------------------------------------------------------------


@dataclass
class DomainKnowledgeExpander:
    workspace_dir: Path

    def expand(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
        knowledge_dir = self.workspace_dir / "knowledge"
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        base_path = knowledge_dir / "domain_knowledge.json"
        existing = {"papers": [], "domains": []}
        if base_path.exists():
            try:
                existing = json.loads(base_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        domain_counter: Counter[str] = Counter(signal.get("domain", "general") for signal in signals)
        paper_counter: Counter[str] = Counter()
        for run_dir in sorted((self.workspace_dir / "runs").glob("*")):
            papers_path = run_dir / "01_literature" / "papers.json"
            if not papers_path.exists():
                continue
            try:
                payload = json.loads(papers_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for paper in payload.get("papers", [])[:5]:
                key = paper.get("doi") or paper.get("title")
                if key:
                    paper_counter[key] += 1

        knowledge = {
            "updated_at": iso_now(),
            "domains": [{"domain": domain, "count": count} for domain, count in domain_counter.most_common(10)],
            "papers": [{"paper": paper, "count": count} for paper, count in paper_counter.most_common(25)],
            "previous_version": existing,
        }
        base_path.write_text(json.dumps(knowledge, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"knowledge_path": str(base_path), "domain_count": len(knowledge["domains"])}


# ---------------------------------------------------------------------------
# 5. Meta-Learning Scheduler
# ---------------------------------------------------------------------------


@dataclass
class MetaLearningScheduler:
    workspace_dir: Path
    min_runs_before_evolution: int = 3

    def decide(self, run_outcomes: dict[str, Any]) -> dict[str, Any]:
        scheduler_dir = self.workspace_dir / "learning"
        scheduler_dir.mkdir(parents=True, exist_ok=True)
        state_path = scheduler_dir / "meta_scheduler_state.json"
        state = {"history": []}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        decision = {
            "timestamp": iso_now(),
            "run_count": run_outcomes.get("run_count", 0),
            "allow_prompt_evolution": run_outcomes.get("run_count", 0) >= self.min_runs_before_evolution,
            "reason": "enough_runs" if run_outcomes.get("run_count", 0) >= self.min_runs_before_evolution else "insufficient_runs",
            "rollback_prompt_overrides": False,
        }
        recent_avg = float(run_outcomes.get("recent_average_quality", 0) or 0)
        historical_avg = float(run_outcomes.get("historical_average_quality", recent_avg) or recent_avg)
        if run_outcomes.get("run_count", 0) >= max(self.min_runs_before_evolution, 4) and recent_avg + 5 < historical_avg:
            decision["rollback_prompt_overrides"] = True
            decision["allow_prompt_evolution"] = False
            decision["reason"] = "quality_regression_detected"
        state.setdefault("history", []).append(decision)
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"state_path": str(state_path), **decision}


# ---------------------------------------------------------------------------
# 4. Persistence helpers
# ---------------------------------------------------------------------------


def save_learning_artifacts(
    workspace_dir: Path,
    run_outcomes: dict[str, Any],
    human_model: dict[str, Any],
    prompt_results: list[dict[str, Any]] | None = None,
    knowledge_result: dict[str, Any] | None = None,
    scheduler_result: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Save all learning artifacts to disk."""
    learning_dir = workspace_dir / "learning"
    learning_dir.mkdir(parents=True, exist_ok=True)

    # Human idea model
    model_path = learning_dir / "human_idea_model.json"
    model_path.write_text(
        json.dumps(human_model, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Run outcomes analysis
    outcomes_path = learning_dir / "run_outcomes.json"
    outcomes_path.write_text(
        json.dumps(run_outcomes, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Human-readable changelog
    changes_path = learning_dir / "learning_changes.md"
    lines = [
        f"# Learning Update — {iso_now()}",
        "",
        f"- Model version: {human_model.get('model_version', 0)}",
        f"- Runs analyzed: {run_outcomes.get('run_count', 0)}",
        f"- Average quality: {run_outcomes.get('average_quality', 0)}",
        "",
        "## Patterns Detected",
        "",
    ]
    for p in run_outcomes.get("patterns", []):
        lines.append(f"- **{p['type']}**: {p['insight']}")
        lines.append(f"  → Action: {p['action']}")
    lines.extend(["", "## Core Obsessions", ""])
    for obs in human_model.get("core_obsessions", []):
        lines.append(f"- {obs['theme']} (strength {obs['strength']}, {obs['trend']})")
    if scheduler_result:
        lines.extend(["", "## Scheduler", "", f"- allow_prompt_evolution: {scheduler_result.get('allow_prompt_evolution')}", f"- reason: {scheduler_result.get('reason')}"])
    if prompt_results:
        lines.extend(["", "## Prompt Evolution", ""])
        for item in prompt_results:
            lines.append(f"- {item['agent']}: {item.get('status', 'unknown')}")
    if knowledge_result:
        lines.extend(["", "## Domain Knowledge", "", f"- knowledge_path: {knowledge_result.get('knowledge_path')}"])

    changes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "model_path": str(model_path),
        "outcomes_path": str(outcomes_path),
        "changes_path": str(changes_path),
    }


def load_active_prompts(workspace_dir: Path) -> dict[str, str]:
    path = workspace_dir / "prompts" / "active_prompts.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    prompts = data.get("prompts", {})
    return prompts if isinstance(prompts, dict) else {}


def save_active_prompts(
    workspace_dir: Path,
    prompts: dict[str, str],
    *,
    reason: str,
) -> Path:
    prompts_dir = workspace_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    path = prompts_dir / "active_prompts.json"
    payload = {
        "updated_at": iso_now(),
        "reason": reason,
        "prompts": prompts,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_signals(workspace_dir: Path) -> list[dict[str, Any]]:
    """Load all learning signals from completed runs."""
    signals = []
    runs_dir = workspace_dir / "runs"
    if not runs_dir.exists():
        return signals
    for run_dir in sorted(runs_dir.iterdir()):
        signal_path = run_dir / "learning_signal.json"
        if signal_path.exists():
            try:
                signals.append(json.loads(signal_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
    return signals


def _load_daily_ideas_for_learning(workspace_dir: Path) -> list[dict[str, Any]]:
    """Load all daily ideas from the idea backlog for human model building."""
    backlog = workspace_dir / "ideas" / "idea_backlog.jsonl"
    if not backlog.exists():
        return []
    ideas = []
    for line in backlog.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ideas.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return ideas


def load_human_idea_model(workspace_dir: Path) -> dict[str, Any] | None:
    """Load the existing human idea model if it exists."""
    path = workspace_dir / "learning" / "human_idea_model.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return None


def run_full_learning_cycle(
    workspace_dir: Path,
    *,
    backend: AIBackend,
    current_prompts: dict[str, str],
    min_runs_before_evolution: int = 3,
) -> dict[str, Any]:
    signals = load_signals(workspace_dir)
    if not signals:
        return {"updated": False, "reason": "No signals found"}

    run_outcomes = analyze_run_outcomes(signals)
    existing_model = load_human_idea_model(workspace_dir)
    daily_ideas = _load_daily_ideas_for_learning(workspace_dir)
    human_model = build_human_idea_model(signals, daily_ideas=daily_ideas, existing_model=existing_model)

    scheduler = MetaLearningScheduler(workspace_dir, min_runs_before_evolution=min_runs_before_evolution)
    scheduler_result = scheduler.decide(run_outcomes)

    prompt_results: list[dict[str, Any]] = []
    active_prompts = load_active_prompts(workspace_dir)
    if scheduler_result.get("allow_prompt_evolution"):
        engine = PromptEvolutionEngine(backend=backend, prompts_dir=workspace_dir / "prompts", min_runs=min_runs_before_evolution)
        for agent_name, prompt in current_prompts.items():
            outcome = engine.evolve(agent_name, prompt, run_outcomes)
            prompt_results.append({"agent": agent_name, "status": "evolved" if outcome.get("evolved") else "skipped", **outcome})
            if outcome.get("evolved") and outcome.get("proposed_prompt"):
                active_prompts[agent_name] = outcome["proposed_prompt"]
    else:
        for agent_name in current_prompts:
            prompt_results.append({"agent": agent_name, "status": "skipped", "reason": scheduler_result.get("reason")})
    if scheduler_result.get("rollback_prompt_overrides"):
        active_prompts = {}
        prompt_state_path = save_active_prompts(workspace_dir, active_prompts, reason="rollback_to_baseline")
    else:
        prompt_state_path = save_active_prompts(workspace_dir, active_prompts, reason=scheduler_result.get("reason", "update"))

    knowledge_result = DomainKnowledgeExpander(workspace_dir).expand(signals)
    paths = save_learning_artifacts(
        workspace_dir,
        run_outcomes,
        human_model,
        prompt_results=prompt_results,
        knowledge_result=knowledge_result,
        scheduler_result=scheduler_result,
    )
    return {
        "updated": True,
        "run_count": run_outcomes.get("run_count", 0),
        "patterns_found": len(run_outcomes.get("patterns", [])),
        "model_version": human_model.get("model_version", 0),
        "prompt_results": prompt_results,
        "knowledge_result": knowledge_result,
        "scheduler_result": scheduler_result,
        "active_prompt_path": str(prompt_state_path),
        "paths": paths,
    }
