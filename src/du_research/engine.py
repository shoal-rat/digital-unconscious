"""Digital Unconscious Engine — the main orchestrator.

This is the top-level coordinator that connects:
  Observation → Compression → Idea Generation → Judging → Briefing

It also drives the learning cycle and triggers the research pipeline
for high-scoring ideas.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from datetime import time as dt_time
from pathlib import Path
from typing import Any

from du_research.agents.analysis_coder import ANALYSIS_CODER_SYSTEM_PROMPT
from du_research.agents.briefing import BRIEFING_SYSTEM_PROMPT, BriefingAgent, save_briefing
from du_research.agents.compressor import COMPRESSOR_SYSTEM_PROMPT, CompressionAgent
from du_research.agents.idea_generator import IDEA_GENERATOR_SYSTEM_PROMPT, IdeaGeneratorAgent
from du_research.agents.judge import JUDGE_SYSTEM_PROMPT, JudgeAgent
from du_research.agents.learning_engine import (
    load_active_prompts,
    load_human_idea_model,
    run_full_learning_cycle,
)
from du_research.agents.reviewer import REVIEWER_SYSTEM_PROMPT
from du_research.agents.revision import REVISION_SYSTEM_PROMPT
from du_research.agents.writer import WRITER_SYSTEM_PROMPT
from du_research.ai_backend import AIBackend, create_backend
from du_research.backlog import IdeaBacklog
from du_research.circuit_breaker import CircuitBreaker
from du_research.config import AppConfig
from du_research.maintenance import WorkspaceMaintenance
from du_research.observation import (
    BehaviorFrame,
    FileObserver,
    ScreenpipeObserver,
    VisionObserver,
    deduplicate_frames,
    group_into_windows,
)
from du_research.pipeline import ResearchPipeline
from du_research.rag import RAGStore
from du_research.task_queue import TaskQueue
from du_research.utils import iso_now

logger = logging.getLogger(__name__)


def _reconcile_evaluation_ids(
    ideas: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> None:
    """Map non-stable judge IDs to stable idea IDs without collapsing aliases.

    Generators run once per observation window and commonly restart numbering at
    ``idea_001``.  Judge prompts omit those response-local IDs, but a model may
    still emit generic numbered IDs from its output template.  Resolve those by
    global input position first and retain an ordered alias queue as a fallback.
    """
    stable_ids = [str(idea.get("idea_id") or idea.get("id") or "") for idea in ideas]
    stable_set = {idea_id for idea_id in stable_ids if idea_id}
    alias_targets: dict[str, list[str]] = {}
    for idea, stable_id in zip(ideas, stable_ids, strict=True):
        alias = str(idea.get("source_model_id") or "")
        if alias and stable_id:
            alias_targets.setdefault(alias, []).append(stable_id)

    claimed: set[str] = set()
    for evaluation in evaluations:
        reported_id = str(evaluation.get("idea_id") or "")
        if reported_id in stable_set:
            claimed.add(reported_id)
            continue

        target = None
        prefix, separator, ordinal = reported_id.rpartition("_")
        if separator and prefix == "idea" and ordinal.isdigit():
            index = int(ordinal) - 1
            if 0 <= index < len(stable_ids) and stable_ids[index] not in claimed:
                target = stable_ids[index]

        if target is None:
            target = next(
                (candidate for candidate in alias_targets.get(reported_id, []) if candidate not in claimed),
                None,
            )
        if target:
            evaluation["source_model_id"] = reported_id
            evaluation["idea_id"] = target
            claimed.add(target)


class DigitalUnconsciousEngine:
    """Main orchestrator for the Digital Unconscious system.

    Usage::

        engine = DigitalUnconsciousEngine(config)
        result = engine.run_daily_cycle()
        # result contains: summaries, ideas, evaluations, briefing_path
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.workspace = Path(config.pipeline.workspace_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Create AI backend
        backend_kwargs: dict[str, Any] = {}
        if config.ai.api_key:
            backend_kwargs["api_key"] = config.ai.api_key
        if config.ai.openai_api_key:
            backend_kwargs["openai_api_key"] = config.ai.openai_api_key
        if config.ai.kimi_api_key:
            backend_kwargs["kimi_api_key"] = config.ai.kimi_api_key
        if config.ai.deepseek_api_key:
            backend_kwargs["deepseek_api_key"] = config.ai.deepseek_api_key
        if config.ai.glm_api_key:
            backend_kwargs["glm_api_key"] = config.ai.glm_api_key
        backend_kwargs["default_model"] = config.ai.default_model
        backend_kwargs["openai_default_model"] = config.ai.openai_default_model
        backend_kwargs["kimi_default_model"] = config.ai.kimi_default_model
        backend_kwargs["deepseek_default_model"] = config.ai.deepseek_default_model
        backend_kwargs["glm_default_model"] = config.ai.glm_default_model
        backend_kwargs["enable_fallback"] = config.ai.fallback
        backend_kwargs["fallback_order"] = config.ai.fallback_order
        raw_backend: AIBackend = create_backend(config.ai.mode, **backend_kwargs)

        # Wrap in circuit breaker
        self.backend = CircuitBreaker(
            backend=raw_backend,
            max_retries=config.circuit_breaker.max_retries,
            initial_wait=config.circuit_breaker.initial_wait,
            failure_threshold=config.circuit_breaker.failure_threshold,
            recovery_timeout=config.circuit_breaker.recovery_timeout,
        )

        prompt_overrides = load_active_prompts(self.workspace)

        # Initialise agents
        self.compressor = CompressionAgent(
            backend=self.backend,
            model=config.ai.compressor_model,
            system_prompt=prompt_overrides.get("compressor"),
        )
        self.idea_generator = IdeaGeneratorAgent(
            backend=self.backend,
            model=config.ai.creative_model,
            system_prompt=prompt_overrides.get("idea_generator"),
            primary_domains=config.idea.primary_domains,
            secondary_domains=config.idea.secondary_domains,
            focus_fields=config.idea.focus_fields,
            think=config.ai.think_idea_budget,
            web_search=config.idea.web_search,
        )
        self.judge = JudgeAgent(
            backend=self.backend,
            model=config.ai.judge_model,
            system_prompt=prompt_overrides.get("judge"),
            think=config.ai.think_judge_budget,
            web_search=config.idea.web_search,
        )
        self.briefing_agent = BriefingAgent(
            backend=self.backend,
            model=config.ai.briefing_model,
            system_prompt=prompt_overrides.get("briefing"),
        )

        # Observers
        _blacklist = {a.lower() for a in config.observation.blacklist_apps}
        self.screenpipe = ScreenpipeObserver(
            base_url=config.observation.screenpipe_url,
            timeout=config.pipeline.network_timeout_seconds,
            blacklist_apps=_blacklist,
        )
        self.file_observer = FileObserver(blacklist_apps=_blacklist)
        # Codex accepts image attachments and Claude Code can inspect an isolated
        # temporary image with its Read tool, so vision no longer requires an API key.
        self.vision = VisionObserver(
            backend=self.backend,
            model=config.observation.vision_model,
            max_dimension=config.observation.vision_max_dimension,
        )
        self.research_pipeline = ResearchPipeline(config, backend=self.backend)
        self.maintenance = WorkspaceMaintenance(self.workspace, config)
        # Use file-based RAG in temp directories (avoids ChromaDB locking issues)
        import tempfile as _tf
        _in_temp = str(self.workspace).startswith(_tf.gettempdir())
        self.rag = RAGStore(
            self.workspace,
            force_file_mode=_in_temp,
            max_documents=config.retention.rag_max_documents,
        )
        self.task_queue = TaskQueue(self.workspace)
        self.idea_backlog = IdeaBacklog(self.workspace, config.retention.idea_backlog_max)

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_daily_cycle(
        self,
        *,
        log_file: str | None = None,
        date_str: str | None = None,
        frames_override: list[BehaviorFrame] | None = None,
    ) -> dict[str, Any]:
        """Execute the full daily cycle: observe → compress → generate → judge → brief.

        Parameters
        ----------
        log_file :
            Optional path to a manual daily-log file (fallback when screenpipe
            is not available).
        date_str :
            Override the date string for the briefing header.
        """
        date_str = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
        logger.info("Starting daily cycle for %s", date_str)

        # Reset usage counters so usage.json reflects only this cycle.
        if hasattr(self.backend, "reset_usage"):
            self.backend.reset_usage()

        # 1. Observe
        frames = frames_override if frames_override is not None else self._observe(log_file)
        logger.info("Observed %d behaviour frames", len(frames))

        # 2. Compress into windows (LLM-only, queue on failure)
        frames = deduplicate_frames(frames)
        windows = group_into_windows(frames, self.config.observation.window_minutes)
        summaries = []
        queued_compressions = 0
        for window in windows:
            summary = self.compressor.compress(window)
            if summary is not None:
                summaries.append(summary)
            else:
                # Queue for later processing
                self.task_queue.enqueue("compression", {
                    "frames": [f.to_dict() for f in window],
                    "date": date_str,
                })
                queued_compressions += 1
        if queued_compressions:
            logger.info("Queued %d compression tasks for later (LLM unavailable)", queued_compressions)
        if not summaries:
            logger.warning("No summaries produced — all compressions queued")
            return self._queued_result(date_str, frames, queued_compressions)
        logger.info("Compressed into %d window summaries", len(summaries))

        # 3. Load context
        human_model = load_human_idea_model(self.workspace)
        existing_ideas = self._load_recent_ideas()

        # 4. Generate ideas from each summary (with RAG context if available)
        rag_context = self._load_rag_context(summaries)
        all_ideas: list[dict[str, Any]] = []
        seen_idea_keys: set[str] = set()
        total_budget = max(1, self.config.idea.max_ideas_per_cycle)
        for summary_index, summary in enumerate(summaries):
            remaining_budget = total_budget - len(all_ideas)
            if remaining_budget <= 0:
                break
            remaining_windows = max(1, len(summaries) - summary_index)
            window_budget = max(1, (remaining_budget + remaining_windows - 1) // remaining_windows)
            ideas = self.idea_generator.generate(
                summary,
                rag_context=rag_context,
                human_idea_model=human_model,
                idea_count=window_budget,
            )
            for raw_idea in ideas:
                idea = self.idea_backlog.normalize(raw_idea)
                key = self.idea_backlog.key(self.idea_backlog.title_of(idea))
                if not key or key in seen_idea_keys:
                    continue
                seen_idea_keys.add(key)
                all_ideas.append(idea)
                if len(all_ideas) >= total_budget:
                    break
        logger.info("Generated %d raw ideas", len(all_ideas))

        if not all_ideas:
            logger.warning("No ideas generated — LLM may be unavailable")
            self.task_queue.enqueue("idea_generation", {
                "summaries": summaries,
                "date": date_str,
            })

        # 5. Judge all ideas (LLM-only, queue on failure)
        # The stable IDs are the only identities exposed to the judge. Keeping
        # repeated response-local IDs in this prompt makes ``idea_001``
        # inherently ambiguous when several observation windows are combined.
        judge_ideas = [
            {key: value for key, value in idea.items() if key != "source_model_id"}
            for idea in all_ideas
        ]
        evaluations = self.judge.evaluate(
            judge_ideas,
            behaviour_summary=summaries[0] if summaries else None,
            primary_domains=self.config.idea.primary_domains,
            existing_ideas=existing_ideas,
            human_idea_model=human_model,
            focus_fields=self.config.idea.focus_fields or None,
        )
        if evaluations is None:
            logger.warning("Judge unavailable — queuing evaluation for later")
            self.task_queue.enqueue("judging", {
                "ideas": judge_ideas,
                "date": date_str,
            }, priority=1)
            evaluations = []
        logger.info("Evaluated %d ideas", len(evaluations))

        # Merge evaluations back into ideas
        _reconcile_evaluation_ids(all_ideas, evaluations)
        eval_map: dict[str, dict[str, Any]] = {}
        for evaluation in evaluations:
            eval_map.setdefault(str(evaluation.get("idea_id") or ""), evaluation)
        scored_ideas = []
        for idea in all_ideas:
            idea_id = idea.get("id", idea.get("idea_id", ""))
            ev = eval_map.get(idea_id, {})
            scored_ideas.append({**idea, **ev})
        scored_ideas.sort(key=lambda x: x.get("total_score", 0), reverse=True)

        # Filter by threshold
        included = [i for i in scored_ideas if i.get("total_score", 0) >= self.config.idea.include_threshold]
        held = [
            i for i in scored_ideas
            if self.config.idea.hold_threshold <= i.get("total_score", 0) < self.config.idea.include_threshold
        ]

        # 6. Generate briefing (LLM-only, queue on failure)
        briefing_text = self.briefing_agent.generate(
            summaries,
            included[:self.config.idea.max_briefing_ideas],
            idea_evaluations=evaluations,
            human_idea_model=human_model,
            date_str=date_str,
        )
        if briefing_text is None:
            logger.warning("Briefing generation unavailable — queuing for later")
            self.task_queue.enqueue("briefing", {
                "summaries": summaries,
                "ideas": included[:self.config.idea.max_briefing_ideas],
                "date": date_str,
            }, priority=2)
            briefing_text = f"# Briefing Pending — {date_str}\n\nThe AI was unavailable. This briefing will be generated when the LLM is back online.\n\nRun `du drain` to retry pending tasks.\n"

        # 7. Save artifacts
        output_dir = self.workspace / "daily" / f"cycle_{date_str}"
        output_dir.mkdir(parents=True, exist_ok=True)

        briefing_path = save_briefing(briefing_text, output_dir, date_str)

        # Save raw data
        (output_dir / "summaries.json").write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "ideas_all.json").write_text(
            json.dumps(scored_ideas, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "ideas_included.json").write_text(
            json.dumps(included, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "evaluations.json").write_text(
            json.dumps(evaluations, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_dir / "frames.json").write_text(
            json.dumps([frame.to_dict() for frame in frames], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 8. Append to idea backlog
        backlog_append_count = self._append_unique_ideas_to_backlog(included, date_str)

        # 9. Auto-promote top included ideas into the research pipeline
        research_runs = []
        if self.config.idea.auto_research_enabled:
            top_ideas = included[: self.config.idea.auto_research_top_k]
            for idea in top_ideas:
                title = idea.get("title") or idea.get("idea_text")
                idea_id = idea.get("id") or idea.get("idea_id")
                if not title:
                    continue
                duplicate = None
                if self.config.idea.auto_research_dedupe_enabled and hasattr(self.research_pipeline, "find_recent_duplicate_research"):
                    duplicate = self.research_pipeline.find_recent_duplicate_research(
                        title,
                        idea_id=idea_id,
                        similarity_threshold=self.config.idea.auto_research_similarity_threshold,
                        cooldown_days=self.config.idea.auto_research_cooldown_days,
                    )
                if duplicate:
                    research_runs.append(
                        {
                            "idea_id": idea_id,
                            "title": title,
                            "skipped": True,
                            "reason": "duplicate_research",
                            "existing_run_id": duplicate["run_id"],
                            "existing_status": duplicate.get("status"),
                            "similarity": duplicate.get("similarity"),
                        }
                    )
                    continue
                run_id = f"auto_{date_str.replace('-', '')}_{idea_id or title[:24].lower().replace(' ', '_')}"
                try:
                    result = self.research_pipeline.run(
                        idea_text=title,
                        idea_id=idea_id,
                        research_seed=idea,
                        run_id=run_id,
                        dry_run=False,
                    )
                    research_runs.append(
                        {
                            "idea_id": idea_id,
                            "title": title,
                            "run_id": result["run_id"],
                            "run_dir": result["run_dir"],
                            "quality_score": result["review"]["overall_score"],
                        }
                    )
                except Exception as exc:
                    research_runs.append(
                        {
                            "idea_id": idea_id,
                            "title": title,
                            "error": str(exc),
                        }
                    )

        (output_dir / "research_runs.json").write_text(
            json.dumps(research_runs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 10. Ingest papers from completed research runs into RAG knowledge base
        rag_ingested = 0
        for rr in research_runs:
            run_dir = rr.get("run_dir")
            if run_dir:
                rag_ingested += self.rag.add_papers_from_run(Path(run_dir))
        if rag_ingested:
            logger.info("Ingested %d papers into RAG knowledge base", rag_ingested)

        # Also ingest user knowledge docs if present
        user_knowledge_dir = self.workspace / "knowledge" / "documents"
        if user_knowledge_dir.exists():
            self.rag.add_knowledge_files(user_knowledge_dir)

        # Record per-cycle model usage (every backend call flows through the breaker).
        usage = self.backend.usage if hasattr(self.backend, "usage") else {}
        (output_dir / "usage.json").write_text(
            json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if usage.get("total_tokens") or usage.get("cost_usd"):
            try:
                cost = usage.get("cost_usd", 0.0)
                cost_text = f" · ${cost:.4f}" if cost else ""
                with briefing_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"\n\n---\n\n_Model usage today: {usage.get('calls', 0)} calls, "
                        f"{usage.get('total_tokens', 0):,} tokens{cost_text}._\n"
                    )
            except OSError:
                logger.debug("Could not append usage footer to briefing")

        logger.info(
            "Daily cycle complete: %d included, %d held, briefing at %s",
            len(included), len(held), briefing_path,
        )

        return {
            "date": date_str,
            "frames_observed": len(frames),
            "windows_compressed": len(summaries),
            "ideas_generated": len(all_ideas),
            "ideas_included": len(included),
            "ideas_held": len(held),
            "ideas_added_to_backlog": backlog_append_count,
            "research_runs_started": len([item for item in research_runs if item.get("run_id")]),
            "briefing_path": str(briefing_path),
            "output_dir": str(output_dir),
            "research_runs": research_runs,
            "circuit_breaker_stats": self.backend.stats,
            "usage": usage,
        }

    def ingest_observation_snapshot(
        self,
        *,
        log_file: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or datetime.now().astimezone()
        frames = deduplicate_frames(self._observe(log_file))
        state = self._load_service_state()
        known_hashes = set(state.get("recent_frame_hashes", []))
        new_frames: list[BehaviorFrame] = []
        for frame in frames:
            fingerprint = self._frame_fingerprint(frame)
            if fingerprint in known_hashes:
                continue
            known_hashes.add(fingerprint)
            new_frames.append(frame)
        journal_path = self._observation_journal_path(now.date().isoformat())
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as handle:
            for frame in new_frames:
                handle.write(json.dumps(frame.to_dict(), ensure_ascii=False) + "\n")
        state["recent_frame_hashes"] = list(known_hashes)[-self.config.observation.recent_frame_hash_limit :]
        state["last_ingest_at"] = iso_now()
        state.setdefault("observation_days", {})
        state["observation_days"][now.date().isoformat()] = {
            "journal_path": str(journal_path),
            "frame_count": self._count_jsonl_lines(journal_path),
        }
        self._prune_service_state(state)
        self._save_service_state(state)
        return {
            "date": now.date().isoformat(),
            "observed_frames": len(frames),
            "new_frames": len(new_frames),
            "journal_path": str(journal_path),
        }

    def run_service_once(
        self,
        *,
        log_file: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        now = now or datetime.now().astimezone()
        ingest_result = self.ingest_observation_snapshot(log_file=log_file, now=now)
        state = self._load_service_state()
        delivery_date = now.date().isoformat()
        briefing_ready = now.time() >= self._briefing_time()
        already_delivered = state.get("last_delivery_date") == delivery_date
        daily_result = None
        if briefing_ready and not already_delivered:
            frames = self._load_observation_frames(delivery_date)
            if frames:
                daily_result = self.run_daily_cycle(
                    date_str=delivery_date,
                    frames_override=frames,
                )
                state["last_delivery_date"] = delivery_date
                state["last_delivery_output_dir"] = daily_result.get("output_dir")
                self._save_service_state(state)
        return {
            "timestamp": now.isoformat(),
            "ingest": ingest_result,
            "briefing_generated": daily_result is not None,
            "daily_result": daily_result,
        }

    def run_observation_service(
        self,
        *,
        interval_minutes: int | None = None,
        iterations: int | None = None,
        log_file: str | None = None,
    ) -> dict[str, Any]:
        interval = max(1, interval_minutes or self.config.observation.service_interval_minutes)
        completed = 0
        runs = []
        while iterations is None or completed < iterations:
            try:
                result = self.run_service_once(log_file=log_file)
                entry = {
                    "timestamp": result["timestamp"],
                    "new_frames": result["ingest"]["new_frames"],
                    "briefing_generated": result["briefing_generated"],
                    "output_dir": result.get("daily_result", {}).get("output_dir") if result.get("daily_result") else None,
                }
            except Exception as exc:
                logger.exception("Service cycle failed")
                entry = {
                    "timestamp": iso_now(),
                    "error": str(exc),
                    "briefing_generated": False,
                    "output_dir": None,
                }
            runs.append(entry)
            runs = runs[-self.config.service.run_history_limit :]
            completed += 1
            state = self._load_service_state()
            state["service_cycle_count"] = int(state.get("service_cycle_count", 0)) + 1
            state["last_cycle_at"] = entry["timestamp"]
            if entry.get("error"):
                state["last_error"] = entry["error"]
            else:
                state["last_error"] = None
            state["recent_runs"] = runs[-self.config.service.run_history_limit :]
            maintenance_result = None
            if state["service_cycle_count"] % max(1, self.config.service.maintenance_every_cycles) == 0:
                maintenance_result = self.maintenance.run()
                state["last_maintenance"] = maintenance_result
            if state["service_cycle_count"] % max(1, self.config.service.gc_every_cycles) == 0:
                gc.collect()
                state["last_gc_at"] = iso_now()
            self._save_service_state(state)
            self._write_service_status(
                {
                    "running": True,
                    "interval_minutes": interval,
                    "completed_cycles": completed,
                    "last_entry": entry,
                    "recent_runs": runs[-min(20, self.config.service.run_history_limit):],
                    "last_maintenance": maintenance_result or state.get("last_maintenance"),
                    "last_gc_at": state.get("last_gc_at"),
                }
            )
            if iterations is not None and completed >= iterations:
                break
            time.sleep(interval * 60)
        return {
            "completed_cycles": completed,
            "interval_minutes": interval,
            "runs": runs[-min(len(runs), self.config.service.run_history_limit):],
        }

    # ------------------------------------------------------------------
    # Task queue
    # ------------------------------------------------------------------

    def drain_queue(self) -> dict[str, Any]:
        """Process all pending tasks in the queue (retry failed LLM calls)."""
        def _handler(task: dict[str, Any]) -> dict[str, Any] | None:
            task_type = task.get("type")
            payload = task.get("payload", {})

            if task_type == "compression":
                frames = [BehaviorFrame(**f) for f in payload.get("frames", [])]
                result = self.compressor.compress(frames)
                return {"summary": result} if result is not None else None

            if task_type == "judging":
                ideas = payload.get("ideas", [])
                result = self.judge.evaluate(ideas)
                return {"evaluations": result} if result is not None else None

            if task_type == "briefing":
                summaries = payload.get("summaries", [])
                ideas = payload.get("ideas", [])
                date = payload.get("date", "")
                result = self.briefing_agent.generate(summaries, ideas, date_str=date)
                if result is not None:
                    output_dir = self.workspace / "daily" / f"cycle_{date}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    save_briefing(result, output_dir, date)
                    return {"briefing_path": str(output_dir / f"briefing_{date}.md")}
                return None

            if task_type == "idea_generation":
                summaries = payload.get("summaries", [])
                all_ideas = []
                for summary in summaries:
                    ideas = self.idea_generator.generate(summary)
                    all_ideas.extend(ideas)
                return {"ideas": all_ideas} if all_ideas else None

            return {"skipped": True, "reason": f"unknown task type: {task_type}"}

        return self.task_queue.drain(_handler)

    def _queued_result(
        self,
        date_str: str,
        frames: list[BehaviorFrame],
        queued_count: int,
    ) -> dict[str, Any]:
        """Return a result dict when the entire cycle was queued."""
        return {
            "date": date_str,
            "frames_observed": len(frames),
            "windows_compressed": 0,
            "ideas_generated": 0,
            "ideas_included": 0,
            "ideas_held": 0,
            "ideas_added_to_backlog": 0,
            "research_runs_started": 0,
            "briefing_path": "",
            "output_dir": "",
            "research_runs": [],
            "queued_tasks": queued_count,
            "circuit_breaker_stats": self.backend.stats,
            "note": "LLM unavailable — tasks queued. Run `du drain` to retry.",
        }

    # ------------------------------------------------------------------
    # Learning cycle
    # ------------------------------------------------------------------

    def run_learning_cycle(self) -> dict[str, Any]:
        """Run the learning engine: analyze outcomes, update model."""
        current_prompts = {
            "compressor": self.compressor.system_prompt or COMPRESSOR_SYSTEM_PROMPT,
            "idea_generator": self.idea_generator.system_prompt or IDEA_GENERATOR_SYSTEM_PROMPT,
            "judge": self.judge.system_prompt or JUDGE_SYSTEM_PROMPT,
            "briefing": self.briefing_agent.system_prompt or BRIEFING_SYSTEM_PROMPT,
            "writer": self.research_pipeline.writer_agent.system_prompt if self.research_pipeline.writer_agent else WRITER_SYSTEM_PROMPT,
            "reviewer": self.research_pipeline.reviewer_agent.system_prompt if self.research_pipeline.reviewer_agent else REVIEWER_SYSTEM_PROMPT,
            "revision": self.research_pipeline.revision_agent.system_prompt if self.research_pipeline.revision_agent else REVISION_SYSTEM_PROMPT,
            "analysis_coder": self.research_pipeline.analysis_coder.system_prompt if self.research_pipeline.analysis_coder else ANALYSIS_CODER_SYSTEM_PROMPT,
        }
        return run_full_learning_cycle(
            self.workspace,
            backend=self.backend,
            current_prompts=current_prompts,
            min_runs_before_evolution=self.config.learning.min_runs_before_evolution,
            retention=self.config.retention,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self, log_file: str | None = None) -> list[BehaviorFrame]:
        """Collect behaviour frames from the configured source.

        ``source = "auto"`` tries screenpipe, then vision (screenshot -> model),
        then a manual log file. An explicit source uses only that path.
        """
        source = (self.config.observation.source or "auto").lower()
        auto = source == "auto"

        if self.config.observation.enabled and (auto or source == "screenpipe") and self.screenpipe.is_available():
            logger.info("Using screenpipe for observation")
            return self.screenpipe.fetch_recent(
                minutes=self.config.observation.window_minutes * self.config.observation.lookback_multiplier,
                limit=200,
            )

        if self.config.observation.enabled and (auto or source == "vision") and self.vision is not None and self.vision.is_available():
            logger.info("Using vision observer (screenshot -> multimodal model)")
            frames = self.vision.capture()
            if frames:
                return frames

        fallback = log_file or self.config.observation.fallback_log_path
        if fallback:
            path = Path(fallback).expanduser().resolve()
            logger.info("Using file observer: %s", path)
            return self.file_observer.read(path)

        logger.warning("No observation source available")
        return []

    def _load_rag_context(self, summaries: list[dict[str, Any]]) -> str | None:
        """Build RAG context via ChromaDB semantic search (or file fallback).

        Extracts key topics from today's behaviour summaries and queries the
        knowledge base for the most relevant past research and user documents.
        """
        if self.rag.count() == 0:
            return None

        # Build query from today's dominant topics
        query_parts: list[str] = []
        for summary in summaries[:3]:
            topics = summary.get("dominant_topics", [])
            if isinstance(topics, list):
                query_parts.extend(str(t) for t in topics)
            query_parts.extend(str(s) for s in summary.get("search_queries", []))
            hints = summary.get("cross_domain_hints", [])
            if isinstance(hints, list):
                query_parts.extend(str(h) for h in hints)

        query_text = " ".join(query_parts) if query_parts else " ".join(
            str(summary.get("dominant_topics", "")) for summary in summaries[:2]
        )
        if not query_text.strip():
            return None

        return self.rag.query_as_context(query_text, n_results=8)

    def _load_recent_ideas(self, max_ideas: int = 50) -> list[str]:
        """Load recent idea titles from the backlog for novelty comparison."""
        return self.idea_backlog.recent_titles(max_ideas)

    def _load_all_daily_ideas(self) -> list[dict[str, Any]]:
        """Load all daily ideas for the human model."""
        return self.idea_backlog.load()

    def _append_unique_ideas_to_backlog(self, ideas: list[dict[str, Any]], date_str: str) -> int:
        return self.idea_backlog.append_unique(ideas, date=date_str, origin="daily_scan")

    def _cap_backlog(self, path: Path, max_entries: int) -> None:
        """Keep only the newest ``max_entries`` backlog lines."""
        if not max_entries or max_entries <= 0:
            return
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) <= max_entries:
            return
        path.write_text("\n".join(lines[-max_entries:]) + "\n", encoding="utf-8")

    def _idea_exists(self, existing: list[dict[str, Any]], title: str, *, idea_id: str | None = None) -> bool:
        return self.idea_backlog.contains(title, idea_id=idea_id, ideas=existing)

    def _idea_key(self, title: str) -> str:
        return self.idea_backlog.key(title)

    def _service_state_path(self) -> Path:
        return self.workspace / "service" / "service_state.json"

    def _load_service_state(self) -> dict[str, Any]:
        path = self._service_state_path()
        if not path.exists():
            return {"recent_frame_hashes": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"recent_frame_hashes": []}

    def _save_service_state(self, state: dict[str, Any]) -> None:
        path = self._service_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def _frame_fingerprint(self, frame: BehaviorFrame) -> str:
        timestamp_component = frame.timestamp[:19]
        if frame.frame_type == "log" or frame.app_name == "text_log":
            timestamp_component = ""
        content = "|".join(
            [
                timestamp_component,
                frame.app_name.lower(),
                frame.window_title.strip().lower(),
                frame.text_content.strip().lower()[:500],
                frame.url or "",
            ]
        )
        return hashlib.sha1(content.encode("utf-8")).hexdigest()

    def _observation_journal_path(self, date_str: str) -> Path:
        return self.workspace / "observation" / f"{date_str}_frames.jsonl"

    def _load_observation_frames(self, date_str: str) -> list[BehaviorFrame]:
        path = self._observation_journal_path(date_str)
        if not path.exists():
            return []
        frames: list[BehaviorFrame] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            frames.append(
                BehaviorFrame(
                    timestamp=payload.get("timestamp", iso_now()),
                    app_name=payload.get("app_name", "unknown"),
                    window_title=payload.get("window_title", ""),
                    text_content=payload.get("text_content", ""),
                    dwell_seconds=float(payload.get("dwell_seconds", 0.0)),
                    frame_type=payload.get("frame_type", "screen"),
                    url=payload.get("url"),
                )
            )
        return frames

    def _briefing_time(self) -> dt_time:
        raw = self.config.daily.briefing_time.strip()
        try:
            hour_text, minute_text = raw.split(":", 1)
            return dt_time(hour=int(hour_text), minute=int(minute_text))
        except Exception:
            return dt_time(hour=22, minute=0)

    def _count_jsonl_lines(self, path: Path) -> int:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

    def _write_service_status(self, payload: dict[str, Any]) -> None:
        status_path = self._workspace_path(self.config.service.status_path)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _prune_service_state(self, state: dict[str, Any]) -> None:
        observation_days = state.get("observation_days", {})
        if not isinstance(observation_days, dict):
            state["observation_days"] = {}
            return
        cutoff = datetime.now(UTC).date()
        keep_days = max(0, self.config.retention.observation_days)
        kept = {}
        for key, value in observation_days.items():
            try:
                day = datetime.fromisoformat(key).date()
            except ValueError:
                continue
            if (cutoff - day).days <= keep_days:
                kept[key] = value
        state["observation_days"] = kept

    def _workspace_path(self, raw: str) -> Path:
        path = Path(raw)
        if path.is_absolute():
            return path
        parts = list(path.parts)
        if parts and parts[0].lower() == "workspace":
            parts = parts[1:]
        return self.workspace.joinpath(*parts)
