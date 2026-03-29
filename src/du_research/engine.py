"""Digital Unconscious Engine — the main orchestrator.

This is the top-level coordinator that connects:
  Observation → Compression → Idea Generation → Judging → Briefing

It also drives the learning cycle and triggers the research pipeline
for high-scoring ideas.
"""
from __future__ import annotations

import gc
import json
import hashlib
import logging
from datetime import datetime, time as dt_time, timezone
from pathlib import Path
import time
from typing import Any

from du_research.ai_backend import AIBackend, create_backend
from du_research.agents.briefing import BriefingAgent, save_briefing
from du_research.agents.briefing import BRIEFING_SYSTEM_PROMPT
from du_research.agents.compressor import CompressionAgent
from du_research.agents.compressor import COMPRESSOR_SYSTEM_PROMPT
from du_research.agents.analysis_coder import ANALYSIS_CODER_SYSTEM_PROMPT
from du_research.agents.idea_generator import IdeaGeneratorAgent
from du_research.agents.idea_generator import IDEA_GENERATOR_SYSTEM_PROMPT
from du_research.agents.judge import JudgeAgent
from du_research.agents.judge import JUDGE_SYSTEM_PROMPT
from du_research.agents.learning_engine import (
    analyze_run_outcomes,
    build_human_idea_model,
    load_active_prompts,
    load_human_idea_model,
    load_signals,
    run_full_learning_cycle,
    save_learning_artifacts,
)
from du_research.agents.reviewer import REVIEWER_SYSTEM_PROMPT
from du_research.agents.revision import REVISION_SYSTEM_PROMPT
from du_research.agents.writer import WRITER_SYSTEM_PROMPT
from du_research.circuit_breaker import CircuitBreaker
from du_research.config import AppConfig
from du_research.maintenance import WorkspaceMaintenance
from du_research.pipeline import ResearchPipeline
from du_research.observation import (
    BehaviorFrame,
    FileObserver,
    ScreenpipeObserver,
    deduplicate_frames,
    group_into_windows,
)
from du_research.rag import RAGStore
from du_research.task_queue import TaskQueue
from du_research.utils import iso_now

logger = logging.getLogger(__name__)


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
        if config.ai.mode == "api":
            backend_kwargs["default_model"] = config.ai.default_model
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
        )
        self.judge = JudgeAgent(
            backend=self.backend,
            model=config.ai.judge_model,
            system_prompt=prompt_overrides.get("judge"),
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
        self.research_pipeline = ResearchPipeline(config, backend=self.backend)
        self.maintenance = WorkspaceMaintenance(self.workspace, config)
        self.rag = RAGStore(self.workspace)
        self.task_queue = TaskQueue(self.workspace)

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
        date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Starting daily cycle for %s", date_str)

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
        for summary in summaries:
            ideas = self.idea_generator.generate(
                summary,
                rag_context=rag_context,
                human_idea_model=human_model,
                idea_count=self.config.idea.max_ideas_per_cycle,
            )
            all_ideas.extend(ideas)
        logger.info("Generated %d raw ideas", len(all_ideas))

        if not all_ideas:
            logger.warning("No ideas generated — LLM may be unavailable")
            self.task_queue.enqueue("idea_generation", {
                "summaries": summaries,
                "date": date_str,
            })

        # 5. Judge all ideas (LLM-only, queue on failure)
        evaluations = self.judge.evaluate(
            all_ideas,
            behaviour_summary=summaries[0] if summaries else None,
            primary_domains=self.config.idea.primary_domains,
            existing_ideas=existing_ideas,
            human_idea_model=human_model,
            focus_fields=self.config.idea.focus_fields or None,
        )
        if evaluations is None:
            logger.warning("Judge unavailable — queuing evaluation for later")
            self.task_queue.enqueue("judging", {
                "ideas": all_ideas,
                "date": date_str,
            }, priority=1)
            evaluations = []
        logger.info("Evaluated %d ideas", len(evaluations))

        # Merge evaluations back into ideas
        eval_map = {e.get("idea_id", ""): e for e in evaluations}
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
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self, log_file: str | None = None) -> list[BehaviorFrame]:
        """Collect behaviour frames from screenpipe or file fallback."""
        # Try screenpipe first
        if self.config.observation.enabled and self.screenpipe.is_available():
            logger.info("Using screenpipe for observation")
            return self.screenpipe.fetch_recent(
                minutes=self.config.observation.window_minutes * self.config.observation.lookback_multiplier,
                limit=200,
            )

        # Fallback to file
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
        backlog = self.workspace / "ideas" / "idea_backlog.jsonl"
        if not backlog.exists():
            return []
        ideas = []
        for line in backlog.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                title = obj.get("title", obj.get("idea_text", ""))
                if title:
                    ideas.append(title)
            except json.JSONDecodeError:
                continue
        return ideas[-max_ideas:]

    def _load_all_daily_ideas(self) -> list[dict[str, Any]]:
        """Load all daily ideas for the human model."""
        backlog = self.workspace / "ideas" / "idea_backlog.jsonl"
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

    def _append_unique_ideas_to_backlog(self, ideas: list[dict[str, Any]], date_str: str) -> int:
        backlog_path = self.workspace / "ideas" / "idea_backlog.jsonl"
        backlog_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._load_all_daily_ideas()
        appended = 0
        with backlog_path.open("a", encoding="utf-8") as handle:
            for idea in ideas:
                title = idea.get("title") or idea.get("idea_text") or ""
                idea_id = idea.get("id") or idea.get("idea_id")
                if self._idea_exists(existing, title, idea_id=idea_id):
                    continue
                entry = {
                    "timestamp": iso_now(),
                    "date": date_str,
                    **idea,
                }
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                existing.append(entry)
                appended += 1
        return appended

    def _idea_exists(self, existing: list[dict[str, Any]], title: str, *, idea_id: str | None = None) -> bool:
        normalized = self._idea_key(title)
        for item in existing:
            if idea_id and idea_id == (item.get("id") or item.get("idea_id")):
                return True
            other_title = item.get("title") or item.get("idea_text") or ""
            if normalized and normalized == self._idea_key(other_title):
                return True
        return False

    def _idea_key(self, title: str) -> str:
        return hashlib.sha1(title.strip().lower().encode("utf-8")).hexdigest() if title.strip() else ""

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
        cutoff = datetime.now(timezone.utc).date()
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
