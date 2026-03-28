from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from du_research.agents.analysis_coder import AnalysisCoderAgent
from du_research.agents.learning_engine import load_active_prompts, run_full_learning_cycle
from du_research.agents.reviewer import ReviewerAgent
from du_research.agents.revision import RevisionAgent
from du_research.agents.writer import WriterAgent
from du_research.ai_backend import AIBackend
from du_research.automation import BrowserAutomationRunner
from du_research.config import AppConfig
from du_research.credential_broker import CredentialBroker
from du_research.integrations.claude_code import export_computer_use_task
from du_research.models import RunManifest
from du_research.storage import RunStorage, load_manifest
from du_research.stages import analysis, daily_capture, data_sources, drafting, feasibility, learning, literature, review
from du_research.submission import SubmissionManager
from du_research.utils import iso_now, overlap_score, slugify, timestamp_for_id


class ResearchPipeline:
    def __init__(
        self,
        config: AppConfig,
        literature_providers: list[literature.LiteratureProvider] | None = None,
        dataset_providers: list[data_sources.DatasetProvider] | None = None,
        backend: AIBackend | None = None,
        credential_broker: CredentialBroker | None = None,
        automation_runner: BrowserAutomationRunner | None = None,
        submission_manager: SubmissionManager | None = None,
    ):
        self.config = config
        self.workspace_dir = Path(config.pipeline.workspace_dir).resolve()
        self.literature_providers = literature_providers
        self.dataset_providers = dataset_providers
        self.backend = backend
        prompt_overrides = load_active_prompts(self.workspace_dir)
        def _workspace_path(raw: str) -> Path:
            path = Path(raw)
            if path.is_absolute():
                return path
            parts = list(path.parts)
            if parts and parts[0].lower() == "workspace":
                parts = parts[1:]
            return self.workspace_dir.joinpath(*parts)
        self.writer_agent = WriterAgent(backend, model=config.ai.writer_model, system_prompt=prompt_overrides.get("writer")) if backend else None
        self.reviewer_agent = ReviewerAgent(backend, model=config.ai.reviewer_model, system_prompt=prompt_overrides.get("reviewer")) if backend else None
        self.revision_agent = RevisionAgent(backend, model=config.ai.revision_model, system_prompt=prompt_overrides.get("revision")) if backend else None
        self.analysis_coder = AnalysisCoderAgent(backend, model=config.ai.analysis_model, system_prompt=prompt_overrides.get("analysis_coder")) if backend else None
        self.credential_broker = credential_broker or CredentialBroker(
            vault_path=_workspace_path(config.credentials.vault_path).resolve(),
            key_path=_workspace_path(config.credentials.key_path).resolve(),
        )
        self.automation_runner = automation_runner or BrowserAutomationRunner(
            runner=config.automation.runner,
            browser=config.automation.browser,
            download_dir=_workspace_path(config.automation.download_dir).resolve(),
            screenshot_dir=_workspace_path(config.automation.screenshot_dir).resolve(),
            headless=config.automation.headless,
            timeout_seconds=config.automation.timeout_seconds,
        )
        self.submission_manager = submission_manager or SubmissionManager(
            approvals_dir=_workspace_path(config.submission.approvals_path).resolve(),
            pending_timeout_hours=config.submission.pending_timeout_hours,
        )

    def run(
        self,
        idea_text: str | None = None,
        data_file: str | None = None,
        run_id: str | None = None,
        idea_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        resolved_idea_text, resolved_idea_id = self.resolve_idea(idea_text=idea_text, idea_id=idea_id)
        if not resolved_idea_text:
            raise ValueError("Either idea_text or a valid idea_id is required.")
        idea_slug = slugify(resolved_idea_text)
        resolved_run_id = run_id or f"run_{timestamp_for_id()}_{idea_slug}"
        storage = RunStorage(self.workspace_dir, resolved_run_id)
        manifest = RunManifest(
            run_id=resolved_run_id,
            idea_text=resolved_idea_text,
            idea_slug=idea_slug,
            idea_id=resolved_idea_id,
            created_at=iso_now(),
            data_file=str(Path(data_file).resolve()) if data_file else None,
        )
        storage.save_manifest(manifest)
        storage.append_trace("pipeline", "started", {"idea_text": resolved_idea_text, "idea_id": resolved_idea_id, "dry_run": dry_run})

        try:
            literature_dir = storage.stage_dir(1, "literature")
            papers, literature_payload, stage = literature.run_stage(
                idea_text=resolved_idea_text,
                output_dir=literature_dir,
                max_results_per_source=self.config.literature.max_results_per_source,
                core_papers=self.config.literature.core_papers,
                timeout=self.config.pipeline.network_timeout_seconds,
                download_pdfs=self.config.literature.download_pdfs,
                max_pdf_downloads=self.config.literature.max_pdf_downloads,
                dry_run=dry_run,
                providers=self.literature_providers,
            )
            manifest.stages.append(stage)
            storage.append_trace("literature", "completed", stage.metrics)
            storage.save_manifest(manifest)

            feasibility_dir = storage.stage_dir(2, "feasibility")
            feasibility_payload, stage = feasibility.run_stage(
                idea_text=resolved_idea_text,
                papers=papers,
                output_dir=feasibility_dir,
            )
            manifest.stages.append(stage)
            storage.append_trace("feasibility", "completed", stage.metrics)
            storage.save_manifest(manifest)

            data_dir = storage.stage_dir(3, "data_sources")
            datasets, dataset_payload, stage = data_sources.run_stage(
                idea_text=resolved_idea_text,
                output_dir=data_dir,
                max_results_per_source=self.config.datasets.max_results_per_source,
                timeout=self.config.pipeline.network_timeout_seconds,
                dry_run=dry_run,
                providers=self.dataset_providers,
            )
            manifest.stages.append(stage)
            storage.append_trace("data_sources", "completed", stage.metrics)
            storage.save_manifest(manifest)

            automation_task_path = export_computer_use_task(
                storage.run_dir,
                credential_lookup=self.credential_broker.get_credential,
                institutional_proxy_url=self.config.automation.institutional_proxy_url,
            )
            automation_result: dict[str, Any] = {
                "task_path": str(automation_task_path),
                "executed": False,
            }
            storage.append_trace("automation", "task_pack_created", {"path": str(automation_task_path)})
            if self.config.automation.enabled and self.config.automation.auto_execute and not dry_run:
                automation_result = {
                    "task_path": str(automation_task_path),
                    "executed": True,
                    **self.run_computer_task(str(automation_task_path)),
                }
                (storage.run_dir / "automation_result.json").write_text(
                    json.dumps(automation_result, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                storage.append_trace(
                    "automation",
                    "executed",
                    {"ok": automation_result.get("ok", False), "status": automation_result.get("status", "completed")},
                )

            analysis_dir = storage.stage_dir(4, "analysis")
            analysis_payload, stage = analysis.run_stage(
                idea_text=resolved_idea_text,
                data_file=data_file,
                output_dir=analysis_dir,
                max_categorical_values=self.config.analysis.max_categorical_values,
                max_numeric_columns=self.config.analysis.max_numeric_columns,
                feasibility=feasibility_payload,
                datasets=datasets,
                figure_dpi=self.config.analysis.figure_dpi,
                timeout_seconds=self.config.analysis.timeout_seconds,
                max_codegen_retries=self.config.analysis.max_codegen_retries,
                analysis_coder=self.analysis_coder if self.config.analysis.enable_ai_codegen else None,
            )
            manifest.stages.append(stage)
            storage.append_trace("analysis", "completed", stage.metrics)
            storage.save_manifest(manifest)

            draft_dir = storage.stage_dir(5, "drafting")
            draft_payload, stage = drafting.run_stage(
                idea_text=resolved_idea_text,
                output_dir=draft_dir,
                papers=papers,
                feasibility=feasibility_payload,
                datasets=datasets,
                analysis=analysis_payload,
                target_venue=self.config.paper.target_venue,
                writer_agent=self.writer_agent,
            )
            manifest.stages.append(stage)
            storage.append_trace("drafting", "completed", stage.metrics)
            storage.save_manifest(manifest)

            review_dir = storage.stage_dir(6, "review")
            review_payload, stage = review.run_stage(
                output_dir=review_dir,
                manuscript_text=draft_payload["manuscript_text"],
                papers=papers,
                feasibility=feasibility_payload,
                analysis=analysis_payload,
                quality_threshold=self.config.pipeline.quality_threshold,
                max_revisions=self.config.pipeline.max_revisions,
                reviewer_agent=self.reviewer_agent,
                revision_agent=self.revision_agent,
            )
            manifest.stages.append(stage)
            manifest.final_quality_score = review_payload["overall_score"]
            storage.append_trace("review", "completed", stage.metrics)
            storage.save_manifest(manifest)

            final_manuscript_text = review_payload.get("final_manuscript_text", draft_payload["manuscript_text"])
            final_md_path = review_dir / "final_manuscript.md"
            final_pdf_path = review_dir / "final_manuscript.pdf"
            final_bib_path = review_dir / "final_references.bib"
            final_md_path.write_text(final_manuscript_text, encoding="utf-8")
            drafting.write_manuscript_pdf(final_pdf_path, final_manuscript_text)
            drafting.write_bibtex(final_bib_path, papers)
            storage.append_trace(
                "review",
                "finalized_manuscript",
                {"final_markdown": str(final_md_path), "final_pdf": str(final_pdf_path)},
            )

            learning_signal = learning.build_learning_signal(
                run_id=resolved_run_id,
                idea_text=resolved_idea_text,
                papers=papers,
                datasets=datasets,
                feasibility=feasibility_payload,
                analysis=analysis_payload,
                review=review_payload,
            )
            storage.write_json(storage.run_dir / "learning_signal.json", learning_signal)
            storage.append_trace("learning", "signal_written", {"blockers": learning_signal["blockers"]})
            if self.config.submission.enabled:
                submission_path = self.submission_manager.create_submission_package(
                    storage.run_dir,
                    final_manuscript_path=final_pdf_path,
                    final_markdown_path=final_md_path,
                )
                storage.append_trace("submission", "package_created", {"path": str(submission_path)})

            aggregate_result = None
            if self.config.pipeline.auto_learn and not dry_run:
                if self.backend is not None:
                    aggregate_result = run_full_learning_cycle(
                        self.workspace_dir,
                        backend=self.backend,
                        current_prompts={
                            "writer": self.writer_agent.system_prompt if self.writer_agent else "",
                            "reviewer": self.reviewer_agent.system_prompt if self.reviewer_agent else "",
                            "revision": self.revision_agent.system_prompt if self.revision_agent else "",
                            "analysis_coder": self.analysis_coder.system_prompt if self.analysis_coder else "",
                        },
                        min_runs_before_evolution=self.config.learning.min_runs_before_evolution,
                    )
                else:
                    aggregate_result = learning.update_learning_model(
                        self.workspace_dir,
                        min_runs_before_update=self.config.learning.min_runs_before_update,
                    )
                storage.append_trace("learning", "aggregate_updated", {"updated": aggregate_result is not None})

            manifest.completed_at = iso_now()
            manifest.status = "completed"
            storage.append_trace("pipeline", "completed", {"final_quality_score": manifest.final_quality_score})
            storage.save_manifest(manifest)
            return {
                "run_id": resolved_run_id,
                "run_dir": str(storage.run_dir),
                "manifest": manifest.to_dict(),
                "literature": literature_payload,
                "datasets": dataset_payload,
                "analysis": analysis_payload,
                "review": review_payload,
                "automation": automation_result,
                "learning_model": aggregate_result,
            }
        except Exception as exc:
            manifest.completed_at = iso_now()
            manifest.status = "failed"
            storage.append_trace("pipeline", "failed", {"error": str(exc)})
            storage.save_manifest(manifest)
            raise

    def status(self, run_id: str) -> dict[str, Any]:
        return load_manifest(self.workspace_dir, run_id)

    def resolve_idea(
        self,
        *,
        idea_text: str | None = None,
        idea_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        if idea_text:
            return idea_text, idea_id
        if not idea_id:
            return None, None
        backlog = self.workspace_dir / "ideas" / "idea_backlog.jsonl"
        if not backlog.exists():
            return None, idea_id
        for line in backlog.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_id = entry.get("id") or entry.get("idea_id")
            if entry_id == idea_id:
                return entry.get("title") or entry.get("idea_text"), idea_id
        return None, idea_id

    def learn(self) -> dict[str, Any] | None:
        return learning.update_learning_model(
            self.workspace_dir,
            min_runs_before_update=self.config.learning.min_runs_before_update,
        )

    def capture_daily(self, input_path: str) -> dict[str, Any]:
        source_path = Path(input_path).expanduser().resolve()
        capture_dir = self.workspace_dir / "daily" / f"capture_{timestamp_for_id()}_{slugify(source_path.stem, 20)}"
        backlog_path = self.workspace_dir / "ideas" / "idea_backlog.jsonl"
        return daily_capture.capture_daily_ideas(
            input_path=source_path,
            output_dir=capture_dir,
            max_ideas=self.config.daily.max_ideas,
            min_idea_score=self.config.daily.min_idea_score,
            backlog_path=backlog_path,
        )

    def export_computer_task(self, run_id: str) -> str:
        run_dir = self.workspace_dir / "runs" / run_id
        return str(
            export_computer_use_task(
                run_dir,
                credential_lookup=self.credential_broker.get_credential,
                institutional_proxy_url=self.config.automation.institutional_proxy_url,
            )
        )

    def set_credential(
        self,
        resource: str,
        username: str,
        password: str,
        *,
        login_url: str | None = None,
        notes: str = "",
        extra_fields: dict[str, str] | None = None,
    ) -> None:
        self.credential_broker.set_credential(
            resource,
            username,
            password,
            login_url=login_url,
            notes=notes,
            extra_fields=extra_fields,
        )

    def list_credentials(self) -> list[str]:
        return self.credential_broker.list_resources()

    def run_computer_task(self, task_path: str) -> dict[str, Any]:
        def lookup(resource: str) -> dict[str, Any] | None:
            return self.credential_broker.get_credential(resource)

        return self.automation_runner.execute_task_pack(Path(task_path).resolve(), credential_lookup=lookup)

    def update_submission_status(self, run_id: str, status: str, notes: str = "") -> str:
        return str(self.submission_manager.update_status(run_id, status, notes))

    def find_recent_duplicate_research(
        self,
        idea_text: str,
        *,
        idea_id: str | None = None,
        similarity_threshold: float = 0.9,
        cooldown_days: int = 14,
    ) -> dict[str, Any] | None:
        runs_dir = self.workspace_dir / "runs"
        if not runs_dir.exists():
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, cooldown_days))
        for manifest_path in sorted(runs_dir.glob("*/run_manifest.json"), reverse=True):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            created_at = manifest.get("created_at")
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    if created_dt < cutoff:
                        continue
                except ValueError:
                    pass
            existing_id = manifest.get("idea_id")
            existing_text = manifest.get("idea_text", "")
            similarity = self._idea_similarity(idea_text, existing_text)
            if idea_id and existing_id and idea_id == existing_id:
                return {
                    "run_id": manifest.get("run_id"),
                    "status": manifest.get("status"),
                    "idea_id": existing_id,
                    "similarity": 1.0,
                }
            if similarity >= similarity_threshold:
                return {
                    "run_id": manifest.get("run_id"),
                    "status": manifest.get("status"),
                    "idea_id": existing_id,
                    "similarity": round(similarity, 4),
                }
        return None

    def _idea_similarity(self, left: str, right: str) -> float:
        if not left.strip() or not right.strip():
            return 0.0
        return max(overlap_score(left, right), overlap_score(right, left))
