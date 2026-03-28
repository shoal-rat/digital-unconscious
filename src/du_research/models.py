from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PaperCandidate:
    source: str
    title: str
    summary: str
    authors: list[str]
    year: int | None
    url: str
    pdf_url: str | None = None
    doi: str | None = None
    citation_count: int = 0
    subjects: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    datasets_used: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetCandidate:
    source: str
    title: str
    summary: str
    year: int | None
    url: str
    access: str
    file_count: int = 0
    formats: list[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageResult:
    order: int
    name: str
    status: str
    summary: str
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    idea_text: str
    idea_slug: str
    created_at: str
    idea_id: str | None = None
    completed_at: str | None = None
    status: str = "running"
    data_file: str | None = None
    final_quality_score: float | None = None
    stages: list[StageResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "idea_text": self.idea_text,
            "idea_slug": self.idea_slug,
            "idea_id": self.idea_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "data_file": self.data_file,
            "final_quality_score": self.final_quality_score,
            "stages": [stage.to_dict() for stage in self.stages],
        }
