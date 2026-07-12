"""Turn local papers and dataset structure into grounded research idea cards.

The module intentionally keeps the scientific state as linked JSON artifacts.
It does not pretend that a language model can prove novelty: models extract
anchored evidence, propose gaps, and challenge study cards; deterministic code
verifies provenance and computes the final ranking.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from du_research.ai_backend import AIBackend
from du_research.backlog import IdeaBacklog
from du_research.config import AppConfig
from du_research.utils import clamp, iso_now, timestamp_for_id

EVIDENCE_KINDS = {
    "claim",
    "finding",
    "method",
    "limitation",
    "open_question",
    "boundary_condition",
    "dataset",
}
OPPORTUNITY_TYPES = {
    "contradiction",
    "explicit_limitation",
    "boundary_extension",
    "method_transfer",
    "replication",
    "measurement_gap",
    "dataset_reuse",
    "temporal_gap",
    "robustness_test",
}

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "statement": {"type": "string"},
                    "anchor": {"type": "string"},
                    "locator": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "statement", "anchor", "locator", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evidence"],
    "additionalProperties": False,
}

OPPORTUNITY_SCHEMA = {
    "type": "object",
    "properties": {
        "opportunity_id": {"type": "string"},
        "type": {"type": "string"},
        "title": {"type": "string"},
        "rationale": {"type": "string"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "research_question": {"type": "string"},
    },
    "required": ["opportunity_id", "type", "title", "rationale", "evidence_ids", "research_question"],
    "additionalProperties": False,
}

IDEA_SCHEMA = {
    "type": "object",
    "properties": {
        "idea_id": {"type": "string"},
        "title": {"type": "string"},
        "opportunity_id": {"type": "string"},
        "research_question": {"type": "string"},
        "hypothesis": {"type": "string"},
        "null_hypothesis": {"type": "string"},
        "population": {"type": "string"},
        "context": {"type": "string"},
        "exposure": {"type": "string"},
        "outcome": {"type": "string"},
        "unit_of_analysis": {"type": "string"},
        "study_design": {"type": "string"},
        "operationalization": {"type": "string"},
        "required_data": {"type": "array", "items": {"type": "string"}},
        "confounders": {"type": "array", "items": {"type": "string"}},
        "negative_control": {"type": "string"},
        "smallest_useful_test": {"type": "string"},
        "falsifier": {"type": "string"},
        "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "opposing_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "contribution": {"type": "string"},
        "novelty_uncertainty": {"type": "string"},
        "status": {"type": "string"},
        "domains": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "idea_id",
        "title",
        "opportunity_id",
        "research_question",
        "hypothesis",
        "null_hypothesis",
        "population",
        "context",
        "exposure",
        "outcome",
        "unit_of_analysis",
        "study_design",
        "operationalization",
        "required_data",
        "confounders",
        "negative_control",
        "smallest_useful_test",
        "falsifier",
        "supporting_evidence_ids",
        "opposing_evidence_ids",
        "contribution",
        "novelty_uncertainty",
        "status",
        "domains",
    ],
    "additionalProperties": False,
}

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "opportunities": {"type": "array", "items": OPPORTUNITY_SCHEMA},
        "ideas": {"type": "array", "items": IDEA_SCHEMA},
    },
    "required": ["opportunities", "ideas"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idea_id": {"type": "string"},
                    "evidence_strength": {"type": "number"},
                    "testability": {"type": "number"},
                    "dataset_fit": {"type": "number"},
                    "information_gain": {"type": "number"},
                    "user_fit": {"type": "number"},
                    "novelty_risk": {"type": "string"},
                    "fatal_flaw": {"type": "string"},
                    "next_validation": {"type": "string"},
                },
                "required": [
                    "idea_id",
                    "evidence_strength",
                    "testability",
                    "dataset_fit",
                    "information_gain",
                    "user_fit",
                    "novelty_risk",
                    "fatal_flaw",
                    "next_validation",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


@dataclass
class SourceDocument:
    source_id: str
    kind: str
    title: str
    path: str
    content_sha256: str
    text: str = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("text", None)
        value["character_count"] = len(self.text)
        return value

    def prompt_dict(self) -> dict[str, Any]:
        """Return model-safe source metadata without local filesystem identity."""
        value = self.manifest_dict()
        value.pop("path", None)
        metadata = dict(value.get("metadata") or {})
        metadata.pop("path", None)
        value["metadata"] = metadata
        if self.kind == "dataset_profile":
            # A filename can be as sensitive as a path. The opaque source ID is
            # enough to link the profile to evidence while keeping local names
            # outside provider prompts.
            value["title"] = f"Local dataset {self.source_id}"
        return value


@dataclass
class EvidenceCard:
    evidence_id: str
    source_id: str
    kind: str
    statement: str
    anchor: str
    locator: str
    confidence: float
    extraction: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchIdeationLab:
    """Provenance-first paper/data → opportunity → study-card workflow."""

    def __init__(self, config: AppConfig, backend: AIBackend | None):
        self.config = config
        self.backend = backend
        self.workspace = Path(config.pipeline.workspace_dir).resolve()
        self.backlog = IdeaBacklog(self.workspace, config.retention.idea_backlog_max)

    def run(
        self,
        *,
        paper_paths: Iterable[str | Path] = (),
        data_paths: Iterable[str | Path] = (),
        context: str = "",
        max_ideas: int | None = None,
        add_to_backlog: bool = True,
        dry_run: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        paper_paths = list(paper_paths)
        data_paths = list(data_paths)
        if not paper_paths and not data_paths:
            raise ValueError("Provide at least one --paper or --data source.")

        max_ideas = max(1, min(max_ideas or self.config.ideation.max_ideas, 12))
        resolved_session_id = session_id or f"session_{timestamp_for_id()}"
        session_dir = self.workspace / "ideation" / resolved_session_id
        if session_dir.exists():
            raise FileExistsError(f"Ideation session already exists: {resolved_session_id}")
        session_dir.mkdir(parents=True, exist_ok=False)

        sources: list[SourceDocument] = []
        errors: list[str] = []
        for raw_path in paper_paths:
            try:
                sources.extend(_read_paper_source(Path(raw_path)))
            except Exception as exc:
                errors.append(f"{raw_path}: {exc}")
        for raw_path in data_paths:
            try:
                sources.append(
                    _profile_dataset(
                        Path(raw_path),
                        max_rows=self.config.ideation.max_profile_rows,
                    )
                )
            except Exception as exc:
                errors.append(f"{raw_path}: {exc}")
        sources = _dedupe_sources(sources)
        if not sources:
            raise ValueError("None of the supplied sources could be read. " + "; ".join(errors))

        manifest = {
            "session_id": resolved_session_id,
            "created_at": iso_now(),
            "context": context,
            "dry_run": dry_run,
            "sources": [source.manifest_dict() for source in sources],
            "errors": errors,
        }
        _write_json(session_dir / "source_manifest.json", manifest)

        evidence: list[EvidenceCard] = []
        for source in sources:
            if source.kind == "dataset_profile":
                evidence.extend(_dataset_evidence(source))
            elif dry_run or self.backend is None:
                evidence.extend(_heuristic_evidence(source))
            else:
                extracted = self._extract_evidence(source)
                evidence.extend(extracted or _heuristic_evidence(source))
        evidence = _dedupe_evidence(evidence)
        _write_json(session_dir / "evidence_cards.json", [card.to_dict() for card in evidence])

        if dry_run or self.backend is None:
            opportunities, ideas = _fallback_synthesis(evidence, sources, max_ideas=max_ideas)
        else:
            opportunities, ideas = self._synthesize(
                sources,
                evidence,
                context=context,
                max_ideas=max_ideas,
            )
            if not ideas:
                opportunities, ideas = _fallback_synthesis(evidence, sources, max_ideas=max_ideas)

        opportunities = _validate_opportunities(opportunities, evidence)
        ideas = _validate_ideas(ideas, evidence, opportunities, max_ideas=max_ideas)
        if not ideas:
            fallback_opportunities, fallback_ideas = _fallback_synthesis(
                evidence, sources, max_ideas=max_ideas
            )
            opportunities = _validate_opportunities(fallback_opportunities, evidence)
            ideas = _validate_ideas(fallback_ideas, evidence, opportunities, max_ideas=max_ideas)
        reviews = [] if dry_run or self.backend is None or not ideas else self._review(ideas, evidence, context)
        ideas = _rank_ideas(ideas, reviews)

        _write_json(session_dir / "opportunities.json", opportunities)
        _write_json(session_dir / "ideas.json", ideas)
        _write_json(session_dir / "reviews.json", reviews)
        report_path = session_dir / "report.md"
        report_path.write_text(
            _render_report(resolved_session_id, sources, evidence, opportunities, ideas, errors, dry_run),
            encoding="utf-8",
        )

        backlog_count = 0
        if add_to_backlog and not dry_run:
            backlog_entries = [
                {
                    **idea,
                    "idea_id": idea["idea_id"],
                    "description": idea.get("contribution", ""),
                    "domains": idea.get("domains", []),
                    "source_session": resolved_session_id,
                    "source_manifest": str(session_dir / "source_manifest.json"),
                }
                for idea in ideas
                if idea.get("verdict") in {"ready", "develop"}
            ]
            backlog_count = self.backlog.append_unique(
                backlog_entries,
                date=datetime.now(UTC).date().isoformat(),
                origin="paper_ideation",
            )

        result = {
            "session_id": resolved_session_id,
            "session_dir": str(session_dir),
            "report_path": str(report_path),
            "source_count": len(sources),
            "evidence_count": len(evidence),
            "opportunity_count": len(opportunities),
            "idea_count": len(ideas),
            "ideas_added_to_backlog": backlog_count,
            "errors": errors,
            "ideas": ideas,
        }
        _write_json(session_dir / "session.json", result)
        return result

    def _extract_evidence(self, source: SourceDocument) -> list[EvidenceCard]:
        excerpt = _bounded_text(source.text, self.config.ideation.max_source_characters)
        prompt = f"""Extract atomic evidence from the supplied research source.

Source id: {source.source_id}
Source title: {source.title}
Source kind: {source.kind}

Rules:
- Every item must contain a short anchor copied EXACTLY from SOURCE TEXT.
- Do not claim that a paper proves novelty or causality unless the anchor says so.
- Separate findings, methods, limitations, open questions, and boundary conditions.
- Prefer 4-8 high-information items; omit generic background.
- confidence is 0.0-1.0 confidence that the statement faithfully paraphrases the anchor.

Return only JSON:
{{"evidence":[{{"kind":"finding","statement":"...","anchor":"exact excerpt","locator":"page/section if visible","confidence":0.8}}]}}

SOURCE TEXT
{excerpt}
"""
        response = self.backend.call(
            prompt,
            mode="strict",
            system="You are a conservative research evidence extractor. Exact source grounding is mandatory.",
            model=self.config.ai.evidence_model,
            max_tokens=2600,
            json_schema=EVIDENCE_SCHEMA,
        )
        if not response.ok:
            return []
        payload = response.structured or _parse_json_object(response.text)
        rows = payload.get("evidence", []) if isinstance(payload, dict) else []
        cards: list[EvidenceCard] = []
        for index, row in enumerate(rows[:10], start=1):
            if not isinstance(row, dict):
                continue
            anchor = _clean_anchor(row.get("anchor"))
            if not anchor or not _anchor_exists(anchor, source.text):
                continue
            statement = str(row.get("statement") or "").strip()
            if not statement:
                continue
            kind = str(row.get("kind") or "claim").strip().lower()
            if kind not in EVIDENCE_KINDS:
                kind = "claim"
            cards.append(
                EvidenceCard(
                    evidence_id=f"ev_{source.source_id.removeprefix('src_')}_{index:02d}",
                    source_id=source.source_id,
                    kind=kind,
                    statement=statement[:700],
                    anchor=anchor,
                    locator=str(row.get("locator") or "source text")[:160],
                    confidence=round(_number(row.get("confidence"), 0.65, 0.0, 1.0), 3),
                    extraction=f"model:{self.config.ai.evidence_model}",
                )
            )
        return cards

    def _synthesize(
        self,
        sources: list[SourceDocument],
        evidence: list[EvidenceCard],
        *,
        context: str,
        max_ideas: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        source_summary = [source.prompt_dict() for source in sources]
        evidence_rows = [card.to_dict() for card in evidence]
        recent = self.backlog.recent_titles(30)
        prompt = f"""Build research opportunities and falsifiable study cards from linked evidence.

USER CONTEXT
{context or 'No extra context supplied.'}

SOURCE MANIFEST
{json.dumps(source_summary, ensure_ascii=False)}

VERIFIED EVIDENCE CARDS
{json.dumps(evidence_rows, ensure_ascii=False)}

RECENT IDEA TITLES TO AVOID REPEATING
{json.dumps(recent, ensure_ascii=False)}

First find opportunities using only these operators:
{', '.join(sorted(OPPORTUNITY_TYPES))}.

Then propose at most {max_ideas} genuinely different study cards. Every card must include:
- a specific research question, hypothesis, and explicit null hypothesis;
- population/context, exposure, outcome, unit of analysis, design, and operationalization;
- required variables or matched dataset profile;
- plausible confounders and a negative control where applicable;
- the smallest useful test and a result that would falsify the hypothesis;
- supporting evidence IDs and, when available, opposing evidence IDs;
- an honest novelty note phrased as uncertainty, never "this has never been studied";
- exploratory or confirmatory status.

An evidence ID is a citation, not permission to invent a fact. If sources are insufficient, propose fewer ideas.
Return only JSON:
{{
  "opportunities":[{{"opportunity_id":"opp_001","type":"measurement_gap","title":"...","rationale":"...","evidence_ids":["ev_..."],"research_question":"..."}}],
  "ideas":[{{"idea_id":"idea_001","title":"...","opportunity_id":"opp_001","research_question":"...","hypothesis":"...","null_hypothesis":"...","population":"...","context":"...","exposure":"...","outcome":"...","unit_of_analysis":"...","study_design":"...","operationalization":"...","required_data":["..."],"confounders":["..."],"negative_control":"...","smallest_useful_test":"...","falsifier":"...","supporting_evidence_ids":["ev_..."],"opposing_evidence_ids":[],"contribution":"...","novelty_uncertainty":"...","status":"exploratory","domains":["..."]}}]
}}
"""
        response = self.backend.call(
            prompt,
            mode="creative",
            system="You are a research design architect. Prefer a small decisive test over a grand vague project.",
            model=self.config.ai.ideation_model,
            max_tokens=6000,
            json_schema=SYNTHESIS_SCHEMA,
            think=self.config.ai.think_idea_budget or "high",
            web_search=False,
        )
        if not response.ok:
            return [], []
        payload = response.structured or _parse_json_object(response.text)
        if not isinstance(payload, dict):
            return [], []
        return list(payload.get("opportunities") or []), list(payload.get("ideas") or [])

    def _review(
        self,
        ideas: list[dict[str, Any]],
        evidence: list[EvidenceCard],
        context: str,
    ) -> list[dict[str, Any]]:
        prompt = f"""Adversarially review these research study cards.

Study cards:
{json.dumps(ideas, ensure_ascii=False)}

Evidence index:
{json.dumps([card.to_dict() for card in evidence], ensure_ascii=False)}

User context: {context or 'none'}

Score 0-100 on evidence_strength, testability, dataset_fit, information_gain, and user_fit.
Do not score novelty itself; describe novelty_risk because the supplied corpus cannot prove global novelty.
Name the strongest fatal flaw, if any, and the next validation step.
Return only JSON:
{{"reviews":[{{"idea_id":"...","evidence_strength":0,"testability":0,"dataset_fit":0,"information_gain":0,"user_fit":0,"novelty_risk":"...","fatal_flaw":"...","next_validation":"..."}}]}}
"""
        response = self.backend.call(
            prompt,
            mode="strict",
            system="You are a skeptical methods reviewer. Penalize unsupported causal language and unavailable variables.",
            model=self.config.ai.ideation_review_model,
            max_tokens=3000,
            json_schema=REVIEW_SCHEMA,
            think=self.config.ai.think_judge_budget or "high",
            web_search=False,
        )
        if not response.ok:
            return []
        payload = response.structured or _parse_json_object(response.text)
        return list(payload.get("reviews") or []) if isinstance(payload, dict) else []


def _read_paper_source(path: Path) -> list[SourceDocument]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"paper source not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError('PDF input requires `pip install "digital-unconscious[papers]"`') from exc
        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[page {index}]\n{text}")
        if not pages:
            raise ValueError("the PDF contains no extractable text; OCR it first")
        return [_make_source("paper", path.stem, path, "\n\n".join(pages), {"pages": len(reader.pages)})]
    if suffix in {".txt", ".md", ".rst"}:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("source is empty")
        return [_make_source("paper", path.stem, path, text, {})]
    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        return _paper_records(path, records)
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            records = value.get("papers") or value.get("extracted_content")
            if not isinstance(records, list):
                records = [value]
        elif isinstance(value, list):
            records = value
        else:
            raise ValueError("JSON paper input must contain an object or list")
        return _paper_records(path, [record for record in records if isinstance(record, dict)])
    raise ValueError("supported paper formats are .pdf, .txt, .md, .rst, .json, and .jsonl")


def _paper_records(path: Path, records: list[dict[str, Any]]) -> list[SourceDocument]:
    sources = []
    for index, record in enumerate(records, start=1):
        title = str(record.get("title") or f"{path.stem} record {index}")
        text_parts = []
        for key in (
            "abstract",
            "summary",
            "text",
            "content",
            "claims",
            "findings",
            "methods",
            "datasets_used",
            "limitations",
            "key_findings",
        ):
            value = record.get(key)
            if isinstance(value, list):
                value = "\n".join(str(item) for item in value)
            if value:
                text_parts.append(f"{key}: {value}")
        text = "\n".join(text_parts) or json.dumps(record, ensure_ascii=False)
        metadata = {
            key: record[key]
            for key in ("doi", "url", "authors", "year", "source")
            if record.get(key) not in (None, "", [])
        }
        sources.append(_make_source("paper_record", title, path, text, metadata, salt=str(index)))
    if not sources:
        raise ValueError("JSON paper input contains no readable records")
    return sources


def _profile_dataset(path: Path, *, max_rows: int) -> SourceDocument:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        profile = _profile_delimited(path, delimiter=delimiter, max_rows=max_rows)
    elif suffix in {".json", ".jsonl"}:
        profile = _profile_json_rows(path, max_rows=max_rows)
    else:
        raise ValueError("supported dataset formats are .csv, .tsv, .json, and .jsonl")
    text = json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True)
    return _make_source("dataset_profile", path.stem, path, text, profile)


def _profile_delimited(path: Path, *, delimiter: str, max_rows: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("dataset has no header")
        rows = []
        total = 0
        for row in reader:
            total += 1
            if len(rows) < max_rows:
                rows.append(row)
    return _rows_profile(path, rows, total_rows=total, fieldnames=list(reader.fieldnames))


def _profile_json_rows(path: Path, *, max_rows: int) -> dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        values = []
        total = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            if len(values) < max_rows:
                value = json.loads(line)
                if isinstance(value, dict):
                    values.append(value)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("rows") or [payload]
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must contain rows as a list of objects")
        total = len(payload)
        values = [row for row in payload[:max_rows] if isinstance(row, dict)]
    fields = sorted({str(key) for row in values for key in row})
    return _rows_profile(path, values, total_rows=total, fieldnames=fields)


def _rows_profile(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    total_rows: int,
    fieldnames: list[str],
) -> dict[str, Any]:
    field_profiles = []
    for name in fieldnames:
        values = [row.get(name) for row in rows]
        present = [value for value in values if value not in (None, "")]
        inferred, numeric = _infer_type(present)
        unique = len({json.dumps(value, sort_keys=True, default=str) for value in present})
        item: dict[str, Any] = {
            "name": name,
            "inferred_type": inferred,
            "missing_count_sample": len(values) - len(present),
            "missing_rate_sample": round((len(values) - len(present)) / max(1, len(values)), 4),
            "unique_count_sample": unique,
        }
        if numeric:
            item["numeric_min_sample"] = min(numeric)
            item["numeric_max_sample"] = max(numeric)
        field_profiles.append(item)
    return {
        "path": str(path),
        "format": path.suffix.lower().lstrip("."),
        "row_count": total_rows,
        "profiled_rows": len(rows),
        "column_count": len(fieldnames),
        "fields": field_profiles,
        "privacy_note": "Only schema and aggregate profile statistics are sent to a model; raw rows are excluded.",
    }


def _infer_type(values: list[Any]) -> tuple[str, list[float]]:
    if not values:
        return "unknown", []
    numeric = []
    for value in values:
        if isinstance(value, bool):
            break
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            break
    if len(numeric) == len(values):
        return "numeric", numeric
    lowered = {str(value).strip().casefold() for value in values}
    if lowered <= {"true", "false", "yes", "no", "0", "1"}:
        return "boolean", []
    return "text", []


def _make_source(
    kind: str,
    title: str,
    path: Path,
    text: str,
    metadata: dict[str, Any],
    *,
    salt: str = "",
) -> SourceDocument:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    identity = hashlib.sha256(f"{path}:{salt}:{digest}".encode()).hexdigest()[:12]
    return SourceDocument(
        source_id=f"src_{identity}",
        kind=kind,
        title=title.strip() or path.stem,
        path=str(path),
        content_sha256=digest,
        text=text,
        metadata=metadata,
    )


def _dedupe_sources(sources: list[SourceDocument]) -> list[SourceDocument]:
    seen = set()
    result = []
    for source in sources:
        key = (source.kind, source.content_sha256)
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


def _bounded_text(text: str, max_characters: int) -> str:
    if len(text) <= max_characters:
        return text
    head = max_characters * 2 // 3
    tail = max_characters - head
    return text[:head] + "\n\n[... middle omitted for bounded extraction ...]\n\n" + text[-tail:]


def _clean_anchor(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:320]


def _anchor_exists(anchor: str, text: str) -> bool:
    def normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    return normalize(anchor) in normalize(text)


def _heuristic_evidence(source: SourceDocument) -> list[EvidenceCard]:
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", source.text) if len(part.strip()) >= 40]
    scored = []
    signals = {
        "limitation": ("limitation", "limited", "future work", "cannot", "however"),
        "method": ("method", "sample", "regression", "experiment", "survey", "model"),
        "finding": ("found", "result", "associated", "effect", "increase", "decrease"),
        "open_question": ("future", "remains", "unknown", "question"),
    }
    for index, chunk in enumerate(chunks):
        lowered = chunk.casefold()
        kind = "claim"
        score = 0
        for candidate_kind, words in signals.items():
            hits = sum(word in lowered for word in words)
            if hits > score:
                kind, score = candidate_kind, hits
        scored.append((score, -index, kind, chunk))
    scored.sort(reverse=True)
    cards = []
    for index, (_, _, kind, chunk) in enumerate(scored[:6], start=1):
        anchor = _clean_anchor(chunk)
        cards.append(
            EvidenceCard(
                evidence_id=f"ev_{source.source_id.removeprefix('src_')}_{index:02d}",
                source_id=source.source_id,
                kind=kind,
                statement=anchor,
                anchor=anchor,
                locator="source text",
                confidence=0.5,
                extraction="deterministic_fallback",
            )
        )
    return cards


def _dataset_evidence(source: SourceDocument) -> list[EvidenceCard]:
    profile = source.metadata
    fields = profile.get("fields", [])
    base = source.source_id.removeprefix("src_")
    cards = [
        EvidenceCard(
            evidence_id=f"ev_{base}_01",
            source_id=source.source_id,
            kind="dataset",
            statement=(
                f"The local dataset profile contains {profile.get('row_count', 0)} rows and "
                f"{profile.get('column_count', len(fields))} columns."
            ),
            anchor=f'"row_count": {profile.get("row_count", 0)}',
            locator="profile.row_count",
            confidence=1.0,
            extraction="local_profile",
        )
    ]
    for index, field_profile in enumerate(fields[:12], start=2):
        name = str(field_profile.get("name") or "")
        if not name:
            continue
        cards.append(
            EvidenceCard(
                evidence_id=f"ev_{base}_{index:02d}",
                source_id=source.source_id,
                kind="dataset",
                statement=(
                    f"Field {name!r} is profiled as {field_profile.get('inferred_type', 'unknown')} "
                    f"with sample missing rate {field_profile.get('missing_rate_sample', 0):.1%}."
                ),
                anchor=f'"name": "{name}"',
                locator=f"profile.fields.{name}",
                confidence=1.0,
                extraction="local_profile",
            )
        )
    return cards


def _dedupe_evidence(cards: list[EvidenceCard]) -> list[EvidenceCard]:
    seen = set()
    result = []
    for card in cards:
        key = (card.source_id, card.kind, " ".join(card.anchor.casefold().split()))
        if key in seen:
            continue
        seen.add(key)
        result.append(card)
    return result


def _validate_opportunities(
    opportunities: list[dict[str, Any]],
    evidence: list[EvidenceCard],
) -> list[dict[str, Any]]:
    valid_ids = {card.evidence_id for card in evidence}
    result = []
    for index, row in enumerate(opportunities):
        if not isinstance(row, dict):
            continue
        cited = [str(value) for value in row.get("evidence_ids", []) if str(value) in valid_ids]
        if not cited:
            continue
        kind = str(row.get("type") or "robustness_test")
        if kind not in OPPORTUNITY_TYPES:
            kind = "robustness_test"
        result.append(
            {
                **row,
                "opportunity_id": str(row.get("opportunity_id") or f"opp_{index + 1:03d}"),
                "type": kind,
                "evidence_ids": cited,
            }
        )
    return result


def _validate_ideas(
    ideas: list[dict[str, Any]],
    evidence: list[EvidenceCard],
    opportunities: list[dict[str, Any]],
    *,
    max_ideas: int,
) -> list[dict[str, Any]]:
    valid_evidence = {card.evidence_id for card in evidence}
    valid_opportunities = {str(row.get("opportunity_id")) for row in opportunities}
    required = (
        "title",
        "research_question",
        "hypothesis",
        "null_hypothesis",
        "study_design",
        "smallest_useful_test",
        "falsifier",
    )
    result = []
    seen_titles = set()
    for row in ideas:
        if not isinstance(row, dict) or any(not str(row.get(key) or "").strip() for key in required):
            continue
        supporting = [
            str(value) for value in row.get("supporting_evidence_ids", []) if str(value) in valid_evidence
        ]
        opposing = [
            str(value) for value in row.get("opposing_evidence_ids", []) if str(value) in valid_evidence
        ]
        if not supporting:
            continue
        title = str(row["title"]).strip()
        title_key = IdeaBacklog.key(title)
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        idea_id = f"idea_{hashlib.sha256((title + str(row.get('research_question'))).encode('utf-8')).hexdigest()[:12]}"
        opportunity_id = str(row.get("opportunity_id") or "")
        result.append(
            {
                **row,
                "idea_id": idea_id,
                "id": idea_id,
                "title": title[:120],
                "opportunity_id": opportunity_id if opportunity_id in valid_opportunities else None,
                "supporting_evidence_ids": supporting,
                "opposing_evidence_ids": opposing,
                "evidence_ids": list(dict.fromkeys([*supporting, *opposing])),
                "status": "confirmatory" if str(row.get("status")).lower() == "confirmatory" else "exploratory",
                "speculative": True,
            }
        )
        if len(result) >= max_ideas:
            break
    return result


def _rank_ideas(ideas: list[dict[str, Any]], reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_map = {str(row.get("idea_id")): row for row in reviews if isinstance(row, dict)}
    weights = {
        "evidence_strength": 0.24,
        "testability": 0.24,
        "dataset_fit": 0.20,
        "information_gain": 0.18,
        "user_fit": 0.14,
    }
    ranked = []
    for idea in ideas:
        review = review_map.get(str(idea.get("idea_id")), {})
        if review:
            scores = {key: round(_number(review.get(key), 50, 0, 100), 1) for key in weights}
        else:
            evidence_count = len(idea.get("evidence_ids", []))
            data_known = bool(idea.get("required_data"))
            scores = {
                "evidence_strength": min(85.0, 48.0 + 8 * evidence_count),
                "testability": 62.0,
                "dataset_fit": 58.0 if data_known else 45.0,
                "information_gain": 60.0,
                "user_fit": 50.0,
            }
        total = round(sum(scores[key] * weight for key, weight in weights.items()), 1)
        verdict = "ready" if total >= 75 else "develop" if total >= 60 else "fragile"
        ranked.append(
            {
                **idea,
                "scores": scores,
                "total_score": total,
                "verdict": verdict,
                "novelty_risk": str(review.get("novelty_risk") or "Novelty is unverified outside the supplied sources."),
                "fatal_flaw": str(review.get("fatal_flaw") or ""),
                "next_validation": str(review.get("next_validation") or idea.get("smallest_useful_test") or ""),
            }
        )
    return sorted(ranked, key=lambda item: item["total_score"], reverse=True)


def _fallback_synthesis(
    evidence: list[EvidenceCard],
    sources: list[SourceDocument],
    *,
    max_ideas: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Conservative fallback that keeps the workflow usable during model outage."""
    opportunities = []
    ideas = []
    limitations = [card for card in evidence if card.kind in {"limitation", "open_question"}]
    dataset_cards = [card for card in evidence if card.kind == "dataset"]
    candidates = limitations or [card for card in evidence if card.kind in {"finding", "claim"}]
    for index, card in enumerate(candidates[:max_ideas], start=1):
        dataset_id = dataset_cards[0].evidence_id if dataset_cards else None
        cited = [card.evidence_id] + ([dataset_id] if dataset_id else [])
        opp_id = f"opp_{index:03d}"
        opportunity_type = "dataset_reuse" if dataset_id else "explicit_limitation" if card.kind == "limitation" else "robustness_test"
        opportunities.append(
            {
                "opportunity_id": opp_id,
                "type": opportunity_type,
                "title": f"Test the boundary of {card.statement[:72]}",
                "rationale": "A source-grounded candidate produced without model synthesis.",
                "evidence_ids": cited,
                "research_question": f"Under what conditions does this claim hold: {card.statement[:180]}?",
            }
        )
        ideas.append(
            {
                "title": f"Boundary test: {card.statement[:80]}",
                "opportunity_id": opp_id,
                "research_question": "Under what observable conditions does this source claim hold?",
                "hypothesis": f"The relationship described in {card.evidence_id} changes across a measurable boundary condition.",
                "null_hypothesis": "The relationship does not vary across the selected boundary condition.",
                "population": "To be specified from the source and available data",
                "context": "Supplied research material",
                "exposure": "Boundary condition to be operationalized",
                "outcome": "Outcome named in the cited evidence",
                "unit_of_analysis": "To be specified",
                "study_design": "Pre-registered robustness or subgroup analysis",
                "operationalization": "Map source constructs to documented dataset variables before analysis.",
                "required_data": ["exposure", "outcome", "boundary condition"],
                "confounders": [],
                "negative_control": "Choose a pre-treatment variable not expected to respond.",
                "smallest_useful_test": "Verify variable availability, direction, and sample support before modeling.",
                "falsifier": "No stable difference across the pre-specified boundary with adequate power.",
                "supporting_evidence_ids": cited,
                "opposing_evidence_ids": [],
                "contribution": "A bounded robustness check grounded in the supplied source.",
                "novelty_uncertainty": "No external novelty search was performed.",
                "status": "exploratory",
                "domains": [],
            }
        )
    return opportunities, ideas


def _render_report(
    session_id: str,
    sources: list[SourceDocument],
    evidence: list[EvidenceCard],
    opportunities: list[dict[str, Any]],
    ideas: list[dict[str, Any]],
    errors: list[str],
    dry_run: bool,
) -> str:
    lines = [
        f"# Research Idea Lab — {session_id}",
        "",
        f"_Generated {iso_now()}. {'Deterministic dry run; model synthesis was skipped.' if dry_run else 'Novelty remains an explicit uncertainty until searched.'}_",
        "",
        "## Source ledger",
        "",
    ]
    for source in sources:
        lines.append(f"- `{source.source_id}` — **{source.title}** ({source.kind}, SHA-256 `{source.content_sha256[:12]}…`)")
    if errors:
        lines.extend(["", "## Source warnings", "", *[f"- {error}" for error in errors]])
    lines.extend(["", f"## Evidence cards ({len(evidence)})", ""])
    for card in evidence:
        lines.append(f"- `{card.evidence_id}` · **{card.kind}** · {card.statement}")
        lines.append(f"  - Anchor: “{card.anchor}” ({card.locator})")
    lines.extend(["", f"## Opportunities ({len(opportunities)})", ""])
    for opportunity in opportunities:
        lines.append(f"### {opportunity.get('title', 'Untitled opportunity')}")
        lines.append("")
        lines.append(f"- Pattern: `{opportunity.get('type', '')}`")
        lines.append(f"- Evidence: {', '.join(f'`{value}`' for value in opportunity.get('evidence_ids', []))}")
        lines.append(f"- Question: {opportunity.get('research_question', '')}")
        lines.append("")
    lines.extend([f"## Ranked study cards ({len(ideas)})", ""])
    for index, idea in enumerate(ideas, start=1):
        lines.append(f"### {index}. {idea['title']} — {idea.get('total_score', 0):.1f}")
        lines.append("")
        lines.append(f"- Verdict: **{idea.get('verdict', 'fragile')}** · {idea.get('status', 'exploratory')}")
        lines.append(f"- Question: {idea.get('research_question', '')}")
        lines.append(f"- Hypothesis: {idea.get('hypothesis', '')}")
        lines.append(f"- Null: {idea.get('null_hypothesis', '')}")
        lines.append(f"- Design: {idea.get('study_design', '')}")
        lines.append(f"- Smallest useful test: {idea.get('smallest_useful_test', '')}")
        lines.append(f"- Falsifier: {idea.get('falsifier', '')}")
        lines.append(f"- Evidence: {', '.join(f'`{value}`' for value in idea.get('evidence_ids', []))}")
        lines.append(f"- Novelty risk: {idea.get('novelty_risk', '')}")
        if idea.get("fatal_flaw"):
            lines.append(f"- Main risk: {idea['fatal_flaw']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(text[start:end])
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


def _number(value: Any, default: float, low: float, high: float) -> float:
    try:
        return clamp(float(value), low, high)
    except (TypeError, ValueError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
