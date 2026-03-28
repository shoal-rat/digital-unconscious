from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from du_research.models import DatasetCandidate, StageResult
from du_research.net import build_url, fetch_json
from du_research.utils import overlap_score, recentness_score, safe_year, strip_html


class DatasetProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[DatasetCandidate]:
        ...


@dataclass
class ZenodoProvider:
    timeout: int = 15
    name: str = "zenodo"

    def search(self, query: str, max_results: int) -> list[DatasetCandidate]:
        url = build_url("https://zenodo.org/api/records", {"q": query, "size": max_results})
        payload = fetch_json(url, timeout=self.timeout)
        records = payload.get("hits", {}).get("hits", [])
        datasets: list[DatasetCandidate] = []
        for record in records:
            metadata = record.get("metadata", {})
            title = metadata.get("title", "")
            summary = strip_html(metadata.get("description", ""))
            year = safe_year(metadata.get("publication_date"))
            file_count = len(record.get("files", []) or [])
            access = metadata.get("access_right", "unknown")
            formats = [item.get("type", "") for item in record.get("files", []) if item.get("type")]
            candidate = DatasetCandidate(
                source=self.name,
                title=title,
                summary=summary,
                year=year,
                url=record.get("links", {}).get("self_html", ""),
                access=access,
                file_count=file_count,
                formats=formats,
            )
            candidate.score = round(
                (0.6 * overlap_score(query, f"{title} {summary}"))
                + (0.2 * recentness_score(year))
                + (0.2 * (1.0 if access == "open" else 0.4)),
                4,
            )
            datasets.append(candidate)
        return datasets


@dataclass
class DataCiteProvider:
    timeout: int = 15
    name: str = "datacite"

    def search(self, query: str, max_results: int) -> list[DatasetCandidate]:
        url = build_url("https://api.datacite.org/dois", {"query": query, "page[size]": max_results})
        payload = fetch_json(url, timeout=self.timeout)
        datasets: list[DatasetCandidate] = []
        for item in payload.get("data", []):
            attributes = item.get("attributes", {})
            types = attributes.get("types", {}) or {}
            resource_general = (types.get("resourceTypeGeneral") or "").lower()
            if resource_general and resource_general != "dataset":
                continue
            title_entries = attributes.get("titles") or []
            title = title_entries[0].get("title", "") if title_entries else ""
            descriptions = attributes.get("descriptions") or []
            summary = descriptions[0].get("description", "") if descriptions else ""
            year = safe_year(attributes.get("published"))
            candidate = DatasetCandidate(
                source=self.name,
                title=title,
                summary=strip_html(summary),
                year=year,
                url=attributes.get("url") or attributes.get("doi", ""),
                access="unknown",
            )
            candidate.score = round(
                (0.7 * overlap_score(query, f"{title} {summary}"))
                + (0.3 * recentness_score(year)),
                4,
            )
            datasets.append(candidate)
        return datasets


def _dedupe(items: list[DatasetCandidate]) -> list[DatasetCandidate]:
    deduped: dict[str, DatasetCandidate] = {}
    for item in items:
        key = (item.url or item.title).lower().strip()
        existing = deduped.get(key)
        if existing is None or item.score > existing.score:
            deduped[key] = item
    return sorted(deduped.values(), key=lambda dataset: dataset.score, reverse=True)


def _summary_markdown(query: str, datasets: list[DatasetCandidate], errors: list[str]) -> str:
    lines = [
        "# Dataset Discovery",
        "",
        f"- Query: `{query}`",
        f"- Dataset candidates: {len(datasets)}",
        "",
        "## Top Candidates",
        "",
    ]
    for index, dataset in enumerate(datasets[:10], start=1):
        lines.append(
            f"{index}. **{dataset.title}** ({dataset.year or 'n.d.'}, {dataset.source}, {dataset.access}, score {dataset.score:.2f})"
        )
        if dataset.summary:
            lines.append(f"   - Summary: {dataset.summary[:240]}")
        lines.append(f"   - URL: {dataset.url}")
    if errors:
        lines.extend(["", "## Provider Errors", ""])
        for error in errors:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def run_stage(
    idea_text: str,
    output_dir: Path,
    max_results_per_source: int,
    timeout: int,
    dry_run: bool = False,
    providers: list[DatasetProvider] | None = None,
) -> tuple[list[DatasetCandidate], dict, StageResult]:
    providers = providers if providers is not None else [ZenodoProvider(timeout), DataCiteProvider(timeout)]
    errors: list[str] = []
    found: list[DatasetCandidate] = []
    provider_counts: dict[str, int] = {}

    if not dry_run:
        for provider in providers:
            try:
                results = provider.search(idea_text, max_results_per_source)
                found.extend(results)
                provider_counts[provider.name] = len(results)
            except Exception as exc:
                provider_counts[provider.name] = 0
                errors.append(f"{provider.name}: {exc}")

    ranked = _dedupe(found)
    acquisition_steps = [
        f"Inspect the top {min(3, len(ranked))} dataset landing pages for variable coverage.",
        "Verify licensing and download format before analysis.",
        "Prefer sources with open access and explicit documentation.",
    ]
    payload = {
        "query": idea_text,
        "dataset_count": len(ranked),
        "provider_counts": provider_counts,
        "errors": errors,
        "acquisition_steps": acquisition_steps,
        "datasets": [dataset.to_dict() for dataset in ranked],
    }
    json_path = output_dir / "datasets.json"
    md_path = output_dir / "plan.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_summary_markdown(idea_text, ranked, errors), encoding="utf-8")
    result = StageResult(
        order=3,
        name="data_sources",
        status="completed" if ranked or not errors else "degraded",
        summary=f"Ranked {len(ranked)} dataset candidate(s).",
        artifacts=[str(json_path), str(md_path)],
        metrics={"dataset_count": len(ranked), "provider_counts": provider_counts},
    )
    return ranked, payload, result
