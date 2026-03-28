from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from du_research.agents.writer import WriterAgent
from du_research.models import DatasetCandidate, PaperCandidate, StageResult
from du_research.net import fetch_json


def _format_reference(index: int, paper: PaperCandidate) -> str:
    author_text = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
    return f"{index}. {author_text} ({paper.year or 'n.d.'}). {paper.title}. {paper.url}"


def _bibtex_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _paper_key(index: int, paper: PaperCandidate) -> str:
    first_author = (paper.authors[0].split()[-1].lower() if paper.authors else "unknown").replace(" ", "")
    return f"{first_author}{paper.year or 'nd'}{index}"


def write_bibtex(output_path: Path, papers: list[PaperCandidate]) -> None:
    entries = []
    for index, paper in enumerate(papers[:20], start=1):
        key = _paper_key(index, paper)
        author_text = " and ".join(_bibtex_escape(name) for name in (paper.authors or ["Unknown"]))
        title = _bibtex_escape(paper.title or "Untitled")
        year = paper.year or "n.d."
        url = _bibtex_escape(paper.url or "")
        doi_line = f"  doi = {{{_bibtex_escape(paper.doi)}}},\n" if paper.doi else ""
        entries.append(
            "@article{" + key + ",\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{author_text}}},\n"
            f"  year = {{{year}}},\n"
            f"  url = {{{url}}},\n"
            f"{doi_line}"
            "}\n"
        )
    output_path.write_text("\n".join(entries), encoding="utf-8")


def _verify_doi(doi: str | None, timeout: int = 15) -> bool:
    if not doi:
        return False
    try:
        fetch_json(f"https://api.crossref.org/works/{doi}", timeout=timeout)
        return True
    except Exception:
        return False


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_manuscript_pdf(output_path: Path, manuscript_text: str) -> None:
    lines = []
    for raw in manuscript_text.splitlines():
        if not raw.strip():
            lines.append("")
            continue
        text = raw.strip()
        while len(text) > 90:
            lines.append(text[:90])
            text = text[90:]
        lines.append(text)
    pages = [lines[index:index + 44] for index in range(0, len(lines), 44)] or [[""]]
    objects: list[bytes] = []
    kids = []
    for page_number, page_lines in enumerate(pages, start=1):
        content_stream = ["BT", "/F1 10 Tf", "50 792 Td", "14 TL"]
        first = True
        for line in page_lines:
            escaped = _escape_pdf_text(line)
            if first:
                content_stream.append(f"({escaped}) Tj")
                first = False
            else:
                content_stream.append(f"T* ({escaped}) Tj")
        content_stream.append("ET")
        stream_bytes = "\n".join(content_stream).encode("latin-1", errors="replace")
        content_obj_num = 4 + (page_number - 1) * 2
        page_obj_num = 5 + (page_number - 1) * 2
        objects.append(
            f"{content_obj_num} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream\nendobj\n"
        )
        objects.append(
            (
                f"{page_obj_num} 0 obj\n"
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj_num} 0 R >>\n"
                f"endobj\n"
            ).encode("ascii")
        )
        kids.append(f"{page_obj_num} 0 R")

    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pages_obj = f"2 0 obj\n<< /Type /Pages /Count {len(pages)} /Kids [{' '.join(kids)}] >>\nendobj\n".encode("ascii")
    font_obj = b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    ordered_objects = [catalog, pages_obj, font_obj, *objects]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj in ordered_objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode("ascii")
    )
    output_path.write_bytes(pdf)


def run_stage(
    idea_text: str,
    output_dir: Path,
    papers: list[PaperCandidate],
    feasibility: dict[str, Any],
    datasets: list[DatasetCandidate],
    analysis: dict[str, Any],
    target_venue: str,
    writer_agent: WriterAgent | None = None,
) -> tuple[dict[str, Any], StageResult]:
    top_titles = [paper.title for paper in papers[:3]]
    dataset_titles = [dataset.title for dataset in datasets[:3]]
    methods = feasibility.get("recommended_methods", [])
    results_lines = []
    if analysis.get("analysis_executed"):
        results_lines.append(
            f"Descriptive analysis covered {analysis.get('row_count', 0)} rows across {analysis.get('column_count', 0)} columns."
        )
        numeric_summary = analysis.get("numeric_summary", {})
        for column, summary in list(numeric_summary.items())[:3]:
            results_lines.append(
                f"- `{column}` mean={summary['mean']}, min={summary['min']}, max={summary['max']}"
            )
    else:
        results_lines.append("No local dataset was supplied, so this draft includes a methods and acquisition plan rather than validated results.")

    manuscript = [
        f"# {idea_text}",
        "",
        f"_Target venue: {target_venue}_",
        "",
        "## Abstract",
        "",
        f"This dossier evaluates the research potential of the idea '{idea_text}'. The current evidence base supports a `{feasibility['decision']}` decision with confidence {feasibility['confidence']}/100. The most distinctive angle is: {feasibility['novel_angle']}.",
        "",
        "## Research Question",
        "",
        f"How can '{idea_text}' be operationalized into a measurable empirical study?",
        "",
        "## Literature Snapshot",
        "",
        f"The literature scout found {len(papers)} relevant open-source candidates. The strongest leads were: {', '.join(top_titles) if top_titles else 'none found'}.",
        "",
        "## Data Plan",
        "",
        f"Dataset discovery surfaced {len(datasets)} candidate sources. Immediate priorities: {', '.join(dataset_titles) if dataset_titles else 'manual sourcing required'}.",
        "",
        "## Methods",
        "",
        f"Recommended methods: {', '.join(methods) if methods else 'descriptive-statistics'}.",
        "",
        "## Results Or Planned Analysis",
        "",
        *results_lines,
        "",
        "## Limitations",
        "",
        "- Literature relevance is heuristic and may miss important adjacent work.",
        "- Dataset rankings prioritize openness and metadata richness, not causal fit.",
        "- Any conclusions remain provisional until domain-specific analysis is completed.",
        "",
        "## Next Actions",
        "",
        "1. Validate the top two literature leads manually.",
        "2. Inspect the top dataset candidate for variable coverage.",
        "3. Extend the analysis from descriptive profiling to hypothesis testing once data suitability is confirmed.",
        "",
        "## References",
        "",
    ]
    for index, paper in enumerate(papers[:8], start=1):
        manuscript.append(_format_reference(index, paper))
    heuristic_manuscript = "\n".join(manuscript) + "\n"
    manuscript_text = heuristic_manuscript
    if writer_agent is not None:
        ai_manuscript = writer_agent.write(
            {
                "idea_text": idea_text,
                "target_venue": target_venue,
                "feasibility": feasibility,
                "analysis": analysis,
                "papers": [paper.to_dict() for paper in papers[:12]],
                "datasets": [dataset.to_dict() for dataset in datasets[:8]],
                "fallback_manuscript": heuristic_manuscript,
            }
        )
        if ai_manuscript:
            manuscript_text = ai_manuscript
    doi_verification = {
        (paper.doi or paper.title): _verify_doi(paper.doi)
        for paper in papers[:12]
    }
    payload = {
        "title": idea_text,
        "reference_count": min(8, len(papers)),
        "analysis_executed": analysis.get("analysis_executed", False),
        "doi_verified_count": sum(1 for value in doi_verification.values() if value),
    }
    md_path = output_dir / "manuscript.md"
    pdf_path = output_dir / "manuscript.pdf"
    bib_path = output_dir / "references.bib"
    json_path = output_dir / "draft_summary.json"
    md_path.write_text(manuscript_text, encoding="utf-8")
    write_manuscript_pdf(pdf_path, manuscript_text)
    write_bibtex(bib_path, papers)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "doi_verification.json").write_text(
        json.dumps(doi_verification, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result = StageResult(
        order=5,
        name="drafting",
        status="completed",
        summary=f"Generated manuscript starter with {payload['reference_count']} references.",
        artifacts=[str(md_path), str(pdf_path), str(bib_path), str(json_path), str(output_dir / "doi_verification.json")],
        metrics=payload,
    )
    return {"manuscript_text": manuscript_text, **payload}, result
