from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from statistics import mean
import tempfile
from typing import Any
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from du_research.agents.analysis_coder import AnalysisCoderAgent, execute_script
from du_research.models import DatasetCandidate, StageResult


def _try_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _profile_rows(rows: list[dict[str, str]], max_categorical_values: int, max_numeric_columns: int) -> dict[str, Any]:
    columns = list(rows[0].keys()) if rows else []
    numeric_columns: list[str] = []
    missing_counts = {column: 0 for column in columns}
    for column in columns:
        values = [row.get(column, "") for row in rows]
        parsed = [_try_float(value) for value in values if (value or "").strip()]
        non_empty = [value for value in values if (value or "").strip()]
        missing_counts[column] = len(values) - len(non_empty)
        if non_empty and len(parsed) >= max(1, int(0.8 * len(non_empty))):
            numeric_columns.append(column)

    numeric_summary = {}
    for column in numeric_columns[:max_numeric_columns]:
        numbers = [_try_float(row.get(column)) for row in rows]
        numbers = [value for value in numbers if value is not None]
        if not numbers:
            continue
        numeric_summary[column] = {
            "count": len(numbers),
            "mean": round(mean(numbers), 4),
            "min": round(min(numbers), 4),
            "max": round(max(numbers), 4),
        }

    categorical_summary = {}
    for column in columns:
        if column in numeric_summary:
            continue
        counts: dict[str, int] = {}
        for row in rows:
            value = (row.get(column) or "").strip()
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        if counts:
            categorical_summary[column] = sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:max_categorical_values]

    return {
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": columns,
        "missing_counts": missing_counts,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }


def _render_svg_bar_chart(title: str, values: dict[str, float]) -> str:
    width = 720
    height = 420
    chart_left = 120
    chart_top = 60
    bar_height = 40
    gap = 18
    max_value = max(values.values()) if values else 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fffaf2"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Verdana" font-size="22" fill="#1f2937">{title}</text>',
    ]
    for index, (label, value) in enumerate(values.items()):
        y = chart_top + index * (bar_height + gap)
        bar_width = 420 * (value / max_value if max_value else 0)
        lines.append(f'<text x="16" y="{y + 25}" font-family="Verdana" font-size="14" fill="#334155">{label}</text>')
        lines.append(
            f'<rect x="{chart_left}" y="{y}" width="{bar_width:.2f}" height="{bar_height}" rx="6" fill="#d97706" />'
        )
        lines.append(
            f'<text x="{chart_left + bar_width + 12:.2f}" y="{y + 25}" font-family="Verdana" font-size="13" fill="#111827">{value:.2f}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def _render_png_bar_chart(title: str, values: dict[str, float], output_path: Path, dpi: int) -> None:
    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), "#fffaf2")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2 - 180, 40), title, fill="#1f2937")
    max_value = max(values.values()) if values else 1.0
    bar_left = 320
    bar_top = 160
    bar_height = 80
    gap = 36
    for index, (label, value) in enumerate(values.items()):
        y = bar_top + index * (bar_height + gap)
        bar_width = int(1100 * (value / max_value if max_value else 0))
        draw.text((30, y + 25), label, fill="#334155")
        draw.rounded_rectangle((bar_left, y, bar_left + bar_width, y + bar_height), radius=10, fill="#d97706")
        draw.text((bar_left + bar_width + 20, y + 25), f"{value:.2f}", fill="#111827")
    image.save(output_path, dpi=(dpi, dpi))


def _render_pdf_bar_chart(title: str, values: dict[str, float], output_path: Path) -> None:
    c = canvas.Canvas(str(output_path), pagesize=letter)
    c.setTitle(title)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, title)
    max_value = max(values.values()) if values else 1.0
    y = 690
    for label, value in values.items():
        bar_width = 320 * (value / max_value if max_value else 0)
        c.setFont("Helvetica", 10)
        c.drawString(72, y + 6, label)
        c.setFillColorRGB(0.85, 0.47, 0.03)
        c.rect(220, y, bar_width, 18, fill=1, stroke=0)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(550, y + 6, f"{value:.2f}")
        y -= 32
    c.save()


def reproduce_analysis_to_directory(
    data_file: Path,
    output_dir: Path,
    max_categorical_values: int,
    max_numeric_columns: int,
    figure_dpi: int = 300,
) -> dict[str, Any]:
    with data_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    profile = _profile_rows(rows, max_categorical_values, max_numeric_columns)
    numeric_summary = profile["numeric_summary"]
    if numeric_summary:
        chart_values = {key: value["mean"] for key, value in numeric_summary.items()}
    else:
        first_category = next(iter(profile["categorical_summary"]), None)
        chart_values = {
            f"{first_category}:{label}": float(count)
            for label, count in profile["categorical_summary"].get(first_category, [])
        }
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "summary_figure.svg"
    png_path = figures_dir / "summary_figure.png"
    pdf_path = figures_dir / "summary_figure.pdf"
    if chart_values:
        figure_path.write_text(_render_svg_bar_chart("Descriptive Summary", chart_values), encoding="utf-8")
        _render_png_bar_chart("Descriptive Summary", chart_values, png_path, figure_dpi)
        _render_pdf_bar_chart("Descriptive Summary", chart_values, pdf_path)
    results = {
        "data_file": str(data_file),
        "analysis_executed": True,
        "figure_path": str(figure_path) if chart_values else None,
        "figure_png_path": str(png_path) if chart_values else None,
        "figure_pdf_path": str(pdf_path) if chart_values else None,
        **profile,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _write_processed_data(data_file: Path, output_dir: Path, profile: dict[str, Any]) -> tuple[str, str]:
    processed_dir = output_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    target_file = processed_dir / data_file.name
    shutil.copy2(data_file, target_file)
    provenance = {
        "source_file": str(data_file),
        "processed_file": str(target_file),
        "row_count": profile.get("row_count"),
        "column_count": profile.get("column_count"),
        "columns": profile.get("columns", []),
        "missing_counts": profile.get("missing_counts", {}),
    }
    provenance_path = processed_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(target_file), str(provenance_path)


def _run_repro_check(
    data_file: Path,
    max_categorical_values: int,
    max_numeric_columns: int,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        rerun = reproduce_analysis_to_directory(
            data_file=data_file,
            output_dir=Path(tmpdir),
            max_categorical_values=max_categorical_values,
            max_numeric_columns=max_numeric_columns,
            figure_dpi=300,
        )
    return {
        "passed": (
            rerun.get("row_count") == baseline.get("row_count")
            and rerun.get("column_count") == baseline.get("column_count")
            and rerun.get("numeric_summary") == baseline.get("numeric_summary")
        ),
        "baseline_row_count": baseline.get("row_count"),
        "rerun_row_count": rerun.get("row_count"),
        "baseline_column_count": baseline.get("column_count"),
        "rerun_column_count": rerun.get("column_count"),
    }


def _write_repro_script(
    script_path: Path,
    data_file: Path,
    output_dir: Path,
    max_categorical_values: int,
    max_numeric_columns: int,
) -> None:
    content = f"""from pathlib import Path
from du_research.stages.analysis import reproduce_analysis_to_directory


if __name__ == "__main__":
    reproduce_analysis_to_directory(
        data_file=Path(r"{data_file}"),
        output_dir=Path(r"{output_dir}"),
        max_categorical_values={max_categorical_values},
        max_numeric_columns={max_numeric_columns},
    )
"""
    script_path.write_text(content, encoding="utf-8")


def run_stage(
    idea_text: str,
    data_file: str | None,
    output_dir: Path,
    max_categorical_values: int,
    max_numeric_columns: int,
    feasibility: dict[str, Any],
    datasets: list[DatasetCandidate],
    figure_dpi: int = 300,
    timeout_seconds: int = 120,
    max_codegen_retries: int = 5,
    analysis_coder: AnalysisCoderAgent | None = None,
) -> tuple[dict[str, Any], StageResult]:
    plan_path = output_dir / "analysis_plan.md"
    script_path = output_dir / "reproduce_analysis.py"

    if not data_file:
        methods = feasibility.get("recommended_methods", [])
        plan_lines = [
            "# Analysis Plan",
            "",
            f"- Idea: {idea_text}",
            f"- Recommended methods: {', '.join(methods) if methods else 'descriptive-statistics'}",
            f"- Dataset candidates available: {len(datasets)}",
            "",
            "## Planned Steps",
            "",
            "1. Acquire the highest-ranked open dataset with coverage for the target outcome.",
            "2. Validate variable dictionary, missingness, and cohort/time coverage.",
            "3. Run descriptive statistics before any model fitting.",
            "4. Add inferential analysis only after confirming variable quality and sample size.",
        ]
        payload = {
            "analysis_executed": False,
            "status": "planned",
            "recommended_methods": methods,
            "dataset_candidates": len(datasets),
        }
        (output_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
        script_path.write_text(
            "# Add a local CSV path and rerun the pipeline to materialize this analysis.\n",
            encoding="utf-8",
        )
        result = StageResult(
            order=4,
            name="analysis",
            status="planned",
            summary="No local CSV supplied; generated an analysis plan instead.",
            artifacts=[str(output_dir / "results.json"), str(plan_path), str(script_path)],
            metrics={"analysis_executed": False},
        )
        return payload, result

    data_path = Path(data_file).expanduser().resolve()
    payload = reproduce_analysis_to_directory(
        data_file=data_path,
        output_dir=output_dir,
        max_categorical_values=max_categorical_values,
        max_numeric_columns=max_numeric_columns,
        figure_dpi=figure_dpi,
    )
    codegen_history = []
    if analysis_coder is not None:
        ai_dir = output_dir / "ai_codegen"
        ai_dir.mkdir(parents=True, exist_ok=True)
        data_profile = {
            "idea_text": idea_text,
            "data_file": str(data_path),
            "feasibility": feasibility,
            "baseline_profile": payload,
            "output_dir": str(ai_dir),
        }
        last_error = ""
        for attempt in range(1, max_codegen_retries + 1):
            script = analysis_coder.generate_script({**data_profile, "attempt": attempt, "last_error": last_error})
            if not script:
                codegen_history.append({"attempt": attempt, "status": "no_script"})
                continue
            script_path = ai_dir / f"analysis_attempt_{attempt}.py"
            script_path.write_text(script, encoding="utf-8")
            execution = execute_script(script_path, timeout_seconds=timeout_seconds)
            codegen_history.append({"attempt": attempt, **execution, "script_path": str(script_path)})
            if execution["returncode"] == 0:
                break
            last_error = execution["stderr"][:4000]
        (ai_dir / "codegen_history.json").write_text(
            json.dumps(codegen_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    processed_file, provenance_path = _write_processed_data(data_path, output_dir, payload)
    repro_check = _run_repro_check(
        data_file=data_path,
        max_categorical_values=max_categorical_values,
        max_numeric_columns=max_numeric_columns,
        baseline=payload,
    )
    payload["processed_file"] = processed_file
    payload["provenance_path"] = provenance_path
    payload["reproducibility_check"] = repro_check
    payload["codegen_history"] = codegen_history
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "reproducibility_check.json").write_text(
        json.dumps(repro_check, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_repro_script(
        script_path=script_path,
        data_file=data_path,
        output_dir=output_dir,
        max_categorical_values=max_categorical_values,
        max_numeric_columns=max_numeric_columns,
    )
    plan_lines = [
        "# Analysis Execution",
        "",
        f"- Data file: `{data_path}`",
        f"- Rows: {payload['row_count']}",
        f"- Columns: {payload['column_count']}",
        f"- Numeric summaries generated: {len(payload['numeric_summary'])}",
        f"- Figure generated: {'yes' if payload.get('figure_path') else 'no'}",
        f"- Reproducibility check: {'passed' if repro_check['passed'] else 'failed'}",
    ]
    plan_path.write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    artifacts = [
        str(output_dir / "results.json"),
        str(plan_path),
        str(script_path),
        str(output_dir / "reproducibility_check.json"),
        processed_file,
        provenance_path,
    ]
    if payload.get("figure_path"):
        artifacts.append(payload["figure_path"])
    if payload.get("figure_png_path"):
        artifacts.append(payload["figure_png_path"])
    if payload.get("figure_pdf_path"):
        artifacts.append(payload["figure_pdf_path"])
    if codegen_history:
        artifacts.append(str(output_dir / "ai_codegen" / "codegen_history.json"))
    result = StageResult(
        order=4,
        name="analysis",
        status="completed",
        summary=f"Executed descriptive analysis on {payload['row_count']} rows.",
        artifacts=artifacts,
        metrics={
            "analysis_executed": True,
            "row_count": payload["row_count"],
            "reproducibility_passed": repro_check["passed"],
            "codegen_attempts": len(codegen_history),
        },
    )
    return payload, result
