"""Dive: check one spark against the literature before investing in it."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unconscious import scholar
from unconscious.llm.base import LLMRequest
from unconscious.mind.dream import person_block
from unconscious.mind.prompts import DIVE_SCHEMA, DIVE_SYSTEM, DIVE_TASK, LANGUAGE_LINE, RESEARCH_DIVE
from unconscious.text import clip

if TYPE_CHECKING:
    from unconscious.app import App


STEPS = ("search", "fetch", "read", "save")


class DiveError(RuntimeError):
    pass


def paper_line(index: int, work: dict[str, Any]) -> str:
    authors = ", ".join(work.get("authors") or [])
    head = f"[{index}] {work['title']}"
    meta = " · ".join(str(x) for x in (work.get("venue"), work.get("year"), authors) if x)
    if meta:
        head += f" — {meta}"
    if work.get("cited_by"):
        head += f" · cited {work['cited_by']}×"
    abstract = clip(work.get("abstract") or "(no abstract available)", 700)
    return f"{head}\n    {abstract}"


def run_dive(
    app: App,
    spark_id: int,
    progress: Callable[[str], None] | None = None,
    *,
    search: Callable[[list[str]], list[dict[str, Any]]] | None = None,
    fetch: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    step = progress or (lambda _name: None)
    spark = app.store.spark(spark_id)
    if spark is None:
        raise DiveError(f"No spark #{spark_id}.")
    terms = list(spark.get("search_terms") or []) or [spark["title"]]

    step("search")
    papers = (search or scholar.search)(terms)
    if not papers:
        raise DiveError("The scholarly indexes returned nothing. Check the network connection and try again.")

    research = app.settings.models.research
    # One folder per dive: the full texts are read in it and leave with it.
    with tempfile.TemporaryDirectory(prefix="dun-seabed-") as folder:
        full_texts: dict[int, str] = {}
        if research:
            step("fetch")
            full_texts = scholar.fetch_pdfs(papers, Path(folder), fetch=fetch)
        step("read")
        result = app.router.call(dive_request(app, spark, papers, full_texts, research, Path(folder)))
    if not result.ok or not result.data:
        raise DiveError(result.error or "The dive model returned nothing.")

    report = result.data
    valid = set(range(1, len(papers) + 1))
    known = []
    for item in report.get("known") or []:
        refs = sorted({int(r) for r in item.get("refs") or [] if isinstance(r, int | float) and int(r) in valid})
        point = clip(str(item.get("point") or ""), 500)
        if point:
            known.append({"point": point, "refs": refs})
    report["known"] = known
    cited = sorted({r for item in known for r in item["refs"]})
    report["cited"] = cited
    report["full_texts"] = sorted(full_texts)

    step("save")
    dive_id = app.store.add_dive(spark_id, report, papers, result.label)
    if spark["status"] in {"new", "drifted"}:  # going to the seabed for a fish is choosing it
        app.store.set_spark_status(spark_id, "pursuing")
    app.store.log_event("dive", str(spark_id), {"dive_id": dive_id, "papers": len(papers)})
    return {"dive_id": dive_id, "spark_id": spark_id, "papers": len(papers), "verdict": report.get("verdict")}


def dive_request(app: App, spark: dict[str, Any], papers: list[dict[str, Any]], full_texts: dict[int, str],
                 research: bool, folder: Path) -> LLMRequest:
    language = app.settings.language
    if full_texts:
        library = "\nFULL TEXTS IN ./papers\n" + "\n".join(f"[{n}] {name}" for n, name in sorted(full_texts.items())) + "\n"
    else:
        library = ""
    return LLMRequest(
        role="dive",
        system=DIVE_SYSTEM + (RESEARCH_DIVE if research else ""),
        prompt=DIVE_TASK.format(
            person=person_block(app),
            title=spark["title"],
            question=spark["question"],
            insight=spark["insight"],
            mechanism=spark["mechanism"],
            papers="\n".join(paper_line(i, w) for i, w in enumerate(papers, 1)),
            library=library,
            language=LANGUAGE_LINE[language],
        ),
        schema=DIVE_SCHEMA,
        max_tokens=5000,
        payload={"spark": spark, "papers": papers, "language": language},
        research=research,
        workdir=folder if research else None,
    )
