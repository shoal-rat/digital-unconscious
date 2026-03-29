from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree as ET

from du_research.models import PaperCandidate, StageResult
from du_research.net import build_url, fetch_bytes, fetch_json
from du_research.utils import citation_score, overlap_score, recentness_score, safe_year, slugify, strip_html

# Optional AI backend for Claude Code computer-use browsing
_AI_BACKEND = None  # Set by caller via run_stage(backend=...)


class LiteratureProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[PaperCandidate]:
        ...


@dataclass
class ArxivProvider:
    timeout: int = 15
    name: str = "arxiv"

    def search(self, query: str, max_results: int) -> list[PaperCandidate]:
        url = build_url(
            "https://export.arxiv.org/api/query",
            {"search_query": f'all:"{query}"', "start": 0, "max_results": max_results},
        )
        payload = fetch_bytes(url, timeout=self.timeout)
        root = ET.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers: list[PaperCandidate] = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            year = safe_year(entry.findtext("atom:published", default="", namespaces=ns))
            authors = [
                author.findtext("atom:name", default="", namespaces=ns).strip()
                for author in entry.findall("atom:author", ns)
            ]
            pdf_url = None
            url_value = entry.findtext("atom:id", default="", namespaces=ns)
            for link in entry.findall("atom:link", ns):
                href = link.attrib.get("href")
                link_type = link.attrib.get("type", "")
                if href and "pdf" in link_type:
                    pdf_url = href
            paper = PaperCandidate(
                source=self.name,
                title=title,
                summary=summary,
                authors=[name for name in authors if name],
                year=year,
                url=url_value,
                pdf_url=pdf_url,
                subjects=[category.attrib.get("term", "") for category in entry.findall("atom:category", ns)],
            )
            paper.score = round(
                (0.6 * overlap_score(query, f"{title} {summary}"))
                + (0.25 * recentness_score(year))
                + (0.15 * (1.0 if pdf_url else 0.2)),
                4,
            )
            papers.append(paper)
        return papers


@dataclass
class CrossrefProvider:
    timeout: int = 15
    name: str = "crossref"

    def search(self, query: str, max_results: int) -> list[PaperCandidate]:
        url = build_url(
            "https://api.crossref.org/works",
            {
                "query.bibliographic": query,
                "rows": max_results,
                "select": "DOI,title,author,issued,URL,abstract,subject,is-referenced-by-count",
            },
        )
        payload = fetch_json(url, timeout=self.timeout)
        items = payload.get("message", {}).get("items", [])
        papers: list[PaperCandidate] = []
        for item in items:
            title_list = item.get("title") or []
            title = title_list[0].strip() if title_list else ""
            authors = []
            for author in item.get("author", []):
                given = author.get("given", "").strip()
                family = author.get("family", "").strip()
                full_name = " ".join(part for part in [given, family] if part)
                if full_name:
                    authors.append(full_name)
            date_parts = item.get("issued", {}).get("date-parts", [])
            year = safe_year(date_parts[0][0]) if date_parts and date_parts[0] else None
            summary = strip_html(item.get("abstract")) or " ".join(item.get("subject", [])[:4])
            citation_count = int(item.get("is-referenced-by-count") or 0)
            paper = PaperCandidate(
                source=self.name,
                title=title,
                summary=summary,
                authors=authors,
                year=year,
                url=item.get("URL", ""),
                doi=item.get("DOI"),
                citation_count=citation_count,
                subjects=item.get("subject", []),
            )
            paper.score = round(
                (0.45 * overlap_score(query, f"{title} {summary}"))
                + (0.25 * recentness_score(year))
                + (0.30 * citation_score(citation_count)),
                4,
            )
            papers.append(paper)
        return papers


@dataclass
class SemanticScholarProvider:
    timeout: int = 15
    name: str = "semantic_scholar"

    def search(self, query: str, max_results: int) -> list[PaperCandidate]:
        url = build_url(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            {
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,year,authors,url,citationCount,openAccessPdf,fieldsOfStudy,externalIds",
            },
        )
        payload = fetch_json(url, timeout=self.timeout)
        papers: list[PaperCandidate] = []
        for item in payload.get("data", []):
            pdf = item.get("openAccessPdf") or {}
            paper = PaperCandidate(
                source=self.name,
                title=item.get("title", ""),
                summary=strip_html(item.get("abstract", "")),
                authors=[author.get("name", "") for author in item.get("authors", []) if author.get("name")],
                year=safe_year(item.get("year")),
                url=item.get("url", ""),
                pdf_url=pdf.get("url"),
                doi=(item.get("externalIds") or {}).get("DOI"),
                citation_count=int(item.get("citationCount") or 0),
                subjects=item.get("fieldsOfStudy", []) or [],
            )
            paper.score = round(
                (0.45 * overlap_score(query, f"{paper.title} {paper.summary}"))
                + (0.25 * recentness_score(paper.year))
                + (0.20 * citation_score(paper.citation_count))
                + (0.10 * (1.0 if paper.pdf_url else 0.2)),
                4,
            )
            papers.append(paper)
        return papers


@dataclass
class PubMedProvider:
    timeout: int = 15
    name: str = "pubmed"

    def search(self, query: str, max_results: int) -> list[PaperCandidate]:
        search_url = build_url(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            {"db": "pubmed", "term": query, "retmode": "json", "retmax": max_results},
        )
        search_payload = fetch_json(search_url, timeout=self.timeout)
        ids = search_payload.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetch_url = build_url(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        )
        payload = fetch_bytes(fetch_url, timeout=self.timeout)
        root = ET.fromstring(payload)
        papers: list[PaperCandidate] = []
        for article in root.findall(".//PubmedArticle"):
            article_meta = article.find(".//Article")
            if article_meta is None:
                continue
            title = "".join(article_meta.findtext("ArticleTitle", default="")).strip()
            abstract_parts = [
                "".join(part.itertext()).strip()
                for part in article_meta.findall(".//Abstract/AbstractText")
                if "".join(part.itertext()).strip()
            ]
            summary = " ".join(abstract_parts)
            authors = []
            for author in article_meta.findall(".//Author"):
                last = author.findtext("LastName", default="").strip()
                fore = author.findtext("ForeName", default="").strip()
                full_name = " ".join(part for part in [fore, last] if part)
                if full_name:
                    authors.append(full_name)
            year = safe_year(
                article.findtext(".//PubDate/Year", default="")
                or article.findtext(".//ArticleDate/Year", default="")
            )
            pmid = article.findtext(".//PMID", default="")
            doi = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.attrib.get("IdType") == "doi":
                    doi = (article_id.text or "").strip()
                    break
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            paper = PaperCandidate(
                source=self.name,
                title=title,
                summary=summary,
                authors=authors,
                year=year,
                url=url,
                doi=doi or None,
                subjects=["pubmed"],
            )
            paper.score = round(
                (0.55 * overlap_score(query, f"{paper.title} {paper.summary}"))
                + (0.25 * recentness_score(paper.year))
                + (0.20 * (1.0 if paper.summary else 0.2)),
                4,
            )
            papers.append(paper)
        return papers


METHOD_KEYWORDS = {
    "regression": {"regression", "ols", "linear", "logistic", "model"},
    "survey": {"survey", "questionnaire", "self-report"},
    "experiment": {"experiment", "randomized", "trial", "intervention"},
    "observational": {"observational", "cohort", "usage", "behavioral"},
    "qualitative": {"interview", "qualitative", "thematic"},
}


DATASET_KEYWORDS = {"dataset", "data", "records", "survey", "cohort", "logs", "corpus"}


def _enrich_paper(paper: PaperCandidate) -> PaperCandidate:
    text = f"{paper.title}. {paper.summary}".strip()
    lower = text.lower()
    claims = []
    if paper.title:
        claims.append(paper.title)
    if paper.summary:
        claims.append(paper.summary.split(".")[0].strip())
    methods = [name for name, words in METHOD_KEYWORDS.items() if any(word in lower for word in words)]
    findings = []
    if paper.summary:
        sentences = [part.strip() for part in paper.summary.split(".") if part.strip()]
        findings.extend(sentences[:2])
    datasets_used = []
    if any(word in lower for word in DATASET_KEYWORDS):
        for sentence in [part.strip() for part in paper.summary.split(".") if part.strip()]:
            if any(word in sentence.lower() for word in DATASET_KEYWORDS):
                datasets_used.append(sentence[:140])
    paper.claims = claims[:2]
    paper.methods = methods or ["unspecified"]
    paper.findings = findings[:2]
    paper.datasets_used = datasets_used[:2]
    return paper


def _dedupe(papers: list[PaperCandidate]) -> list[PaperCandidate]:
    deduped: dict[str, PaperCandidate] = {}
    for paper in papers:
        key = (paper.doi or paper.title).lower().strip()
        existing = deduped.get(key)
        if existing is None or paper.score > existing.score:
            deduped[key] = paper
    return sorted(deduped.values(), key=lambda item: item.score, reverse=True)


def _summary_markdown(query: str, papers: list[PaperCandidate], errors: list[str], core_count: int) -> str:
    lines = [
        "# Literature Summary",
        "",
        f"- Query: `{query}`",
        f"- Papers ranked: {len(papers)}",
        f"- Core references: {min(core_count, len(papers))}",
    ]
    if errors:
        lines.append(f"- Provider errors: {len(errors)}")
    lines.extend(["", "## Top Papers", ""])
    for index, paper in enumerate(papers[:10], start=1):
        author_text = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
        lines.append(
            f"{index}. **{paper.title}** ({paper.year or 'n.d.'}, {paper.source}, score {paper.score:.2f})"
        )
        lines.append(f"   - Authors: {author_text}")
        if paper.summary:
            lines.append(f"   - Summary: {paper.summary[:280]}")
        lines.append(f"   - URL: {paper.url}")
    if errors:
        lines.extend(["", "## Provider Errors", ""])
        for error in errors:
            lines.append(f"- {error}")
    return "\n".join(lines) + "\n"


def _download_open_pdfs(
    papers: list[PaperCandidate],
    output_dir: Path,
    timeout: int,
    max_pdf_downloads: int,
) -> tuple[list[dict[str, str]], list[str]]:
    downloads = []
    errors = []
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for index, paper in enumerate(papers):
        if len(downloads) >= max_pdf_downloads:
            break
        pdf_url = paper.pdf_url
        if not pdf_url and paper.url.endswith(".pdf"):
            pdf_url = paper.url
        if not pdf_url:
            continue
        try:
            data = fetch_bytes(pdf_url, timeout=timeout, headers={"Accept": "application/pdf,*/*"})
            filename = f"{index + 1:02d}_{slugify(paper.title, max_length=40)}.pdf"
            target = pdf_dir / filename
            target.write_bytes(data)
            downloads.append({"title": paper.title, "pdf_url": pdf_url, "path": str(target)})
        except Exception as exc:
            errors.append(f"{paper.title}: {exc}")
    return downloads, errors


def _browse_and_download_papers(
    papers: list[PaperCandidate],
    output_dir: Path,
    backend: object,
    max_papers: int = 5,
) -> tuple[list[dict[str, str]], list[str]]:
    """Use Claude Code computer-use to browse paper pages and download PDFs.

    This calls Claude Code with browser/computer tools enabled, instructing it
    to visit each paper URL, read the abstract, and download the PDF — exactly
    like a human researcher would.
    """
    import json as _json
    downloads: list[dict[str, str]] = []
    errors: list[str] = []
    pdf_dir = output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for paper in papers[:max_papers]:
        url = paper.pdf_url or paper.url
        if not url:
            continue

        prompt = (
            f"Browse to this academic paper and download the PDF.\n\n"
            f"Title: {paper.title}\n"
            f"URL: {url}\n"
            f"DOI: {paper.doi or 'N/A'}\n\n"
            f"Instructions:\n"
            f"1. Open the URL in Chrome\n"
            f"2. If it's a PDF, download it to: {pdf_dir}\n"
            f"3. If it's a landing page, find and click the PDF download link\n"
            f"4. If there's a 'Download PDF' or 'View PDF' button, click it\n"
            f"5. For arXiv, append .pdf to the abstract URL\n"
            f"6. Save the file as: {slugify(paper.title, max_length=40)}.pdf\n"
            f"7. If the PDF is behind a paywall, try Unpaywall or Sci-Hub alternatives\n\n"
            f"Proceed without stopping. Do not ask for permission."
        )
        try:
            response = backend.call(
                prompt,
                mode="strict",
                model="sonnet",
                allowed_tools=["computer", "bash"],
                max_tokens=3000,
            )
            if response.ok:
                downloads.append({
                    "title": paper.title,
                    "url": url,
                    "method": "claude_code_browser",
                    "response": response.text[:500],
                })
            else:
                errors.append(f"Claude Code browse failed for {paper.title}: {response.raw.get('error', 'unknown')}")
        except Exception as exc:
            errors.append(f"Claude Code browse error for {paper.title}: {exc}")

    return downloads, errors


def _extract_paper_content(
    papers: list[PaperCandidate],
    output_dir: Path,
    backend: object,
    max_papers: int = 3,
) -> list[dict[str, str]]:
    """Use Claude Code to read downloaded PDFs and extract structured findings."""
    import json as _json
    extracted: list[dict[str, str]] = []
    pdf_dir = output_dir / "pdfs"
    if not pdf_dir.exists():
        return extracted

    pdf_files = list(pdf_dir.glob("*.pdf"))[:max_papers]
    for pdf_path in pdf_files:
        prompt = (
            f"Read this PDF file and extract structured information.\n\n"
            f"File: {pdf_path}\n\n"
            f"Extract and return ONLY JSON:\n"
            f'{{"title": "...", "abstract": "...", "key_findings": ["..."], '
            f'"methods": ["..."], "datasets_used": ["..."], "limitations": ["..."]}}'
        )
        try:
            response = backend.call(
                prompt,
                mode="strict",
                model="sonnet",
                allowed_tools=["Read"],
                max_tokens=2000,
            )
            if response.ok:
                try:
                    data = _json.loads(response.text)
                    extracted.append(data)
                except _json.JSONDecodeError:
                    # Try to find JSON in response
                    text = response.text
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        try:
                            extracted.append(_json.loads(text[start:end]))
                        except _json.JSONDecodeError:
                            pass
        except Exception:
            pass

    return extracted


def _llm_rank_papers(
    papers: list[PaperCandidate],
    idea_text: str,
    backend: object,
    top_n: int = 30,
) -> list[PaperCandidate]:
    """Use LLM to rank papers by relevance to the research idea."""
    if not papers:
        return papers
    paper_summaries = []
    for i, p in enumerate(papers[:top_n]):
        paper_summaries.append({
            "index": i,
            "title": p.title,
            "year": p.year,
            "summary": (p.summary or "")[:150],
            "source": p.source,
            "citation_count": p.citation_count,
        })

    prompt = (
        f"Rank these {len(paper_summaries)} papers by relevance to the research idea:\n"
        f'"{idea_text}"\n\n'
        f"Papers:\n{json.dumps(paper_summaries, indent=2, ensure_ascii=False)}\n\n"
        f"Return ONLY a JSON array of indices in order of relevance (most relevant first).\n"
        f"Also assign a relevance score 0-100 to each.\n"
        f"Format: [{{\"index\": 0, \"score\": 85}}, {{\"index\": 3, \"score\": 72}}, ...]"
    )
    try:
        response = backend.call(prompt, mode="strict", model="sonnet", max_tokens=1000)
        if response.ok:
            text = response.text
            try:
                rankings = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    rankings = json.loads(text[start:end])
                else:
                    return papers[:top_n]

            # Apply LLM scores
            for entry in rankings:
                idx = entry.get("index", -1)
                if 0 <= idx < len(papers):
                    papers[idx].score = round(entry.get("score", 50) / 100, 4)

            # Sort by score
            return sorted(papers[:top_n], key=lambda p: p.score, reverse=True)
    except Exception:
        pass
    return papers[:top_n]


def _llm_enrich_papers(
    papers: list[PaperCandidate],
    idea_text: str,
    backend: object,
    max_papers: int = 10,
) -> list[PaperCandidate]:
    """Use LLM to extract structured claims, methods, findings from papers."""
    if not papers:
        return papers
    paper_data = []
    for p in papers[:max_papers]:
        paper_data.append({
            "title": p.title,
            "summary": (p.summary or "")[:300],
        })

    prompt = (
        f"For each paper below, extract structured research metadata.\n"
        f"Research idea context: \"{idea_text}\"\n\n"
        f"Papers:\n{json.dumps(paper_data, indent=2, ensure_ascii=False)}\n\n"
        f"Return ONLY JSON array:\n"
        f'[{{"claims": ["..."], "methods": ["..."], "findings": ["..."], "datasets_used": ["..."]}}]'
    )
    try:
        response = backend.call(prompt, mode="strict", model="haiku", max_tokens=1500)
        if response.ok:
            text = response.text
            try:
                enrichments = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("[")
                end = text.rfind("]") + 1
                if start >= 0 and end > start:
                    enrichments = json.loads(text[start:end])
                else:
                    return papers

            for i, enrichment in enumerate(enrichments):
                if i < len(papers):
                    papers[i].claims = enrichment.get("claims", [])[:3]
                    papers[i].methods = enrichment.get("methods", [])[:3]
                    papers[i].findings = enrichment.get("findings", [])[:3]
                    papers[i].datasets_used = enrichment.get("datasets_used", [])[:3]
    except Exception:
        pass
    return papers


def run_stage(
    idea_text: str,
    output_dir: Path,
    max_results_per_source: int,
    core_papers: int,
    timeout: int,
    download_pdfs: bool = True,
    max_pdf_downloads: int = 3,
    dry_run: bool = False,
    providers: list[LiteratureProvider] | None = None,
    backend: object | None = None,
) -> tuple[list[PaperCandidate], dict, StageResult]:
    providers = providers if providers is not None else [
        ArxivProvider(timeout),
        CrossrefProvider(timeout),
        SemanticScholarProvider(timeout),
        PubMedProvider(timeout),
    ]
    errors: list[str] = []
    found: list[PaperCandidate] = []
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

    deduped = _dedupe(found)

    # Use LLM for ranking and enrichment when available
    if backend is not None and deduped:
        ranked = _llm_rank_papers(deduped, idea_text, backend, top_n=30)
        ranked = _llm_enrich_papers(ranked, idea_text, backend, max_papers=10)
    else:
        ranked = [_enrich_paper(paper) for paper in deduped]
    downloads: list[dict[str, str]] = []
    download_errors: list[str] = []
    browser_downloads: list[dict[str, str]] = []
    extracted_content: list[dict[str, str]] = []
    if download_pdfs and not dry_run:
        # First try direct HTTP downloads for open-access papers
        downloads, download_errors = _download_open_pdfs(ranked, output_dir, timeout, max_pdf_downloads)
        # Then use Claude Code computer-use to browse and download remaining papers
        if backend is not None:
            remaining = [p for p in ranked if not any(d["title"] == p.title for d in downloads)]
            browser_downloads, browser_errors = _browse_and_download_papers(
                remaining, output_dir, backend, max_papers=max_pdf_downloads,
            )
            download_errors.extend(browser_errors)
            # Extract structured content from downloaded PDFs
            extracted_content = _extract_paper_content(ranked, output_dir, backend, max_papers=3)
    payload = {
        "query": idea_text,
        "paper_count": len(ranked),
        "provider_counts": provider_counts,
        "errors": errors,
        "downloaded_pdfs": downloads,
        "browser_downloaded": browser_downloads,
        "extracted_content": extracted_content,
        "download_errors": download_errors,
        "core_reference_titles": [paper.title for paper in ranked[:core_papers]],
        "papers": [paper.to_dict() for paper in ranked],
    }
    papers_path = output_dir / "papers.json"
    summary_path = output_dir / "summary.md"
    papers_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path.write_text(_summary_markdown(idea_text, ranked, errors, core_papers), encoding="utf-8")
    summary = f"Ranked {len(ranked)} literature candidates from {len(providers)} provider(s)."
    if errors and not ranked:
        status = "degraded"
        summary = f"No literature results returned. Provider errors: {len(errors)}."
    elif errors:
        status = "degraded"
    else:
        status = "completed"
    result = StageResult(
        order=1,
        name="literature",
        status=status,
        summary=summary,
        artifacts=[str(papers_path), str(summary_path)],
        metrics={
            "paper_count": len(ranked),
            "provider_counts": provider_counts,
            "downloaded_pdf_count": len(downloads),
        },
    )
    return ranked, payload, result
