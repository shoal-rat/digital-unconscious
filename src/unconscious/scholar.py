"""Open scholarly search: OpenAlex first (broad, keyless), arXiv as a fallback."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from unconscious import __version__
from unconscious.text import clip, normalize_title

log = logging.getLogger(__name__)

USER_AGENT = f"digital-unconscious/{__version__} (+https://github.com/shoal-rat/digital-unconscious)"


def _get(url: str, timeout: float = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positions: dict[int, str] = {}
    for word, places in inverted.items():
        for place in places:
            positions[place] = word
    return " ".join(positions[i] for i in sorted(positions))


def openalex(query: str, per_page: int = 6) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({
        "search": query,
        "per-page": per_page,
        "select": "id,doi,display_name,publication_year,cited_by_count,primary_location,best_oa_location,"
                  "abstract_inverted_index,authorships",
    })
    data = json.loads(_get(f"https://api.openalex.org/works?{params}"))
    out = []
    for work in data.get("results") or []:
        location = work.get("primary_location") or {}
        source = location.get("source") or {}
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (work.get("authorships") or [])[:3]
        ]
        doi = work.get("doi") or ""
        free = work.get("best_oa_location") or {}
        out.append({
            "title": work.get("display_name") or "",
            "year": work.get("publication_year"),
            "venue": source.get("display_name") or "",
            "cited_by": work.get("cited_by_count") or 0,
            "authors": [a for a in authors if a],
            "url": doi or location.get("landing_page_url") or work.get("id") or "",
            "abstract": clip(_abstract(work.get("abstract_inverted_index")), 900),
            "pdf": free.get("pdf_url") or "",
            "source": "openalex",
            "query": query,
        })
    return out


def arxiv(query: str, max_results: int = 6) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode({"search_query": f"all:{query}", "max_results": max_results})
    root = ET.fromstring(_get(f"https://export.arxiv.org/api/query?{params}"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("a:entry", ns):
        published = entry.findtext("a:published", default="", namespaces=ns)
        out.append({
            "title": clip(entry.findtext("a:title", default="", namespaces=ns), 300),
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "venue": "arXiv",
            "cited_by": 0,
            "authors": [a.findtext("a:name", default="", namespaces=ns) for a in entry.findall("a:author", ns)][:3],
            "url": entry.findtext("a:id", default="", namespaces=ns),
            "abstract": clip(entry.findtext("a:summary", default="", namespaces=ns), 900),
            "pdf": entry.findtext("a:id", default="", namespaces=ns).replace("/abs/", "/pdf/"),
            "source": "arxiv",
            "query": query,
        })
    return out


def search(queries: list[str], per_query: int = 6, limit: int = 12) -> list[dict[str, Any]]:
    """Run several queries, interleave results so each query is represented, dedupe by title."""
    batches: list[list[dict[str, Any]]] = []
    for query in [q for q in queries if q.strip()][:4]:
        works: list[dict[str, Any]] = []
        try:
            works = openalex(query, per_query)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            log.info("OpenAlex failed for %r: %s", query, exc)
        if not works:
            try:
                works = arxiv(query, per_query)
            except (urllib.error.URLError, OSError, ValueError, ET.ParseError) as exc:
                log.info("arXiv failed for %r: %s", query, exc)
        batches.append(works)
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for rank in range(per_query):
        for batch in batches:
            if rank < len(batch):
                work = batch[rank]
                key = normalize_title(work["title"])
                if key and key not in seen:
                    seen.add(key)
                    merged.append(work)
    with_abstract = [w for w in merged if w["abstract"]]
    without = [w for w in merged if not w["abstract"]]
    return (with_abstract + without)[:limit]


PDF_LIMIT = 4  # full texts fetched per dive
PDF_MAX_BYTES = 40 * 1024 * 1024
PDF_SECONDS = 60  # all full texts together; the rest of the dive waits no longer


def fetch_pdfs(works: list[dict[str, Any]], folder: Path, limit: int = PDF_LIMIT,
               fetch: Callable[[str], bytes] | None = None) -> dict[int, str]:
    """Download open-access full texts into ``folder/papers``, named by each work's number
    in the list (``03.pdf`` is work [3]). Only real PDFs under the size limit are kept, the
    first ``limit`` in list order. Downloads run side by side, so one slow host costs its own
    timeout, not everyone's. Returns {number: relative path}. The folder belongs to one errand
    and goes with it."""
    shelf = folder / "papers"
    shelf.mkdir(parents=True, exist_ok=True)
    wanted = [(n, w["pdf"]) for n, w in enumerate(works, 1) if str(w.get("pdf") or "").startswith(("https://", "http://"))]
    wanted = wanted[: limit * 2]  # a few spares for dead links and paywalls

    deadline = time.monotonic() + PDF_SECONDS

    def one(url: str) -> bytes | None:
        try:
            data = (fetch or _get_pdf)(url, deadline) if fetch is None else fetch(url)
        except Exception as exc:  # paywalls, dead links, slow hosts: the abstract still stands
            log.info("could not fetch %s: %s", url, exc)
            return None
        return data if data.startswith(b"%PDF-") and len(data) <= PDF_MAX_BYTES else None

    pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="dun-pdf")
    futures = [pool.submit(one, url) for _, url in wanted]
    wait(futures, timeout=max(0.0, deadline - time.monotonic()))
    pool.shutdown(wait=False, cancel_futures=True)  # stragglers give up at the deadline on their own
    bodies = [f.result() if f.done() and not f.cancelled() else None for f in futures]
    got: dict[int, str] = {}
    for (number, _url), data in zip(wanted, bodies, strict=True):
        if data is None or len(got) >= limit:
            continue
        name = f"papers/{number:02d}.pdf"
        (folder / name).write_bytes(data)
        got[number] = name
    return got


def _get_pdf(url: str, deadline: float) -> bytes:
    """Read a PDF in chunks, giving up when the dive's download time is spent."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf"})
    chunks: list[bytes] = []
    size = 0
    with urllib.request.urlopen(req, timeout=min(15.0, max(1.0, deadline - time.monotonic()))) as resp:
        while size <= PDF_MAX_BYTES:
            if time.monotonic() > deadline:
                raise TimeoutError("out of time for full texts")
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    return b"".join(chunks)
