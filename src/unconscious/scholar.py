"""Open scholarly search: OpenAlex first (broad, keyless), arXiv as a fallback."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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
        "select": "id,doi,display_name,publication_year,cited_by_count,primary_location,abstract_inverted_index,authorships",
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
        out.append({
            "title": work.get("display_name") or "",
            "year": work.get("publication_year"),
            "venue": source.get("display_name") or "",
            "cited_by": work.get("cited_by_count") or 0,
            "authors": [a for a in authors if a],
            "url": doi or location.get("landing_page_url") or work.get("id") or "",
            "abstract": clip(_abstract(work.get("abstract_inverted_index")), 900),
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
