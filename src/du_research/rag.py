"""RAG knowledge base — ChromaDB vector store for semantic retrieval.

Stores document chunks (paper abstracts, domain knowledge, user notes) as
embeddings and retrieves the most relevant context for idea generation.

Falls back to file-based keyword matching when ChromaDB is not installed.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import ChromaDB; fall back gracefully
# ---------------------------------------------------------------------------

_CHROMA_AVAILABLE = False
try:
    import chromadb  # type: ignore[import-untyped]
    from chromadb.config import Settings as ChromaSettings  # type: ignore[import-untyped]
    _CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment,misc]


def chroma_available() -> bool:
    return _CHROMA_AVAILABLE


# ---------------------------------------------------------------------------
# ChromaDB-backed RAG store
# ---------------------------------------------------------------------------


class RAGStore:
    """Local vector knowledge base powered by ChromaDB.

    If ChromaDB is not installed, all methods degrade to file-based keyword
    search so the rest of the pipeline still works.
    """

    def __init__(self, workspace_dir: Path, collection_name: str = "knowledge",
                 force_file_mode: bool = False):
        self.workspace_dir = workspace_dir
        self.collection_name = collection_name
        self._collection = None
        self._client = None
        self._fallback_docs: list[dict[str, Any]] | None = None

        if _CHROMA_AVAILABLE and not force_file_mode:
            persist_dir = workspace_dir / "chromadb"
            persist_dir.mkdir(parents=True, exist_ok=True)
            try:
                self._client = chromadb.PersistentClient(
                    path=str(persist_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(
                    "ChromaDB collection '%s' loaded (%d documents)",
                    collection_name,
                    self._collection.count(),
                )
            except Exception as exc:
                logger.warning("ChromaDB init failed, falling back to file mode: %s", exc)
                self._collection = None
                self._client = None

    @property
    def is_vector_mode(self) -> bool:
        return self._collection is not None

    # ------------------------------------------------------------------
    # Add documents
    # ------------------------------------------------------------------

    def add_paper(
        self,
        title: str,
        abstract: str,
        *,
        doi: str | None = None,
        source: str = "literature",
        domain: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a paper to the knowledge base. Returns the document id."""
        doc_id = self._doc_id(doi or title)
        text = f"{title}\n\n{abstract}".strip()
        if not text:
            return doc_id

        meta = {
            "source": source,
            "domain": domain,
            "doi": doi or "",
            "title": title,
            **(metadata or {}),
        }

        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[meta],
                )
            except Exception as exc:
                logger.warning("ChromaDB upsert failed: %s", exc)
                self._fallback_add(doc_id, text, meta)
        else:
            self._fallback_add(doc_id, text, meta)

        return doc_id

    def add_text(
        self,
        text: str,
        *,
        doc_id: str | None = None,
        source: str = "user",
        domain: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add arbitrary text to the knowledge base."""
        doc_id = doc_id or self._doc_id(text)
        meta = {"source": source, "domain": domain, **(metadata or {})}

        if self._collection is not None:
            try:
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[text],
                    metadatas=[meta],
                )
            except Exception as exc:
                logger.warning("ChromaDB upsert failed: %s", exc)
                self._fallback_add(doc_id, text, meta)
        else:
            self._fallback_add(doc_id, text, meta)

        return doc_id

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        domain_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant documents for *query_text*."""
        if self._collection is not None and self._collection.count() > 0:
            return self._chroma_query(query_text, n_results, domain_filter)
        return self._fallback_query(query_text, n_results, domain_filter)

    def query_as_context(
        self,
        query_text: str,
        n_results: int = 8,
        domain_filter: str | None = None,
    ) -> str | None:
        """Return formatted RAG context string for prompt injection."""
        results = self.query(query_text, n_results, domain_filter)
        if not results:
            return None

        parts: list[str] = []
        for doc in results:
            title = doc.get("title") or doc.get("metadata", {}).get("title", "")
            text = doc.get("text", "")[:400]
            source = doc.get("metadata", {}).get("source", "")
            if title:
                parts.append(f"### {title} ({source})\n{text}")
            else:
                parts.append(f"### [{source}]\n{text}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def add_papers_from_run(self, run_dir: Path) -> int:
        """Ingest papers from a completed research run into the knowledge base."""
        papers_path = run_dir / "01_literature" / "papers.json"
        if not papers_path.exists():
            return 0
        try:
            data = json.loads(papers_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0

        count = 0
        for paper in data.get("papers", []):
            title = paper.get("title", "")
            summary = paper.get("summary", "")
            if title and summary:
                self.add_paper(
                    title,
                    summary,
                    doi=paper.get("doi"),
                    source=paper.get("source", "literature"),
                    metadata={
                        "year": paper.get("year"),
                        "score": paper.get("score"),
                        "url": paper.get("url", ""),
                    },
                )
                count += 1
        return count

    def add_knowledge_files(self, knowledge_dir: Path) -> int:
        """Ingest user knowledge documents (txt, md, json) from a directory."""
        if not knowledge_dir.exists():
            return 0
        count = 0
        for path in sorted(knowledge_dir.iterdir()):
            if path.suffix.lower() in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    self.add_text(text, doc_id=self._doc_id(str(path)), source="user_document",
                                  metadata={"filename": path.name})
                    count += 1
            elif path.suffix.lower() == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        for item in data:
                            text = item.get("text", item.get("content", ""))
                            if text:
                                self.add_text(text, source="user_document",
                                              metadata={"filename": path.name})
                                count += 1
                except (json.JSONDecodeError, OSError):
                    pass
        return count

    def count(self) -> int:
        if self._collection is not None:
            return self._collection.count()
        docs = self._load_fallback_docs()
        return len(docs)

    # ------------------------------------------------------------------
    # Internal: ChromaDB
    # ------------------------------------------------------------------

    def _chroma_query(
        self,
        query_text: str,
        n_results: int,
        domain_filter: str | None,
    ) -> list[dict[str, Any]]:
        where = {"domain": domain_filter} if domain_filter else None
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self._collection.count()),
                where=where if where and self._collection.count() > 0 else None,
            )
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return self._fallback_query(query_text, n_results, domain_filter)

        output: list[dict[str, Any]] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        for i, doc in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 1.0
            output.append({
                "text": doc,
                "metadata": meta,
                "title": meta.get("title", ""),
                "score": round(1.0 - dist, 4),
            })
        return output

    # ------------------------------------------------------------------
    # Internal: file-based fallback
    # ------------------------------------------------------------------

    def _fallback_path(self) -> Path:
        return self.workspace_dir / "knowledge" / "rag_documents.jsonl"

    def _load_fallback_docs(self) -> list[dict[str, Any]]:
        if self._fallback_docs is not None:
            return self._fallback_docs
        path = self._fallback_path()
        if not path.exists():
            self._fallback_docs = []
            return self._fallback_docs
        docs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self._fallback_docs = docs
        return docs

    def _fallback_add(self, doc_id: str, text: str, meta: dict[str, Any]) -> None:
        path = self._fallback_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"id": doc_id, "text": text, "metadata": meta}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Invalidate cache
        self._fallback_docs = None

    def _fallback_query(
        self,
        query_text: str,
        n_results: int,
        domain_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Simple keyword overlap scoring when ChromaDB is not available."""
        docs = self._load_fallback_docs()
        query_tokens = set(query_text.lower().split())

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in docs:
            if domain_filter:
                doc_domain = doc.get("metadata", {}).get("domain", "")
                if doc_domain and doc_domain != domain_filter:
                    continue
            doc_text = doc.get("text", "")
            doc_tokens = set(doc_text.lower().split())
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            scored.append((overlap, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "title": doc.get("metadata", {}).get("title", ""),
                "score": round(score, 4),
            }
            for score, doc in scored[:n_results]
            if score > 0
        ]

    @staticmethod
    def _doc_id(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]
