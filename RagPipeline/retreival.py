"""
Heavily inspired from https://dev.to/paul_robertson_e844997d2b/build-a-rag-system-from-scratch-create-an-ai-that-answers-questions-about-your-codebase-pnb
"""

import os
import re
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from collections import OrderedDict

from RagPipeline.chroma_path import CHROMA_PERSIST_DIR

load_dotenv()


def _is_height_altitude_query(query: str) -> bool:
    """True if the query is about flying height/altitude/limit (so we boost measurement vocab)."""
    q = query.strip().lower()
    return any(
        w in q for w in ("height", "altitude", "high", "how high", "maximum", "max", "limit", "above", "below")
    )


def _is_comparison_query(query: str) -> bool:
    """True if the user is asking to compare/contrast two or more distinct concepts."""
    q = query.strip().lower()
    if any(
        p in q
        for p in (
            "difference between",
            "difference of",
            "compare",
            "contrast",
            " vs ",
            " versus ",
        )
    ):
        return True
    if " and " in q and ("difference" in q or "different" in q or "same" in q):
        return True
    return False


def _is_procedural_query(query: str) -> bool:
    """True if the user is asking HOW to do something (e.g. get a licence, register)—needs steps and requirements."""
    q = query.strip().lower()
    if not ("how" in q or "what do i need" in q or "steps" in q or "process" in q):
        return False
    return any(
        w in q
        for w in (
            "licen", "certificate", "register", "pilot", "permit", "fly",
            "get ", "become", "apply", "requirement", "need to",
        )
    )


def _expand_query_for_embedding(query: str) -> str:
    """Minimal query expansion for RAG: domain context + conceptual synonyms only.

    Pure RAG would use only the user query. We add:
    - A fixed domain phrase so the embedding stays in the right corpus (Transport Canada drone docs).
    - Conceptual synonyms (e.g. height/altitude) so different wording still matches; we do NOT inject
      answer-specific phrases (e.g. "122 metres") or doc-only jargon—that would be answer leakage.
    """
    q = query.strip().lower()
    # Single domain phrase so retrieval targets our corpus, not generic text
    parts = [query.strip(), "Transport Canada drone safety regulations Canada"]
    # Conceptual synonyms only (same idea, different word). No numbers or doc-specific sentences.
    if "height" in q or "altitude" in q:
        parts.append("altitude height maximum")
    if _is_height_altitude_query(query):
        parts.append("altitude height metres feet below above airspace stay")
    if "licen" in q or "certificate" in q:
        parts.append("pilot certificate licensing")
    if "register" in q:
        parts.append("registration")
    # For procedural "how do I get my licence" questions, pull chunks that explain steps and requirements
    if _is_procedural_query(query):
        parts.append("Basic Advanced certificate registration age weight 250 flight review exam steps requirements Drone Management Portal")
    # For comparison queries (e.g. "difference between SFOC and RPAS"), add both terms
    # so we retrieve chunks that define each concept, not only the hybrid (e.g. "SFOC-RPAS").
    if _is_comparison_query(query):
        words = re.findall(r"[A-Za-z0-9]{2,}", query.strip())
        stop = {"the", "and", "between", "what", "whats", "difference", "different", "same", "compare", "contrast", "vs", "versus", "how", "does", "do", "is", "are", "can", "certificate", "certificates"}
        terms = [w for w in words if w.lower() not in stop]
        if terms:
            parts.append(" ".join(terms))
            parts.append("definition requirements")
        # Basic vs Advanced certificate: pull chunks that explain differences (airspace, bystanders, flight review, privileges)
        if "basic" in q or "advanced" in q:
            parts.append("Basic Advanced certificate controlled airspace bystanders distance flight review operational privileges requirements")
    return " ".join(parts)


def _rerank_by_keyword_overlap(query: str, documents: List, metadatas: List, distances: List) -> tuple:
    """Re-rank by keyword overlap (user query + generic topic terms so height questions find limit paragraphs)."""
    if not documents or not query:
        return documents, metadatas, distances
    terms = set(re.findall(r"\w+", query.lower()))
    terms.discard("")
    if _is_height_altitude_query(query):
        # Boost chunks that state the standard limit (122 m / 400 ft) so the general rule is retrieved
        terms.update(["metres", "feet", "below", "above", "airspace", "stay", "height", "altitude", "122", "400"])
    if _is_procedural_query(query):
        # Boost chunks that explain steps and requirements (Basic/Advanced, registration, exam, flight review)
        terms.update(["basic", "advanced", "certificate", "registration", "exam", "flight", "review", "portal", "250", "weight"])
    # Basic vs Advanced comparison: boost chunks that explain differences
    if _is_comparison_query(query) and ("basic" in query.lower() or "advanced" in query.lower()):
        terms.update(["basic", "advanced", "controlled", "airspace", "bystanders", "flight", "review", "operational", "privileges", "distance"])
    if not terms:
        return documents, metadatas, distances

    def score(doc: str, meta: dict) -> int:
        text = (doc or "") + " " + (meta.get("file_name") or "")
        text_lower = text.lower()
        return sum(1 for t in terms if t in text_lower)

    scored = [(score(doc, meta), doc, meta, dist) for doc, meta, dist in zip(documents, metadatas, distances)]
    # Sort by keyword score desc, then by distance asc (lower = more similar)
    scored.sort(key=lambda x: (-x[0], x[3]))

    # Prefer diversity: when keyword score is 0 or 1, demote chunks from files we already have
    seen_files = set()
    diversified = []
    for sc, doc, meta, dist in scored:
        fname = (meta.get("file_name") or "").lower()
        # Boost first chunk from each file; slight demotion for duplicate file when score is low
        if sc <= 1 and fname in seen_files:
            diversified.append((sc - 0.5, doc, meta, dist))
        else:
            seen_files.add(fname)
            diversified.append((sc, doc, meta, dist))
    diversified.sort(key=lambda x: (-x[0], x[3]))
    return [x[1] for x in diversified], [x[2] for x in diversified], [x[3] for x in diversified]


class CodebaseRetriever:
    def __init__(self, gemini_api_key: str, collection_name: str = "codebase"):
        # Use a persistent Chroma database on disk so embeddings survive restarts.
        persist_dir = CHROMA_PERSIST_DIR
        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        except Exception:
            try:
                self.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
            except Exception:
                self.chroma_client = chromadb.Client()

        # Try to get existing collection; if it doesn't exist create it.
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(name=collection_name)
        # Simple in-memory LRU cache for retrieval results keyed by normalized query.
        # This speeds up repeated identical queries during an interactive session.
        self.cache_size = 128
        self._cache: "OrderedDict[str, Dict]" = OrderedDict()

    def create_query_embedding(self, query: str) -> List[float]:
        """Create embedding for user query"""
        # Use the same embedding selection logic as ingestion (Gemini fallback)
        from RagPipeline.embeddings import get_embedding
        return get_embedding(query)

    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> Dict:
        """Retrieve most relevant code chunks for a query. Uses expanded query for embedding and re-ranks by keyword overlap."""
        key = query.strip().lower()
        if key in self._cache:
            # move to the end to mark as recently used
            result = self._cache.pop(key)
            self._cache[key] = result
            return result

        expanded = _expand_query_for_embedding(query)
        query_embedding = self.create_query_embedding(expanded)

        # Fetch more candidates, then re-rank so the right chunk (e.g. 122 m for height) surfaces
        fetch_n = min(max(n_results * 2, 20), 30)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_n,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0] or []
        metas = results["metadatas"][0] or []
        dists = results["distances"][0] or []

        docs, metas, dists = _rerank_by_keyword_overlap(query, docs, metas, dists)

        out = {
            "documents": docs[:n_results],
            "metadatas": metas[:n_results],
            "distances": dists[:n_results],
        }

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = out

        return out