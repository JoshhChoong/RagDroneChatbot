"""
Heavily inspired from https://dev.to/paul_robertson_e844997d2b/build-a-rag-system-from-scratch-create-an-ai-that-answers-questions-about-your-codebase-pnb
"""

import os
import re
import json
import tiktoken
from pathlib import Path
from typing import List, Dict
from RagPipeline.embeddings import get_embedding
from RagPipeline.chroma_path import CHROMA_PERSIST_DIR
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# Boilerplate that appears at the start of Canada.ca JSON text; strip so first chunk is content-rich.
_NAV_BOILERPLATE_RE = re.compile(
    r"\s*Language selection WxT Language switcher.*?You are here Canada\.ca Transport Canada Aviation Drone safety\s+",
    re.IGNORECASE | re.DOTALL,
)


class CodebaseIngester:
    def __init__(self, gemini_api_key: str):
        persist_dir = CHROMA_PERSIST_DIR
        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        except Exception:
            try:
                self.chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
            except Exception:
                self.chroma_client = chromadb.Client()

        # create collection if it doesn't exist yet
        try:
            self.collection = self.chroma_client.get_collection(name="codebase")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name="codebase",
                metadata={"hnsw:space": "cosine"}
            )
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def read_file(self, file_path: Path) -> str:
        """Read file content with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Skip binary files
            return ""

    def _strip_canada_nav_boilerplate(self, text: str) -> str:
        """Remove Canada.ca nav block from start of text so first chunk is content-rich for retrieval."""
        if not text or not text.strip():
            return text
        return _NAV_BOILERPLATE_RE.sub(" ", text, count=1).strip()

    def chunk_text(
        self, text: str, max_tokens: int = 420, overlap_sentences: int = 1
    ) -> List[str]:
        """Split text into chunks at sentence boundaries so each chunk stays topically coherent."""
        # Split into sentences (avoid splitting on common abbreviations)
        sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
        parts = sentence_end.split(text)
        sentences = [p.strip() for p in parts if p.strip()]

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(sentences):
            chunk_sentences = []
            chunk_tokens = 0
            i = start
            while i < len(sentences):
                s = sentences[i]
                n = len(self.encoding.encode(s))
                if chunk_tokens + n > max_tokens and chunk_sentences:
                    break
                chunk_sentences.append(s)
                chunk_tokens += n
                i += 1
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))
            if i >= len(sentences):
                break
            # Overlap: next chunk starts overlap_sentences back so context carries over
            start = max(0, i - overlap_sentences)

        return chunks

    def create_embedding(self, text: str) -> List[float]:
        """Generate embedding via RagPipeline.embeddings (Gemini or local fallback)."""
        return get_embedding(text)

    def reset_collection(self):
        """Delete and recreate the codebase collection. Use before re-ingestion to avoid duplicates."""
        try:
            self.chroma_client.delete_collection(name="codebase")
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(
            name="codebase",
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_directory(
        self,
        directory_path: str,
        file_extensions: List[str] = None,
        reset_collection: bool = False,
    ):
        """Ingest all relevant files from a directory."""
        if file_extensions is None:
            file_extensions = [".json"]
        if reset_collection:
            self.reset_collection()

        directory = Path(directory_path)
        documents = []
        metadatas = []
        embeddings = []
        ids = []

        for file_path in directory.rglob('*'):
            if file_path.suffix in file_extensions and file_path.is_file():
                raw = self.read_file(file_path)
                if not raw.strip():
                    continue

                # For JSON docs with "text" + "meta.title": strip nav boilerplate, then prepend title for retrieval.
                content = raw
                source_url = None
                source_title = None
                if file_path.suffix.lower() == ".json":
                    try:
                        data = json.loads(raw)
                        if isinstance(data, dict) and "text" in data:
                            text = data.get("text") or ""
                            text = self._strip_canada_nav_boilerplate(text)
                            meta = data.get("meta") or {}
                            if isinstance(meta, dict):
                                source_title = meta.get("title")
                                source_url = meta.get("url")
                            title = source_title
                            if title:
                                content = f"Document: {title}\n\n{text}"
                            else:
                                content = text
                    except (json.JSONDecodeError, TypeError):
                        pass

                chunks = self.chunk_text(content)

                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very small chunks
                        continue

                    doc_id = f"{file_path.name}_{i}"
                    embedding = self.create_embedding(chunk)

                    meta_entry = {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "chunk_index": i,
                    }
                    if source_url:
                        meta_entry["source_url"] = source_url
                    if source_title:
                        meta_entry["source_title"] = source_title
                    documents.append(chunk)
                    metadatas.append(meta_entry)
                    embeddings.append(embedding)
                    ids.append(doc_id)

                    print(f"Processed: {file_path.name} (chunk {i})")

        # Batch insert into ChromaDB (PersistentClient writes to disk automatically)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        print(f"Ingested {len(documents)} chunks from {directory_path}")

    def close(self):
        """Release the Chroma client so the DB is flushed and the next process can read it."""
        if getattr(self.chroma_client, "close", None) and callable(self.chroma_client.close):
            self.chroma_client.close()