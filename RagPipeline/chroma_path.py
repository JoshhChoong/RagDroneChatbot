"""
Single source of truth for ChromaDB persist path so ingestion and retrieval always use the same directory across processes.
"""
import os
from pathlib import Path

# Project root = parent of RagPipeline; chroma_db lives there (absolute, resolved)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_raw = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_PERSIST_DIR = os.path.abspath(os.path.expanduser(_raw)) if _raw else str(_PROJECT_ROOT / "chroma_db")
