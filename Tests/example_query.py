import os
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv not installed or .env not present; rely on environment variables
    pass

# Ensure project root is on sys.path so local packages import correctly
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from RagPipeline.ingestion import CodebaseIngester
from RagPipeline.generation import RAGSystem
from RagPipeline.chroma_path import CHROMA_PERSIST_DIR

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in environment. Set it or add to .env file.")
    raise SystemExit(1)

# Ingest processed JSON files into Chroma (will skip files already present if IDs collide)
print("Starting ingestion of Data/files/processed ...")
ing = CodebaseIngester(GEMINI_API_KEY)
ing.ingest_directory("Data/files/processed", file_extensions=[".json"], reset_collection=True)
ing.close()  # Flush and release DB so a separate process (e.g. test_retrieval) can read it
print("Ingestion complete. ChromaDB path:", CHROMA_PERSIST_DIR)

# Run a sample query
rag = RAGSystem(GEMINI_API_KEY)
query = "how do i get my drone liscence in canada"
print(f'Query: {query}')
result = rag.generate_answer(query)

print('\n--- Answer ---')
print(result.get('answer'))
print('\n--- Sources ---')
print(json.dumps(result.get('sources', []), indent=2))
print('\nContext parts used:', result.get('context_used'))
