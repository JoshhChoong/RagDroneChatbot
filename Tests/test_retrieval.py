import os
import json
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure project root is on sys.path so local packages import correctly
import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from RagPipeline.retreival import CodebaseRetriever
from RagPipeline.chroma_path import CHROMA_PERSIST_DIR

# Get API key (needed for retriever initialization; embeddings use env in get_embedding)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "dummy-key")

# Test retrieval (no LLM call, only embedding + Chroma)
print("Testing RAG Retrieval System...")
print("ChromaDB path:", CHROMA_PERSIST_DIR, "\n")

retriever = CodebaseRetriever(GEMINI_API_KEY)

# Ensure DB was populated (ingestion uses same default path: project root / chroma_db)
try:
    n_docs = retriever.collection.count()
except Exception:
    n_docs = 0
if n_docs == 0:
    print("No documents in ChromaDB. Run ingestion first from the project root:")
    print("  python Tests/example_query.py")
    print("Or: ingest_directory('Data/files/processed', reset_collection=True)")
    sys.exit(1)
print(f"ChromaDB has {n_docs} chunks.\n")

# Test queries
test_queries = [
    'How do I register a drone?',
    'What are the requirements for advanced drone certification in Canada?',
    'Can I fly at night?',
    'What is the maximum legal altitude in Canada?'
]

for query in test_queries:
    print(f'Query: {query}')
    print('-' * 60)
    
    try:
        result = retriever.retrieve_relevant_chunks(query, n_results=3)
        
        print(f'Found {len(result["documents"])} relevant chunks:\n')
        
        for i, (doc, metadata, distance) in enumerate(zip(
            result["documents"], 
            result["metadatas"], 
            result["distances"]
        ), 1):
            print(f'Chunk {i} (distance: {distance:.4f}):')
            print(f'  Source: {metadata.get("file_name", "unknown")}')
            print(f'  Preview: {doc[:200]}...' if len(doc) > 200 else f'  Content: {doc}')
            print()
        
        print('=' * 60)
        print()
        
    except Exception as e:
        print(f'Error: {e}\n')
        print('=' * 60)
        print()

print('Retrieval test complete!')

