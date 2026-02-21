import os
import sys
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import chromadb
from chromadb.config import Settings
from RagPipeline.chroma_path import CHROMA_PERSIST_DIR

persist_dir = CHROMA_PERSIST_DIR
print(f"Checking ChromaDB at: {persist_dir}")

try:
    chroma_client = chromadb.PersistentClient(path=persist_dir)
except Exception:
    try:
        chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
    except Exception:
        chroma_client = chromadb.Client()

try:
    collection = chroma_client.get_collection(name="codebase")
    count = collection.count()
    print(f"\nCollection 'codebase' found with {count} documents")
    
    if count > 0:
        # Get a sample
        sample = collection.get(limit=3)
        print(f"\nSample document IDs: {sample['ids'][:3]}")
        print(f"\nSample metadata:")
        for i, meta in enumerate(sample['metadatas'][:2]):
            print(f"  {i+1}. {meta}")
    else:
        print("\nCollection is empty!")
        
except Exception as e:
    print(f"\nError accessing collection: {e}")
    print("\nAvailable collections:")
    try:
        collections = chroma_client.list_collections()
        for col in collections:
            print(f"  - {col.name}")
    except Exception as e2:
        print(f"  Could not list collections: {e2}")

