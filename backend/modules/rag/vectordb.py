from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb

from backend.config import RAG_COLLECTION_NAME, VECTOR_DB_PATH


@lru_cache(maxsize=1)
def get_collection():
    vector_db_path = Path(VECTOR_DB_PATH)
    vector_db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_db_path))
    return client.get_or_create_collection(name=RAG_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})