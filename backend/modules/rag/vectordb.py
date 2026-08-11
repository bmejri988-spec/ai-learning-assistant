from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from backend.config import RAG_COLLECTION_NAME, VECTOR_DB_PATH


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    vector_db_path = Path(VECTOR_DB_PATH)
    vector_db_path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(vector_db_path),
    )


@lru_cache(maxsize=1)
def get_collection() -> Collection:
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=RAG_COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )