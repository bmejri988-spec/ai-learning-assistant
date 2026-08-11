from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import RAG_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=RAG_EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )