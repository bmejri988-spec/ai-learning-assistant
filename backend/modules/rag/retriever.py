from __future__ import annotations

from functools import lru_cache
from typing import Any

from backend.config import RAG_DEFAULT_TOP_K
from backend.modules.rag.embeddings import get_embedding_model
from backend.modules.rag.vectordb import get_collection


class RagRetriever:
    def retrieve(
        self,
        query: str,
        top_k: int = RAG_DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        query = query.strip()

        if not query:
            return {
                "query": query,
                "top_k": top_k,
                "documents": [],
            }

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        collection = get_collection()

        if collection.count() == 0:
            return {
                "query": query,
                "top_k": top_k,
                "documents": [],
            }

        query_embedding = get_embedding_model().embed_query(query)

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        doc_list = result.get("documents", [[]])[0] or []
        metadata_list = result.get("metadatas", [[]])[0] or []
        distance_list = result.get("distances", [[]])[0] or []

        documents: list[dict[str, Any]] = []

        for index, text in enumerate(doc_list):
            if not text:
                continue

            distance = (
                distance_list[index]
                if index < len(distance_list)
                else None
            )

            metadata = (
                metadata_list[index]
                if index < len(metadata_list)
                else {}
            )

            # Chroma is configured with cosine distance.
            # For cosine distance:
            #   similarity = 1 - distance
            score = (
                round(1.0 - float(distance), 4)
                if distance is not None
                else None
            )

            documents.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "distance": distance,
                    "score": score,
                }
            )

        return {
            "query": query,
            "top_k": top_k,
            "documents": documents,
        }


@lru_cache(maxsize=1)
def get_rag_retriever() -> RagRetriever:
    return RagRetriever()