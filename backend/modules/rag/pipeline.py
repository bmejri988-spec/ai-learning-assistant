from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from backend.modules.rag.embeddings import get_embedding_model
from backend.modules.rag.loader import load_pdf_text
from backend.modules.rag.splitter import split_text
from backend.modules.rag.vectordb import get_collection


class RagPipeline:
    def extract_text(self, pdf_path: Path) -> str:
        return load_pdf_text(pdf_path)

    def chunk_text(self, text: str):
        return split_text(text)

    def embed_chunks(self, chunks: list[str]):
        embeddings = get_embedding_model().embed_documents(chunks)
        return embeddings

    def store_chunks(self, document_name: str, chunks: list[str], embeddings: list[list[float]]) -> int:
        collection = get_collection()
        ids = [f"{Path(document_name).stem}-{uuid4().hex[:12]}-{index}" for index in range(len(chunks))]
        metadatas = [{"document_name": document_name, "chunk_index": index} for index in range(len(chunks))]
        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        return collection.count()

    def index_pdf(self, pdf_path: Path) -> dict[str, object]:
        text = self.extract_text(pdf_path)
        split_result = self.chunk_text(text)

        if not split_result.chunks:
            raise HTTPException(status_code=400, detail="The PDF does not contain readable text")

        embeddings = self.embed_chunks(split_result.chunks)
        collection_count = self.store_chunks(pdf_path.name, split_result.chunks, embeddings)

        embedding_dim = len(embeddings[0]) if embeddings else 0
        return {
            "document_name": pdf_path.name,
            "extracted_text_length": len(text),
            "chunks": split_result.count,
            "average_chunk_size": round(split_result.average_chunk_size, 2),
            "embedding_dimension": embedding_dim,
            "vector_count": len(embeddings),
            "stored_documents": collection_count,
        }


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RagPipeline:
    return RagPipeline()