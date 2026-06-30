from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from fastapi import HTTPException
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from backend.config import (
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_COLLECTION_NAME,
    RAG_EMBEDDING_MODEL,
    VECTOR_DB_PATH,
)


@dataclass(slots=True)
class RagIngestionResult:
    document_name: str
    chunks: int
    collection_name: str
    vector_db_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_name": self.document_name,
            "chunks": self.chunks,
            "collection_name": self.collection_name,
            "vector_db_path": self.vector_db_path,
        }


class RagIngestionService:
    def __init__(self) -> None:
        self._vector_db_path = Path(VECTOR_DB_PATH)
        self._vector_db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._vector_db_path))
        self._collection = self._client.get_or_create_collection(name=RAG_COLLECTION_NAME)
        self._embedder: SentenceTransformer | None = None

    def ingest_pdf(self, document_name: str, pdf_bytes: bytes) -> dict[str, Any]:
        text = self._extract_pdf_text(pdf_bytes)
        chunks = self._split_text(text)

        if not chunks:
            raise HTTPException(status_code=400, detail="The PDF does not contain readable text")

        embeddings = self._get_embedder().encode(chunks, normalize_embeddings=True)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        ids = [f"{Path(document_name).stem}-{uuid4().hex[:12]}-{index}" for index in range(len(chunks))]
        metadatas = [
            {"document_name": document_name, "chunk_index": index}
            for index in range(len(chunks))
        ]

        self._collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

        return RagIngestionResult(
            document_name=document_name,
            chunks=len(chunks),
            collection_name=RAG_COLLECTION_NAME,
            vector_db_path=str(self._vector_db_path),
        ).as_dict()

    def _extract_pdf_text(self, pdf_bytes: bytes) -> str:
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid PDF file") from exc

        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(page for page in pages if page.strip()).strip()
        return text

    def _split_text(self, text: str) -> list[str]:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            return []

        chunks: list[str] = []
        start = 0
        text_length = len(normalized_text)

        while start < text_length:
            end = min(text_length, start + RAG_CHUNK_SIZE)
            chunk = normalized_text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end == text_length:
                break

            start = max(end - RAG_CHUNK_OVERLAP, 0)

        return chunks

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(RAG_EMBEDDING_MODEL)
        return self._embedder


@lru_cache(maxsize=1)
def get_rag_ingestion_service() -> RagIngestionService:
    return RagIngestionService()