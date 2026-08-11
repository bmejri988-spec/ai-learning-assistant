from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from backend.modules.rag.embeddings import get_embedding_model
from backend.modules.rag.loader import load_pdf_pages
from backend.modules.rag.llm import get_rag_llm
from backend.modules.rag.prompt import get_rag_prompt_builder
from backend.modules.rag.retriever import (
    RagRetriever,
    get_rag_retriever,
)
from backend.modules.rag.splitter import split_text
from backend.modules.rag.vectordb import get_collection


class RagPipeline:
    """Pipeline responsible for indexing PDF documents."""

    def chunk_text(self, text: str):
        return split_text(text)

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        if not chunks:
            return []

        return get_embedding_model().embed_documents(chunks)

    def store_chunks(
        self,
        document_name: str,
        page_number: int,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings"
            )

        if not chunks:
            return 0

        collection = get_collection()

        ids = [
            (
                f"{Path(document_name).stem}"
                f"-p{page_number}"
                f"-{uuid4().hex[:12]}"
                f"-{index}"
            )
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "filename": document_name,
                "page": page_number,
                "chunk_id": index,
            }
            for index in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def index_pdf(self, pdf_path: Path) -> dict[str, object]:
        pages = load_pdf_pages(pdf_path)

        if not pages:
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain readable text",
            )

        total_chunks = 0
        total_embeddings = 0
        total_text_length = 0
        embedding_dimension = 0

        for page in pages:
            page_text = page.get("page_content", "").strip()

            if not page_text:
                continue

            page_number = (
                int(page.get("metadata", {}).get("page", 0)) + 1
            )

            split_result = self.chunk_text(page_text)

            if not split_result.chunks:
                continue

            embeddings = self.embed_chunks(split_result.chunks)

            if embeddings and embedding_dimension == 0:
                embedding_dimension = len(embeddings[0])

            stored_count = self.store_chunks(
                document_name=pdf_path.name,
                page_number=page_number,
                chunks=split_result.chunks,
                embeddings=embeddings,
            )

            total_chunks += split_result.count
            total_embeddings += len(embeddings)
            total_text_length += len(page_text)

            if stored_count != len(split_result.chunks):
                raise RuntimeError(
                    "Not all chunks were stored successfully"
                )

        if total_chunks == 0:
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain readable text",
            )

        return {
            "document_name": pdf_path.name,
            "extracted_text_length": total_text_length,
            "chunks": total_chunks,
            "average_chunk_size": round(
                total_text_length / total_chunks,
                2,
            ),
            "embedding_dimension": embedding_dimension,
            "vector_count": total_embeddings,
            "stored_documents": total_chunks,
        }


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RagPipeline:
    return RagPipeline()


class RagAnswerPipeline:
    """Pipeline responsible for answering questions using RAG."""

    def __init__(
        self,
        retriever: RagRetriever | None = None,
        prompt_builder=None,
        llm=None,
    ) -> None:
        self._retriever = retriever or get_rag_retriever()
        self._prompt_builder = (
            prompt_builder or get_rag_prompt_builder()
        )
        self._llm = llm or get_rag_llm()

    def _has_relevant_context(
        self,
        documents: list[dict[str, object]],
    ) -> bool:
        if not documents:
            return False

        scores = [
            float(document.get("score", 0.0))
            for document in documents
            if document.get("score") is not None
        ]

        if not scores:
            return False

        return max(scores) >= 0.35

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, object]:
        question = question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty",
            )

        if top_k <= 0:
            raise HTTPException(
                status_code=400,
                detail="top_k must be greater than zero",
            )

        start_time = time.perf_counter()

        retrieval = self._retriever.retrieve(
            question,
            top_k,
        )

        documents = retrieval.get("documents", [])

        retrieval_seconds = time.perf_counter() - start_time

        if not documents:
            return {
                "answer": "I don't know.",
                "sources": [],
                "retrieved_documents": 0,
                "retrieval_time_ms": round(
                    retrieval_seconds * 1000,
                    2,
                ),
                "llm_time_ms": 0.0,
                "total_latency_ms": round(
                    retrieval_seconds * 1000,
                    2,
                ),
            }

        if not self._has_relevant_context(documents):
            return {
                "answer": "I don't know.",
                "sources": [],
                "retrieved_documents": len(documents),
                "retrieval_time_ms": round(
                    retrieval_seconds * 1000,
                    2,
                ),
                "llm_time_ms": 0.0,
                "total_latency_ms": round(
                    retrieval_seconds * 1000,
                    2,
                ),
            }

        prompt = self._prompt_builder.build(
            question,
            documents,
        )

        llm_start = time.perf_counter()

        answer = self._llm.answer(prompt).strip()

        llm_seconds = time.perf_counter() - llm_start

        if not answer:
            answer = "I don't know."

        sources = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            sources.append(
                {
                    "chunk_number": index,
                    "metadata": document.get(
                        "metadata",
                        {},
                    ),
                    "distance": document.get(
                        "distance",
                    ),
                    "score": document.get(
                        "score",
                    ),
                }
            )

        total_seconds = (
            retrieval_seconds + llm_seconds
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_documents": len(documents),
            "retrieval_time_ms": round(
                retrieval_seconds * 1000,
                2,
            ),
            "llm_time_ms": round(
                llm_seconds * 1000,
                2,
            ),
            "total_latency_ms": round(
                total_seconds * 1000,
                2,
            ),
        }


@lru_cache(maxsize=1)
def get_rag_answer_pipeline() -> RagAnswerPipeline:
    return RagAnswerPipeline()