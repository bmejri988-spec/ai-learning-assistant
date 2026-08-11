from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from backend.config import RAG_MIN_RELEVANCE_SCORE
from backend.modules.rag.embeddings import get_embedding_model
from backend.modules.rag.loader import load_pdf_pages
from backend.modules.rag.llm import get_rag_llm
from backend.modules.rag.prompt import get_rag_prompt_builder
from backend.modules.rag.retriever import RagRetriever, get_rag_retriever
from backend.modules.rag.splitter import split_text
from backend.modules.rag.vectordb import get_collection


class RagPipeline:
    def extract_text(self, pdf_path: Path) -> str:
        pages = load_pdf_pages(pdf_path)
        return "\n".join(
            page["page_content"]
            for page in pages
            if page["page_content"]
        ).strip()

    def chunk_text(self, text: str):
        return split_text(text)

    def embed_chunks(self, chunks: list[str]):
        return get_embedding_model().embed_documents(chunks)

    def store_chunks(
        self,
        document_name: str,
        page_number: int,
        chunks: list[str],
        embeddings: list[list[float]],
        start_chunk_index: int = 0,
    ) -> int:
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
                "document_name": document_name,
                "page": page_number,
                "chunk_index": start_chunk_index + index,
            }
            for index in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return collection.count()

    def index_pdf(self, pdf_path: Path) -> dict[str, object]:
        pages = load_pdf_pages(pdf_path)

        readable_pages = [
            page
            for page in pages
            if page.get("page_content", "").strip()
        ]

        if not readable_pages:
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain readable text",
            )

        total_chunks = 0
        total_embeddings = 0
        total_text_length = 0
        collection_count = 0
        global_chunk_index = 0

        for page in readable_pages:
            page_text = page["page_content"]
            page_number = int(
                page.get("metadata", {}).get("page", 0)
            ) + 1

            split_result = self.chunk_text(page_text)

            if not split_result.chunks:
                continue

            embeddings = self.embed_chunks(split_result.chunks)

            collection_count = self.store_chunks(
                document_name=pdf_path.name,
                page_number=page_number,
                chunks=split_result.chunks,
                embeddings=embeddings,
                start_chunk_index=global_chunk_index,
            )

            global_chunk_index += split_result.count
            total_chunks += split_result.count
            total_embeddings += len(embeddings)
            total_text_length += len(page_text)

        if total_chunks == 0:
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain readable text",
            )

        embedding_dim = (
            len(embeddings[0])
            if total_embeddings > 0 and embeddings
            else 0
        )

        return {
            "document_name": pdf_path.name,
            "extracted_text_length": total_text_length,
            "chunks": total_chunks,
            "average_chunk_size": round(
                total_text_length / total_chunks,
                2,
            ),
            "embedding_dimension": embedding_dim,
            "vector_count": total_embeddings,
            "stored_documents": collection_count,
        }


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RagPipeline:
    return RagPipeline()


class RagAnswerPipeline:
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

    def _build_sources(
        self,
        documents: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        sources = []

        for index, document in enumerate(documents, start=1):
            sources.append(
                {
                    "chunk_number": index,
                    "metadata": document.get("metadata", {}),
                    "distance": document.get("distance"),
                    "score": document.get("score"),
                }
            )

        return sources

    def answer_question(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict[str, object]:
        start_time = time.perf_counter()

        retrieval = self._retriever.retrieve(
            question,
            top_k,
        )

        documents = retrieval.get("documents", []) or []

        retrieval_seconds = time.perf_counter() - start_time

        retrieved_documents = len(documents)
        sources = self._build_sources(documents)

        # Nothing was retrieved.
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

        # Reject weak retrieval before calling the LLM.
        scores = [
            float(document.get("score", 0.0))
            for document in documents
            if document.get("score") is not None
        ]

        best_score = max(scores, default=0.0)

        if best_score < RAG_MIN_RELEVANCE_SCORE:
            return {
                "answer": "I don't know.",
                "sources": sources,
                "retrieved_documents": retrieved_documents,
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

        # Build grounded prompt.
        prompt = self._prompt_builder.build(
            question,
            documents,
        )

        llm_start = time.perf_counter()

        answer = self._llm.answer(prompt)

        llm_seconds = time.perf_counter() - llm_start

        if not answer.strip():
            answer = "I don't know."

        return {
            "answer": answer.strip(),
            "sources": sources,
            "retrieved_documents": retrieved_documents,
            "retrieval_time_ms": round(
                retrieval_seconds * 1000,
                2,
            ),
            "llm_time_ms": round(
                llm_seconds * 1000,
                2,
            ),
            "total_latency_ms": round(
                (retrieval_seconds + llm_seconds) * 1000,
                2,
            ),
        }


@lru_cache(maxsize=1)
def get_rag_answer_pipeline() -> RagAnswerPipeline:
    return RagAnswerPipeline()