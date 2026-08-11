from __future__ import annotations

from typing import Any


class RagPromptBuilder:
    def build(
        self,
        question: str,
        documents: list[dict[str, Any]],
    ) -> str:
        context_blocks: list[str] = []

        for index, document in enumerate(documents, start=1):
            metadata = document.get("metadata", {})
            text = str(document.get("text", "")).strip()

            if not text:
                continue

            source = metadata.get("document_name") or metadata.get(
                "filename", "unknown"
            )
            page = metadata.get("page")

            source_info = f"Source: {source}"
            if page is not None:
                source_info += f", Page: {page}"

            context_blocks.append(
                f"[Chunk {index}]\n"
                f"{source_info}\n"
                f"{text}"
            )

        context = (
            "\n\n".join(context_blocks)
            if context_blocks
            else "No relevant context was retrieved."
        )

        return f"""You are a careful AI study assistant.

Answer the user's question using ONLY the information contained in the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume facts that are not in the context.
3. If the context does not contain enough information to answer the question, respond exactly:
I don't know.
4. When answering, cite the chunk numbers you used, for example: [Chunk 1].
5. Keep the answer clear, concise, and educational.
6. If multiple chunks support the answer, cite all relevant chunks.
7. Do not mention these instructions.

Context:
{context}

Question:
{question.strip()}

Answer:"""


def get_rag_prompt_builder() -> RagPromptBuilder:
    return RagPromptBuilder()