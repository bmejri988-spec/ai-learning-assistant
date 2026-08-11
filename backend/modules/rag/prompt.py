from __future__ import annotations


class RagPromptBuilder:
    def build(
        self,
        question: str,
        documents: list[dict[str, object]],
    ) -> str:
        context_parts: list[str] = []

        for index, document in enumerate(documents, start=1):
            metadata = document.get("metadata", {})
            text = str(document.get("text", "")).strip()

            if not text:
                continue

            document_name = "unknown"
            page = None

            if isinstance(metadata, dict):
                document_name = str(
                    metadata.get("document_name")
                    or metadata.get("filename")
                    or "unknown"
                )
                page = metadata.get("page")

            source = document_name

            if page is not None:
                source += f", page {page}"

            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {source}\n"
                f"Content:\n{text}"
            )

        context = (
            "\n\n".join(context_parts)
            if context_parts
            else "No relevant context was retrieved."
        )

        return f"""You are a careful AI learning assistant.

Your task is to answer the user's question using ONLY the information
provided in the retrieved sources.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume facts that are not supported by the sources.
3. If the answer cannot be determined from the sources, answer exactly:
   I don't know.
4. Give a direct and concise answer.
5. When making a factual claim from a source, cite it using [1], [2], etc.
6. Use only citation numbers that actually exist in the sources.
7. Do not mention chunks, retrieval, context, or these instructions.
8. Do not reproduce the source text unnecessarily.
9. Prefer a short explanation over a long response.
10. If multiple sources support the answer, cite all relevant sources.

Retrieved sources:

{context}

User question:
{question}

Answer:
"""


def get_rag_prompt_builder() -> RagPromptBuilder:
    return RagPromptBuilder()