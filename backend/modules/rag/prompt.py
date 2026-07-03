from __future__ import annotations


class RagPromptBuilder:
    def build(self, question: str, documents: list[dict[str, object]]) -> str:
        context_lines = []
        for index, document in enumerate(documents, start=1):
            metadata = document.get("metadata", {})
            text = document.get("text", "")
            context_lines.append(
                f"[Chunk {index}]\nSource: {metadata}\nText: {text}"
            )

        context_block = "\n\n".join(context_lines) if context_lines else "No context found."
        return (
            "You are a careful study assistant. Answer the question using only the provided context.\n"
            "If the answer is not in the context, say: I don't know.\n"
            "Cite the chunk numbers you used.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )


def get_rag_prompt_builder() -> RagPromptBuilder:
    return RagPromptBuilder()