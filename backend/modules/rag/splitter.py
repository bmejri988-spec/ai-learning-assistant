from __future__ import annotations

from dataclasses import dataclass

from backend.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE


@dataclass(slots=True)
class SplitResult:
    chunks: list[str]

    @property
    def count(self) -> int:
        return len(self.chunks)

    @property
    def average_chunk_size(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(len(chunk) for chunk in self.chunks) / len(self.chunks)


def split_text(text: str) -> SplitResult:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return SplitResult(chunks=[])

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

    return SplitResult(chunks=chunks)