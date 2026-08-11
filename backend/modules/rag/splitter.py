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
    """
    Split text into overlapping chunks while trying to preserve
    word boundaries.
    """

    normalized_text = " ".join(text.split())

    if not normalized_text:
        return SplitResult(chunks=[])

    if RAG_CHUNK_SIZE <= 0:
        raise ValueError("RAG_CHUNK_SIZE must be greater than 0")

    if RAG_CHUNK_OVERLAP < 0:
        raise ValueError("RAG_CHUNK_OVERLAP cannot be negative")

    if RAG_CHUNK_OVERLAP >= RAG_CHUNK_SIZE:
        raise ValueError(
            "RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE"
        )

    chunks: list[str] = []
    start = 0
    text_length = len(normalized_text)

    while start < text_length:
        end = min(start + RAG_CHUNK_SIZE, text_length)

        if end < text_length:
            boundary = normalized_text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = end - RAG_CHUNK_OVERLAP

        if next_start <= start:
            next_start = end

        start = next_start

    return SplitResult(chunks=chunks)