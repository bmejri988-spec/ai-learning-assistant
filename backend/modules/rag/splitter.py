from __future__ import annotations

from dataclasses import dataclass
import re

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


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph boundaries.
    """
    paragraphs = []

    for paragraph in re.split(r"\n\s*\n", text):
        normalized = " ".join(paragraph.split())

        if normalized:
            paragraphs.append(normalized)

    return "\n\n".join(paragraphs)


def _split_long_text(text: str) -> list[str]:
    """
    Split text that is larger than the configured chunk size.

    Prefer sentence boundaries before falling back to word boundaries.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > RAG_CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = ""

            words = sentence.split()
            word_chunk = ""

            for word in words:
                candidate = (
                    f"{word_chunk} {word}".strip()
                    if word_chunk
                    else word
                )

                if len(candidate) > RAG_CHUNK_SIZE:
                    if word_chunk:
                        chunks.append(word_chunk.strip())

                    word_chunk = word
                else:
                    word_chunk = candidate

            if word_chunk:
                current = word_chunk

            continue

        candidate = (
            f"{current} {sentence}".strip()
            if current
            else sentence
        )

        if len(candidate) <= RAG_CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())

            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


def _add_overlap(chunks: list[str]) -> list[str]:
    """
    Add a small overlap between chunks without duplicating entire chunks.
    """
    if not chunks or RAG_CHUNK_OVERLAP <= 0:
        return chunks

    overlapped: list[str] = [chunks[0]]

    for index in range(1, len(chunks)):
        previous = chunks[index - 1]

        overlap = previous[-RAG_CHUNK_OVERLAP:].strip()

        if overlap:
            overlapped.append(
                f"{overlap} {chunks[index]}".strip()
            )
        else:
            overlapped.append(chunks[index])

    return overlapped


def split_text(text: str) -> SplitResult:
    normalized_text = _normalize_text(text)

    if not normalized_text:
        return SplitResult(chunks=[])

    paragraphs = normalized_text.split("\n\n")

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > RAG_CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = ""

            chunks.extend(_split_long_text(paragraph))
            continue

        candidate = (
            f"{current}\n\n{paragraph}".strip()
            if current
            else paragraph
        )

        if len(candidate) <= RAG_CHUNK_SIZE:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())

            current = paragraph

    if current:
        chunks.append(current.strip())

    chunks = _add_overlap(chunks)

    return SplitResult(chunks=chunks)