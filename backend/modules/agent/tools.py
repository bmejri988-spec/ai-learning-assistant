from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib import error, request

from fastapi import HTTPException

from backend.config import RAG_API_BASE_URL
from backend.modules.agent.schemas import ToolSpec
from backend.modules.rag.llm import get_rag_llm

if TYPE_CHECKING:
    from backend.modules.agent.memory import ConversationTurn

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "what",
    "when",
    "where",
    "how",
    "why",
    "which",
    "about",
    "your",
    "have",
    "will",
    "been",
    "their",
    "there",
    "into",
    "only",
    "using",
    "used",
    "use",
    "are",
    "was",
    "were",
    "can",
    "could",
    "should",
    "would",
    "than",
    "then",
    "also",
    "not",
    "does",
    "did",
    "has",
    "had",
    "but",
    "you",
    "our",
    "they",
    "them",
    "his",
    "her",
    "its",
    "who",
    "whom",
    "is",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "as",
    "or",
    "it",
    "be",
    "do",
    "me",
    "my",
}


class RagApiClient:
    def __init__(self, base_url: str | None = None, opener=None) -> None:
        self._base_url = (base_url or RAG_API_BASE_URL).rstrip("/")
        self._opener = opener or request.urlopen

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        payload = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
        http_request = request.Request(
            f"{self._base_url}/rag/retrieve",
            data=payload,
            headers={"Content-Type": "application/json", "accept": "application/json"},
            method="POST",
        )

        try:
            with self._opener(http_request, timeout=60) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"RAG API is unavailable at {self._base_url}. Start the backend before using the agent.",
            ) from exc

        if not response_payload.get("success", False):
            raise HTTPException(status_code=502, detail="RAG API returned an error")

        data = response_payload.get("data", {})
        return list(data.get("documents", []))

    def ask(self, question: str, top_k: int = 3) -> dict[str, Any]:
        payload = json.dumps({"question": question, "top_k": top_k}).encode("utf-8")
        http_request = request.Request(
            f"{self._base_url}/rag/ask",
            data=payload,
            headers={"Content-Type": "application/json", "accept": "application/json"},
            method="POST",
        )

        try:
            with self._opener(http_request, timeout=60) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"RAG API is unavailable at {self._base_url}. Start the backend before using the agent.",
            ) from exc

        if not response_payload.get("success", False):
            raise HTTPException(status_code=502, detail="RAG API returned an error")

        return dict(response_payload.get("data", {}))


@lru_cache(maxsize=1)
def get_rag_api_client() -> RagApiClient:
    return RagApiClient()


class AgentTool(ABC):
    spec: ToolSpec

    @abstractmethod
    def run(self, message: str, top_k: int = 3, history: list[ConversationTurn] | None = None) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(slots=True)
class SearchDocumentsTool(AgentTool):
    rag_client: RagApiClient
    spec: ToolSpec = ToolSpec(
        name="search_docs",
        description="Answer a question using the RAG answer API.",
        input_schema='{"message": "string", "top_k": 3}',
        output_schema='{"answer": "string", "sources": [], "retrieved_documents": 0}',
    )

    def run(self, message: str, top_k: int = 3, history: list[ConversationTurn] | None = None) -> dict[str, Any]:
        return self.rag_client.ask(message, top_k=top_k)


@dataclass(slots=True)
class SummarizeTool(AgentTool):
    rag_client: RagApiClient
    spec: ToolSpec = ToolSpec(
        name="summarize",
        description="Summarize retrieved document chunks into a short study note.",
        input_schema='{"message": "string", "top_k": 3}',
        output_schema='{"summary": "string", "sources": []}',
    )

    def run(self, message: str, top_k: int = 3, history: list[ConversationTurn] | None = None) -> dict[str, Any]:
        documents = self.rag_client.retrieve(message, top_k=top_k)
        if not documents:
            return {"summary": "I don't know.", "sources": [], "retrieved_documents": 0}

        combined_text = self._combine_text(documents)
        summary = self._generate_summary_with_llm(message, combined_text)

        return {
            "summary": summary,
            "sources": self._build_sources(documents),
            "retrieved_documents": len(documents),
        }

    def _combine_text(self, documents: list[dict[str, Any]]) -> str:
        return " ".join(document.get("text", "") for document in documents if document.get("text"))

    def _generate_summary_with_llm(self, message: str, text: str) -> str:
        llm = get_rag_llm()
        
        prompt = f"""Create a concise summary of the following text based on the user's request.

User request: {message}

Text:
{text}

Provide a clear, well-structured summary that captures the main points."""

        try:
            response = llm.answer(prompt)
            return response.strip() if response else self._fallback_summary(text)
        except Exception:
            return self._fallback_summary(text)

    def _fallback_summary(self, text: str) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return "I don't know."
        selected = sentences[:3]
        return " ".join(selected)

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _build_sources(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_number": index,
                "metadata": document.get("metadata", {}),
                "distance": document.get("distance"),
                "score": document.get("score"),
            }
            for index, document in enumerate(documents, start=1)
        ]


@dataclass(slots=True)
class QuizTool(AgentTool):
    rag_client: RagApiClient
    spec: ToolSpec = ToolSpec(
        name="quiz",
        description="Create multiple-choice questions from retrieved notes.",
        input_schema='{"message": "string", "top_k": 3}',
        output_schema='{"quiz": [{"question": "string", "choices": [], "answer": "string"}]}',
    )

    def run(self, message: str, top_k: int = 3, history: list[ConversationTurn] | None = None) -> dict[str, Any]:
        documents = self.rag_client.retrieve(message, top_k=top_k)
        if not documents:
            return {"quiz": [], "sources": [], "retrieved_documents": 0}

        combined_text = self._combine_text(documents)
        quiz = self._generate_quiz_with_llm(message, combined_text)

        return {
            "quiz": quiz,
            "sources": self._build_sources(documents),
            "retrieved_documents": len(documents),
        }

    def _combine_text(self, documents: list[dict[str, Any]]) -> str:
        return " ".join(document.get("text", "") for document in documents if document.get("text"))

    def _generate_quiz_with_llm(self, message: str, text: str) -> list[dict[str, Any]]:
        llm = get_rag_llm()
        
        prompt = f"""Create 3 multiple-choice questions based on the following text.

User request: {message}

Text:
{text}

Return ONLY a JSON array with this format:
[
  {{
    "question": "question text",
    "choices": ["option1", "option2", "option3", "option4"],
    "answer": "correct option"
  }}
]

Make questions challenging but fair based on the provided text."""

        try:
            response = llm.answer(prompt)
            return self._parse_quiz_response(response)
        except Exception:
            return self._fallback_quiz(text)

    def _parse_quiz_response(self, response: str) -> list[dict[str, Any]]:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return []
        
        try:
            quiz_data = json.loads(json_match.group())
            if not isinstance(quiz_data, list):
                return []
            
            validated_quiz = []
            for item in quiz_data[:3]:
                if isinstance(item, dict) and "question" in item and "choices" in item and "answer" in item:
                    validated_quiz.append({
                        "question": item["question"],
                        "choices": item["choices"],
                        "answer": item["answer"]
                    })
            
            return validated_quiz
        except json.JSONDecodeError:
            return []

    def _fallback_quiz(self, text: str) -> list[dict[str, Any]]:
        topics = self._extract_topics_from_text(text)
        quiz = []
        for index in range(min(3, len(topics))):
            correct = topics[index]
            distractors = self._distractors(topics, correct)
            quiz.append(
                {
                    "question": f"Which term best matches the retrieved notes for question {index + 1}?",
                    "choices": [correct, *distractors],
                    "answer": correct,
                }
            )
        return quiz

    def _extract_topics_from_text(self, text: str) -> list[str]:
        words = []
        for token in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", text):
            lowered = token.lower()
            if lowered not in STOPWORDS and lowered not in words:
                words.append(lowered)
        if not words:
            words = ["concept", "topic", "idea"]
        return words[:6]

    def _distractors(self, topics: list[str], correct: str) -> list[str]:
        options = [topic for topic in topics if topic != correct]
        while len(options) < 3:
            options.append(f"distractor_{len(options) + 1}")
        return options[:3]

    def _build_sources(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_number": index,
                "metadata": document.get("metadata", {}),
                "distance": document.get("distance"),
                "score": document.get("score"),
            }
            for index, document in enumerate(documents, start=1)
        ]


@dataclass(slots=True)
class FlashcardTool(AgentTool):
    rag_client: RagApiClient
    spec: ToolSpec = ToolSpec(
        name="flashcards",
        description="Create question-answer flashcards from retrieved notes.",
        input_schema='{"message": "string", "top_k": 3}',
        output_schema='{"flashcards": [{"front": "string", "back": "string"}]}',
    )

    def run(self, message: str, top_k: int = 3, history: list[ConversationTurn] | None = None) -> dict[str, Any]:
        documents = self.rag_client.retrieve(message, top_k=top_k)
        if not documents:
            return {"flashcards": [], "sources": [], "retrieved_documents": 0}

        combined_text = self._combine_text(documents)
        flashcards = self._generate_flashcards_with_llm(message, combined_text)

        return {
            "flashcards": flashcards,
            "sources": self._build_sources(documents),
            "retrieved_documents": len(documents),
        }

    def _combine_text(self, documents: list[dict[str, Any]]) -> str:
        return " ".join(document.get("text", "") for document in documents if document.get("text"))

    def _generate_flashcards_with_llm(self, message: str, text: str) -> list[dict[str, Any]]:
        llm = get_rag_llm()
        
        prompt = f"""Create 5 high-quality question-answer flashcards based on the following text.

User request: {message}

Text:
{text}

Return ONLY a JSON array with this format:
[
  {{
    "front": "question",
    "back": "answer"
  }}
]

Make questions clear and answers concise but comprehensive based on the provided text."""

        try:
            response = llm.answer(prompt)
            return self._parse_flashcard_response(response)
        except Exception:
            return self._fallback_flashcards_from_text(text)

    def _parse_flashcard_response(self, response: str) -> list[dict[str, Any]]:
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return []
        
        try:
            flashcard_data = json.loads(json_match.group())
            if not isinstance(flashcard_data, list):
                return []
            
            validated_flashcards = []
            for item in flashcard_data[:5]:
                if isinstance(item, dict) and "front" in item and "back" in item:
                    validated_flashcards.append({
                        "front": item["front"],
                        "back": item["back"]
                    })
            
            return validated_flashcards
        except json.JSONDecodeError:
            return []

    def _fallback_flashcards_from_text(self, text: str) -> list[dict[str, Any]]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        flashcards = []
        for index, sentence in enumerate(sentences[:5], start=1):
            if not sentence.strip():
                continue
            flashcards.append(
                {
                    "front": f"What is the key idea in point {index}?",
                    "back": sentence.strip(),
                }
            )
        return flashcards

    def _first_sentence(self, text: str) -> str:
        sentence = re.split(r"(?<=[.!?])\s+", text.strip())[0:1]
        return sentence[0].strip() if sentence else text.strip()

    def _build_sources(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_number": index,
                "metadata": document.get("metadata", {}),
                "distance": document.get("distance"),
                "score": document.get("score"),
            }
            for index, document in enumerate(documents, start=1)
        ]


def build_default_tools() -> dict[str, AgentTool]:
    rag_client = get_rag_api_client()
    return {
        "search_docs": SearchDocumentsTool(rag_client),
        "summarize": SummarizeTool(rag_client),
        "quiz": QuizTool(rag_client),
        "flashcards": FlashcardTool(rag_client),
    }
