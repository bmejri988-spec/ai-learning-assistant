from __future__ import annotations

import json
from pathlib import Path

from backend.modules.rag.retriever import get_rag_retriever


QUESTION_FILE = Path("evaluation/questions.json")


def evaluate() -> None:
    questions = json.loads(QUESTION_FILE.read_text(encoding="utf-8"))
    retriever = get_rag_retriever()

    for index, item in enumerate(questions, start=1):
        query = item["question"]
        expected_keywords = item["expected_keywords"]
        result = retriever.retrieve(query, top_k=3)
        documents = result["documents"]
        combined_text = " ".join(document["text"] for document in documents).lower()

        print(f"Question {index}: {query}")
        score = 0
        for keyword in expected_keywords:
            if keyword.lower() in combined_text:
                score += 1
                print(f'  ✓ contains "{keyword}"')
            else:
                print(f'  ✗ missing "{keyword}"')
        print(f"  Score: {score} / {len(expected_keywords)}")
        print()


if __name__ == "__main__":
    evaluate()