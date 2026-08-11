import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure pytest can import the workspace package without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.main import app
from backend.modules.rag.pipeline import get_rag_answer_pipeline


client = TestClient(app)


def test_health_endpoint_returns_standard_response() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "healthy",
        "data": {"status": "healthy"},
    }


def test_placeholder_routes_return_standard_response() -> None:
    class FakeAnswerPipeline:
        def answer_question(self, question: str, top_k: int = 3):
            return {
                "answer": "AI is the science of making intelligent machines.",
                "sources": [{"chunk_number": 1, "metadata": {}, "distance": 0.1, "score": 0.9}],
                "retrieved_documents": 1,
            }

    app.dependency_overrides[get_rag_answer_pipeline] = lambda: FakeAnswerPipeline()

    response = client.post("/rag/ask", json={"question": "What is AI?", "top_k": 3})

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["success"] is True

    for path in ["/image/predict", "/text/predict", "/ml/predict"]:
        response = client.get(path)

        assert response.status_code == 200
        assert response.json() == {"success": True, "message": "Coming soon", "data": None}


def test_invalid_route_returns_404_with_standard_error() -> None:
    response = client.get("/missing-route")

    assert response.status_code == 404
    assert response.json() == {"success": False, "message": "Not Found"}