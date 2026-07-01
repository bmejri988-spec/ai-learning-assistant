from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_retrieve_route_returns_documents(tmp_path, monkeypatch) -> None:
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int = 3):
            assert query == "What is machine learning?"
            assert top_k == 3
            return {
                "query": query,
                "top_k": top_k,
                "documents": [
                    {"text": "machine learning uses data", "metadata": {"chunk_index": 0}, "distance": 0.1, "score": 0.9}
                ],
            }

    from backend.modules.rag.retriever import get_rag_retriever

    app.dependency_overrides[get_rag_retriever] = lambda: FakeRetriever()

    response = client.post(
        "/rag/retrieve",
        json={"query": "What is machine learning?", "top_k": 3},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["data"]["documents"][0]["text"] == "machine learning uses data"


def test_retrieve_route_rejects_empty_query() -> None:
    response = client.post("/rag/retrieve", json={"query": "", "top_k": 3})

    assert response.status_code == 422
