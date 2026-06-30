from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.modules.rag.service import RagIngestionService


client = TestClient(app)


def test_rag_splitter_creates_chunks() -> None:
    service = RagIngestionService()
    chunks = service._split_text("This is a long sentence. " * 80)

    assert chunks
    assert all(chunk for chunk in chunks)


def test_rag_ingestion_persists_chunks(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.modules.rag.service.VECTOR_DB_PATH", str(tmp_path / "vector_store"))

    service = RagIngestionService()
    monkeypatch.setattr(service, "_extract_pdf_text", lambda _: "alpha beta gamma delta " * 120)

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(texts)]

    monkeypatch.setattr(service, "_get_embedder", lambda: FakeEmbedder())

    result = service.ingest_pdf("notes.pdf", b"%PDF-1.4 fake")

    assert result["document_name"] == "notes.pdf"
    assert result["chunks"] > 0
    assert service._collection.count() == result["chunks"]
    assert (tmp_path / "vector_store").exists()


def test_rag_upload_route_uses_ingestion_service(tmp_path, monkeypatch) -> None:
    class FakeService:
        def ingest_pdf(self, document_name: str, pdf_bytes: bytes):
            assert document_name == "sample.pdf"
            assert pdf_bytes == b"%PDF-1.4 fake"
            return {"document_name": document_name, "chunks": 2, "vector_db_path": str(tmp_path)}

    app.dependency_overrides.clear()
    from backend.modules.rag.service import get_rag_ingestion_service

    app.dependency_overrides[get_rag_ingestion_service] = lambda: FakeService()

    response = client.post(
        "/rag/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "PDF indexed",
        "data": {"document_name": "sample.pdf", "chunks": 2, "vector_db_path": str(tmp_path)},
    }


def test_rag_upload_rejects_non_pdf() -> None:
    response = client.post(
        "/rag/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"success": False, "message": "Only PDF files are supported"}


def test_rag_route_appears_in_openapi_and_docs() -> None:
    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert openapi_response.status_code == 200
    assert "/rag/upload" in openapi_response.json()["paths"]