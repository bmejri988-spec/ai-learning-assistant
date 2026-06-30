from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend.main import app
from backend.modules.rag.loader import load_pdf_text
from backend.modules.rag.pipeline import RagPipeline
from backend.modules.rag.splitter import split_text


client = TestClient(app)


def _build_test_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)

    resources = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})})
    page[NameObject("/Resources")] = resources

    content = DecodedStreamObject()
    content.set_data(f"BT /F1 12 Tf 72 150 Td ({text}) Tj ET".encode("utf-8"))
    page[NameObject("/Contents")] = writer._add_object(content)

    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_rag_upload_route_saves_pdf_and_returns_success(tmp_path, monkeypatch) -> None:
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr("backend.config.UPLOADS_DIR", str(upload_dir))
    monkeypatch.setattr("backend.modules.rag.uploads.UPLOADS_DIR", str(upload_dir))

    response = client.post(
        "/rag/upload",
        files={"file": ("sample.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    saved_file = upload_dir / "sample.pdf"

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "PDF uploaded",
        "data": {"file_name": "sample.pdf", "saved_path": str(saved_file)},
    }
    assert saved_file.exists()


def test_rag_upload_route_rejects_non_pdf() -> None:
    response = client.post(
        "/rag/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"success": False, "message": "Only PDF files are supported"}


def test_rag_loader_extracts_text(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_build_test_pdf("Test document text"))

    extracted_text = load_pdf_text(pdf_path)

    assert "Test document text" in extracted_text


def test_rag_splitter_reports_chunk_stats() -> None:
    result = split_text("alpha beta gamma delta " * 80)

    assert result.count > 0
    assert result.average_chunk_size > 0


def test_rag_pipeline_indexes_pdf_without_llm(tmp_path, monkeypatch) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(_build_test_pdf("alpha beta gamma delta " * 120))

    pipeline = RagPipeline()

    class FakeEmbeddings:
        def embed_documents(self, docs):
            return [[float(index + 1), 0.0, 0.0] for index, _ in enumerate(docs)]

    class FakeCollection:
        def __init__(self) -> None:
            self.documents = []

        def add(self, ids, documents, embeddings, metadatas):
            self.documents.extend(documents)

        def count(self):
            return len(self.documents)

    fake_collection = FakeCollection()
    monkeypatch.setattr("backend.modules.rag.pipeline.get_embedding_model", lambda: FakeEmbeddings())
    monkeypatch.setattr("backend.modules.rag.pipeline.get_collection", lambda: fake_collection)

    result = pipeline.index_pdf(pdf_path)

    assert result["document_name"] == "sample.pdf"
    assert result["chunks"] > 0
    assert result["embedding_dimension"] == 3
    assert result["stored_documents"] == result["chunks"]


def test_rag_route_appears_in_openapi_and_docs() -> None:
    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert openapi_response.status_code == 200
    assert "/rag/upload" in openapi_response.json()["paths"]
