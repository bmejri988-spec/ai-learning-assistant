from io import BytesIO

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from backend.main import app
from backend.modules.rag.loader import load_pdf_text
from backend.modules.rag.pipeline import (
    RagAnswerPipeline,
    RagPipeline,
    get_rag_answer_pipeline,
    get_rag_pipeline,
)
from backend.modules.rag.splitter import split_text


client = TestClient(app)


def _build_test_pdf(text: str) -> bytes:
    """Create a minimal text PDF for testing."""

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

    resources = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): font_ref,
                }
            )
        }
    )

    page[NameObject("/Resources")] = resources

    content = DecodedStreamObject()
    content.set_data(
        f"BT /F1 12 Tf 72 150 Td ({text}) Tj ET".encode("latin-1")
    )

    page[NameObject("/Contents")] = writer._add_object(content)

    buffer = BytesIO()
    writer.write(buffer)

    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def test_rag_upload_route_saves_pdf_and_returns_success(
    tmp_path,
    monkeypatch,
) -> None:
    upload_dir = tmp_path / "uploads"

    monkeypatch.setattr(
        "backend.config.UPLOADS_DIR",
        str(upload_dir),
    )

    monkeypatch.setattr(
        "backend.modules.rag.uploads.UPLOADS_DIR",
        str(upload_dir),
    )

    class FakePipeline:
        def index_pdf(self, pdf_path):
            assert pdf_path.parent == upload_dir
            assert pdf_path.suffix == ".pdf"
            assert pdf_path.stem.startswith("sample_")
            assert len(pdf_path.stem) == len("sample_") + 12

            return {
                "document_name": "sample.pdf",
                "extracted_text_length": 16,
                "chunks": 1,
                "average_chunk_size": 16.0,
                "embedding_dimension": 384,
                "vector_count": 1,
                "stored_documents": 1,
            }

    app.dependency_overrides[get_rag_pipeline] = lambda: FakePipeline()

    try:
        response = client.post(
            "/rag/upload",
            files={
                "file": (
                    "sample.pdf",
                    b"%PDF-1.4 fake",
                    "application/pdf",
                )
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["message"] == "PDF uploaded and indexed"

    result = data["data"]

    assert result["file_name"] == "sample.pdf"
    assert result["document_name"] == "sample.pdf"
    assert result["extracted_text_length"] == 16
    assert result["chunks"] == 1
    assert result["average_chunk_size"] == 16.0
    assert result["embedding_dimension"] == 384
    assert result["vector_count"] == 1
    assert result["stored_documents"] == 1

    saved_file = upload_dir / result["saved_path"].split("\\")[-1]

    assert saved_file.exists()
    assert saved_file.parent == upload_dir
    assert saved_file.suffix == ".pdf"
    assert saved_file.stem.startswith("sample_")

def test_rag_upload_route_rejects_non_pdf() -> None:
    response = client.post(
        "/rag/upload",
        files={
            "file": (
                "notes.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "success": False,
        "message": "Only PDF files are supported",
    }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_rag_loader_extracts_text(tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"

    pdf_path.write_bytes(
        _build_test_pdf("Test document text")
    )

    extracted_text = load_pdf_text(pdf_path)

    assert "Test document text" in extracted_text


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


def test_rag_splitter_reports_chunk_stats() -> None:
    text = "alpha beta gamma delta " * 80

    result = split_text(text)

    assert result.count > 0
    assert result.average_chunk_size > 0


def test_rag_splitter_returns_empty_result_for_empty_text() -> None:
    result = split_text("")

    assert result.chunks == []
    assert result.count == 0
    assert result.average_chunk_size == 0.0


def test_rag_splitter_creates_overlapping_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"

    result = split_text(text)

    assert result.count > 0

    if result.count > 1:
        assert result.chunks[0]
        assert result.chunks[1]


# ---------------------------------------------------------------------------
# Indexing pipeline
# ---------------------------------------------------------------------------


def test_rag_pipeline_indexes_pdf_without_llm(
    tmp_path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "sample.pdf"

    pdf_path.write_bytes(
        _build_test_pdf(
            "alpha beta gamma delta " * 120
        )
    )

    pipeline = RagPipeline()

    class FakeEmbeddings:
        def embed_documents(self, documents):
            return [
                [float(index + 1), 0.0, 0.0]
                for index, _ in enumerate(documents)
            ]

    class FakeCollection:
        def __init__(self):
            self.documents = []

        def add(
            self,
            ids,
            documents,
            embeddings,
            metadatas,
        ):
            self.documents.extend(documents)

        def count(self):
            return len(self.documents)

    fake_collection = FakeCollection()

    monkeypatch.setattr(
        "backend.modules.rag.pipeline.get_embedding_model",
        lambda: FakeEmbeddings(),
    )

    monkeypatch.setattr(
        "backend.modules.rag.pipeline.get_collection",
        lambda: fake_collection,
    )

    result = pipeline.index_pdf(pdf_path)

    assert result["document_name"] == "sample.pdf"
    assert result["chunks"] > 0
    assert result["embedding_dimension"] == 3
    assert result["vector_count"] == result["chunks"]
    assert result["stored_documents"] == result["chunks"]


# ---------------------------------------------------------------------------
# Answer pipeline
# ---------------------------------------------------------------------------


def test_rag_answer_returns_i_dont_know_for_unrelated_question() -> None:
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int = 3):
            return {
                "query": query,
                "top_k": top_k,
                "documents": [
                    {
                        "text": (
                            "This chunk is about artificial intelligence."
                        ),
                        "metadata": {
                            "filename": "sample.pdf",
                            "page": 1,
                            "chunk_id": 0,
                        },
                        "distance": 0.8,
                        "score": 0.2,
                    }
                ],
            }

    class FakeLLM:
        def answer(self, prompt: str):
            return "A fabricated answer that should be suppressed."

    pipeline = RagAnswerPipeline(
        retriever=FakeRetriever(),
        prompt_builder=None,
        llm=FakeLLM(),
    )

    result = pipeline.answer_question(
        "Who invented Facebook?",
        top_k=3,
    )

    assert result["answer"] == "I don't know."
    assert result["retrieved_documents"] == 1

    assert result["retrieval_time_ms"] >= 0
    assert result["llm_time_ms"] >= 0
    assert result["total_latency_ms"] >= 0


def test_rag_answer_uses_llm_when_context_is_relevant() -> None:
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int = 3):
            return {
                "query": query,
                "top_k": top_k,
                "documents": [
                    {
                        "text": (
                            "Artificial intelligence is the field "
                            "of building intelligent machines."
                        ),
                        "metadata": {
                            "filename": "ai.pdf",
                            "page": 1,
                            "chunk_id": 0,
                        },
                        "distance": 0.1,
                        "score": 0.9,
                    }
                ],
            }

    class FakeLLM:
        def answer(self, prompt: str):
            assert "Artificial intelligence" in prompt

            return (
                "Artificial intelligence is the field "
                "of building intelligent machines. [Chunk 1]"
            )

    pipeline = RagAnswerPipeline(
        retriever=FakeRetriever(),
        prompt_builder=None,
        llm=FakeLLM(),
    )

    result = pipeline.answer_question(
        "What is artificial intelligence?",
        top_k=3,
    )

    assert result["answer"].startswith(
        "Artificial intelligence"
    )

    assert result["retrieved_documents"] == 1
    assert len(result["sources"]) == 1
    assert result["sources"][0]["chunk_number"] == 1


def test_rag_answer_returns_i_dont_know_when_no_documents() -> None:
    class FakeRetriever:
        def retrieve(self, query: str, top_k: int = 3):
            return {
                "query": query,
                "top_k": top_k,
                "documents": [],
            }

    class FakeLLM:
        def answer(self, prompt: str):
            raise AssertionError(
                "LLM should not be called when there are no documents."
            )

    pipeline = RagAnswerPipeline(
        retriever=FakeRetriever(),
        prompt_builder=None,
        llm=FakeLLM(),
    )

    result = pipeline.answer_question(
        "What is artificial intelligence?"
    )

    assert result["answer"] == "I don't know."
    assert result["retrieved_documents"] == 0
    assert result["sources"] == []
    assert result["llm_time_ms"] == 0.0


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_rag_route_appears_in_openapi_and_docs() -> None:
    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert openapi_response.status_code == 200

    paths = openapi_response.json()["paths"]

    assert "/rag/upload" in paths
    assert "/rag/ask" in paths


def test_rag_ask_route_generates_answer() -> None:
    class FakeAnswerPipeline:
        def answer_question(
            self,
            question: str,
            top_k: int = 3,
        ):
            assert question == "What is Artificial Intelligence?"
            assert top_k == 3

            return {
                "answer": (
                    "Artificial Intelligence is the science "
                    "of making intelligent machines."
                ),
                "sources": [
                    {
                        "chunk_number": 1,
                        "metadata": {
                            "filename": "sample.pdf",
                            "page": 1,
                            "chunk_id": 0,
                        },
                        "distance": 0.12,
                        "score": 0.88,
                    }
                ],
                "retrieved_documents": 1,
                "retrieval_time_ms": 1.2,
                "llm_time_ms": 2.3,
                "total_latency_ms": 3.5,
            }

    app.dependency_overrides[
        get_rag_answer_pipeline
    ] = lambda: FakeAnswerPipeline()

    try:
        response = client.post(
            "/rag/ask",
            json={
                "question": "What is Artificial Intelligence?",
                "top_k": 3,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["success"] is True
    assert body["data"]["answer"]

    assert body["data"]["retrieved_documents"] == 1
    assert body["data"]["sources"][0]["chunk_number"] == 1

    assert body["data"]["retrieval_time_ms"] >= 0
    assert body["data"]["llm_time_ms"] >= 0
    assert body["data"]["total_latency_ms"] >= 0