# AI Learning Assistant

Step 1 scaffold for the AI Learning Assistant project.

## Current Progress

- Backend API foundation with modular routing
- RAG ingestion pipeline for PDF upload, text extraction, chunking, embeddings, and Chroma storage
- Retrieval engine with `/rag/retrieve`
- Answer generation with Ollama `llama3.2:3b`
- Deterministic agent layer with `/agent/chat` that routes through RAG as a tool
- Retrieval study with real PDF measurements in [docs/rag-retrieval-study.md](docs/rag-retrieval-study.md)

## RAG Notes

- Put real PDFs in the system by uploading them through `POST /rag/upload` in the API docs; that saves the file in `data/uploads/` and indexes it into `data/vector_store/`.
- Multiple PDFs are supported in the same Chroma collection.
- Metadata includes `filename`, `page`, and `chunk_id` so answers can be traced back to a source page.
- The retrieval path uses top-k similarity search over ChromaDB.
- Retrieval tests pass with the `all-MiniLM-L6-v2` embedding model.
- See GitHub issue #1 for the LangChain deprecation follow-up

## Example Response

```json
{
  "success": true,
  "message": "Answer generated",
  "data": {
    "answer": "...",
    "sources": [
      {
        "chunk_number": 1,
        "metadata": {
          "filename": "artificial_intelligence_tutorial.pdf",
          "page": 24,
          "chunk_id": 0
        },
        "distance": 0.31,
        "score": 0.69
      }
    ],
    "retrieved_documents": 3,
    "retrieval_time_ms": 12.5,
    "llm_time_ms": 840.2,
    "total_latency_ms": 852.7
  }
}
```

## Current Limitations

- The app still depends on a local Ollama server for answer generation.
- The agent layer uses the RAG API as a separate HTTP dependency, so the backend must be running before `/agent/chat` can be used.
- LangChain emits a deprecation warning that is tracked in GitHub issue #1.
- Scanned PDFs without selectable text will not index well without OCR.

## How To Test With Real PDFs

1. Start the app with `uv run uvicorn backend.app.main:app --reload`.
2. Open `http://localhost:8000/docs`.
3. Use `POST /rag/upload` to upload a real PDF.
4. Call `POST /rag/retrieve` with a real query from that document.
5. Use `POST /agent/chat` to ask for a search, summary, quiz, or flashcards.
6. Run `uv run pytest` to verify the ingestion, retrieval, and agent pipelines still work.

## Run the backend

```powershell
uv sync
uv run uvicorn backend.app.main:app --reload
```

Open http://localhost:8000 to see the API response.

## Evaluate Retrieval

The earlier static evaluation scaffold was removed so the repository stays focused on real PDF ingestion and retrieval. Use the API docs, the test suite, and the retrieval study note for validation.

## Retrieval Evaluation

The retrieval study in [docs/rag-retrieval-study.md](docs/rag-retrieval-study.md) records:

- chunk size comparisons for `300`, `500`, and `800`
- retrieval latency measurements
- synonym query behavior
- hallucination checks for unrelated questions
