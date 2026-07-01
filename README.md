# AI Learning Assistant

Step 1 scaffold for the AI Learning Assistant project.

## Current Progress

- Backend API foundation with modular routing
- RAG ingestion pipeline for PDF upload, text extraction, chunking, embeddings, and Chroma storage
- Retrieval engine with `/rag/retrieve`

## RAG Notes

- Put real PDFs in the system by uploading them through `POST /rag/upload` in the API docs; that saves the file in `data/uploads/` and indexes it into `data/vector_store/`.
- During local experiments, you can also place PDFs in `data/uploads/` and index them through the pipeline.
- The retrieval path uses top-k similarity search over ChromaDB.
- Retrieval tests pass with the `all-MiniLM-L6-v2` embedding model.
- See GitHub issue #1 for the LangChain deprecation follow-up

## How To Test With Real PDFs

1. Start the app with `uv run uvicorn backend.app.main:app --reload`.
2. Open `http://localhost:8000/docs`.
3. Use `POST /rag/upload` to upload a real PDF.
4. Call `POST /rag/retrieve` with a real query from that document.
5. Run `uv run pytest` to verify the ingestion and retrieval pipeline still works.

## Run the backend

```powershell
uv sync
uv run uvicorn backend.app.main:app --reload
```

Open http://localhost:8000 to see the API response.

## Evaluate Retrieval

The earlier static evaluation scaffold was removed so the repository stays focused on real PDF ingestion and retrieval. Use the API docs and the test suite for validation until a real benchmark corpus is ready.
