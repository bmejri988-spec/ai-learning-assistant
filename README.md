# AI Learning Assistant

Step 1 scaffold for the AI Learning Assistant project.

## Current Progress

- Backend API foundation with modular routing
- RAG ingestion pipeline for PDF upload, text extraction, chunking, embeddings, and Chroma storage
- Retrieval engine with `/rag/retrieve`
- Evaluation dataset and `evaluate.py` for quick retrieval checks

## RAG Notes

- PDF uploads are saved in `data/uploads/`
- The retrieval path uses top-k similarity search over ChromaDB
- Current retrieval tests pass with the `all-MiniLM-L6-v2` embedding model
- See GitHub issue #1 for the LangChain deprecation follow-up

## Retrieval Experiment Snapshot

I compared chunk sizes of 300, 500, and 800 on a local test document. The first run was dominated by the initial embedding model download, while later runs completed much faster. Retrieval quality was stable on the test query, which is a useful baseline before we add generation.

## Run the backend

```powershell
uv sync
uv run uvicorn backend.app.main:app --reload
```

Open http://localhost:8000 to see the API response.

## Evaluate Retrieval

```powershell
uv run python evaluate.py
```
