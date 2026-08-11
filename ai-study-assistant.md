- Backend runs with `uv run uvicorn backend.app.main:app` for compatibility and `backend.main` as the modular entrypoint.
- Shared API helpers live in `backend/api/utils/`.
- Tests use `pytest` and `fastapi.testclient` with `httpx2`.
- `uv sync` is the install command for this repo.
- Standard API response shape: `{success, message, data}`.
- Global error handler returns standardized JSON for 404/validation/500.

- RAG is split into `backend/modules/rag/loader.py`, `splitter.py`, `embeddings.py`, `vectordb.py`, and `pipeline.py`.
- `backend/api/routes/rag.py` exposes `POST /rag/upload` for save-only PDF uploads into `data/uploads/`.
- RAG tests cover upload validation, PDF text extraction, chunk stats, embedding storage, and OpenAPI exposure.

- Retrieval phase adds `backend/modules/rag/retriever.py` and `POST /rag/retrieve`.
- Real PDFs are uploaded through the API docs into `data/uploads/`; indexed chunks live in `data/vector_store/`.
- GitHub issue #1 tracks the LangChain deprecation follow-up after RAG completes.
- Retrieval tests cover similarity search wiring and request validation.

- Retrieval study on `artificial_intelligence_tutorial.pdf` found `500` chunk size to be the best balance among 300/500/800.
- Real retrieval latency stayed around 10-18 ms locally after indexing.
- Synonym queries like `What is AI?` still retrieved the definition chunk reliably.

- Answer generation uses Ollama with `llama3.2:3b` via `backend/modules/rag/llm.py`.
- Prompt construction is isolated in `backend/modules/rag/prompt.py`.
- `POST /rag/ask` now returns answer plus sources from the retrieved chunks.

- Hallucination guard returns `I don't know.` for unrelated questions while still returning sources.
- Chunks now store `filename`, `page`, and `chunk_id` metadata for multi-document support.
- `/rag/ask` reports retrieval, LLM, and total latency timings.

- The agent layer now lives under `backend/modules/agent/` and uses the frozen RAG API over HTTP via `RAG_API_BASE_URL`.
- `POST /agent/chat` selects a deterministic tool (`search_docs`, `summarize`, `quiz`, or `flashcards`) and is tested separately from the RAG internals.