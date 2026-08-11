# AI Learning Assistant

AI Learning Assistant is a modular backend for studying with documents using **Retrieval-Augmented Generation (RAG)** and a deterministic agent layer.

The project focuses on building a reliable, document-grounded learning assistant rather than relying on an LLM's general knowledge.

---

## Current Progress

### Backend

* Modular FastAPI backend with separated API, RAG, and agent layers
* Centralized application configuration
* Local development setup using `uv`
* Automated test suite with `pytest`
* Dependency injection for RAG components
* Persistent local vector storage

### RAG Pipeline

The system implements the complete document-to-answer workflow:

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Page-aware Processing
    ↓
Text Normalization & Chunking
    ↓
Embedding Generation
    ↓
ChromaDB Vector Storage
    ↓
Semantic Retrieval
    ↓
Relevance Filtering
    ↓
Grounded Prompt Construction
    ↓
Ollama LLM
    ↓
Answer + Sources + Latency Metrics
```

Current RAG components include:

* PDF ingestion with `pypdf`
* Page-aware text extraction
* Text normalization
* Overlapping text chunking
* `sentence-transformers/all-MiniLM-L6-v2` embeddings
* Persistent ChromaDB storage
* Cosine similarity retrieval
* Top-k semantic search
* Relevance scoring
* Source metadata tracking
* Grounded prompt generation
* Explicit `I don't know` behavior for insufficient context
* Source citations in generated answers
* Retrieval, LLM, and total latency measurements

### LLM

Answer generation currently uses:

```text
Ollama
└── llama3.2:3b
```

The RAG LLM is kept separate from the agent planner configuration.

### Agent

The project also includes an agent layer exposed through:

```text
POST /agent/chat
```

The agent can use the RAG system as a tool for document-based learning tasks such as:

* Search
* Questions and answers
* Summaries
* Quizzes
* Flashcards

---

# RAG Improvements & Evaluation

The RAG system was iteratively tested and improved using a real AI tutorial PDF containing more than 70 pages.

The improvements focused on:

* Retrieval quality
* Answer grounding
* Source attribution
* Chunking
* Prompt quality
* Latency
* Resistance to unsupported answers
* Automated testing

## 1. Improved Answer Quality Through Prompt Engineering

The first version of the RAG prompt allowed the model to expose internal retrieval terminology and produced a verbose answer:

```text
[Chunk 1]
According to [Chunk 1], Artificial Intelligence (AI) is a science and
technology based on disciplines such as Computer Science, Biology,
Psychology, Linguistics, Mathematics, and Engineering.

[Chunk 3]
Additionally, [Chunk 3] states that AI is a way of making a computer,
a computer-controlled robot, or a software think intelligently...
```

The prompt was then redesigned to explicitly control the output format and grounding behavior.

After the prompt improvement, the same type of question produced a significantly cleaner answer:

```text
Artificial intelligence (AI) refers to the science and technology based
on disciplines such as Computer Science, Biology, Psychology,
Linguistics, Mathematics, and Engineering [1]. It aims to create systems
that understand, think, learn, and behave like humans.
```

The answer therefore moved from an **internal chunk-oriented response** to a **concise, user-facing, cited response**.

The improved prompt instructs the LLM to:

* Use only the retrieved document context
* Avoid external knowledge
* Avoid unsupported claims
* Say `I don't know based on the provided documents.` when the context is insufficient
* Cite supporting sources
* Avoid exposing internal RAG terminology
* Provide concise answers

---

## 2. Improved Source Attribution

The initial response exposed internal retrieval terminology:

```text
According to [Chunk 1]...
[Chunk 3] states...
```

After prompt refinement, citations became user-facing:

```text
Artificial intelligence refers to a science and engineering
that makes machines think intelligently, similar to humans [1].
```

The API separately returns the source metadata:

```json
{
  "chunk_number": 1,
  "metadata": {
    "document_name": "artificial_intelligence_tutorial.pdf",
    "page": 8,
    "chunk_index": 17
  },
  "distance": 0.1928,
  "score": 0.8072
}
```

This creates a clear traceability chain:

```text
Answer
  ↓
Citation [1]
  ↓
Retrieved source
  ↓
Document + page + chunk
```

---

## 3. Retrieval Quality Improvement

During the initial evaluation, a query such as:

```text
What is AI?
```

produced the following relevance scores:

```text
0.6802
0.6430
0.6423
```

After improving the RAG pipeline and re-indexing the document, the same type of query produced:

```text
0.8072
0.7549
0.6649
```

The strongest retrieved result therefore increased from:

```text
0.6802 → 0.8072
```

This is approximately an **18.7% relative improvement in the normalized relevance score**.

The improved retrieval also identified highly relevant pages:

```text
Page 8
Page 9
Page 69
```

with the strongest result coming from page 8.

> The score is a normalized retrieval signal derived from the vector distance. It should not be interpreted as a probability or absolute accuracy measurement.

---

## 4. Improved Chunking and Page Traceability

The ingestion pipeline was changed to process PDFs page by page rather than treating the entire document as one text block.

Each chunk retains metadata such as:

```json
{
  "document_name": "artificial_intelligence_tutorial.pdf",
  "page": 8,
  "chunk_index": 17
}
```

Current chunking defaults:

```text
Chunk size:    1000 characters
Chunk overlap: 150 characters
```

The overlap helps preserve context across neighboring chunks.

Page-aware processing also makes it possible to trace retrieved information back to the original document.

---

## 5. Retrieval Latency Improvement Through Caching

The first execution was significantly slower because local models and resources had to be initialized.

One observed first execution produced approximately:

```text
Retrieval: ~21.5 seconds
```

After the embedding model was loaded and cached, subsequent retrieval requests dropped to approximately:

```text
Retrieval: ~15–20 ms
```

This is a substantial reduction in retrieval overhead.

The embedding model is cached and reused rather than being initialized for every request.

---

## 6. End-to-End Latency Improvement

An early request produced approximately:

```text
Retrieval:      21,548 ms
LLM:             2,394 ms
Total:          23,943 ms
```

After model/resource caching, a subsequent request produced:

```text
Retrieval:          16 ms
LLM:             1,907 ms
Total:           1,923 ms
```

The observed end-to-end latency therefore decreased from approximately:

```text
23.9 seconds → 1.9 seconds
```

The main improvement came from eliminating the repeated embedding-model initialization cost.

After caching, the LLM became the dominant component of the request:

```text
~1–2 seconds
```

which is expected when running `llama3.2:3b` locally through Ollama.

> These measurements are local observations and depend on hardware, model state, document size, and system caching.

---

## 7. Improved Grounding and Relevance Handling

The answer pipeline was also improved to prevent the LLM from confidently answering questions when the retrieved context is not sufficiently relevant.

The intended behavior is:

```text
Relevant context
      ↓
Grounded answer
```

and:

```text
Insufficient / irrelevant context
      ↓
I don't know based on the provided documents.
```

This is particularly important for a document-based learning assistant because the system should prioritize **document evidence over general LLM knowledge**.

---

# RAG Architecture

The current RAG architecture is:

```text
                    ┌──────────────────┐
                    │    PDF Upload    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  PDF Extraction  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │ Normalize / Split│
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    Embeddings    │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │    ChromaDB      │
                    └────────┬─────────┘
                             ↓
Question ──────────→ Semantic Retrieval
                             ↓
                    ┌──────────────────┐
                    │ Relevance Check  │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  Grounded Prompt │
                    └────────┬─────────┘
                             ↓
                    ┌──────────────────┐
                    │  Ollama / LLM    │
                    └────────┬─────────┘
                             ↓
                    Answer + Sources
```

---

# RAG Configuration

RAG configuration is centralized in:

```text
backend/config.py
```

Current defaults:

| Setting             | Value                                    |
| ------------------- | ---------------------------------------- |
| Embedding model     | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | `384`                                    |
| Chunk size          | `1000`                                   |
| Chunk overlap       | `150`                                    |
| Default top-k       | `3`                                      |
| Vector database     | ChromaDB                                 |
| Vector metric       | Cosine                                   |
| RAG LLM             | `llama3.2:3b`                            |
| Ollama URL          | `http://127.0.0.1:11434`                 |
| Vector storage      | `data/vector_store`                      |
| Uploaded PDFs       | `data/uploads`                           |

Agent-specific configuration is maintained in the same configuration module.

---

# Example RAG Response

```json
{
  "success": true,
  "message": "Answer generated",
  "data": {
    "answer": "Artificial intelligence refers to a science and engineering that makes machines think intelligently, similar to humans [1].",
    "sources": [
      {
        "chunk_number": 1,
        "metadata": {
          "chunk_index": 17,
          "document_name": "artificial_intelligence_tutorial.pdf",
          "page": 8
        },
        "distance": 0.1928,
        "score": 0.8072
      },
      {
        "chunk_number": 2,
        "metadata": {
          "chunk_index": 20,
          "document_name": "artificial_intelligence_tutorial.pdf",
          "page": 9
        },
        "distance": 0.2451,
        "score": 0.7549
      }
    ],
    "retrieved_documents": 3,
    "retrieval_time_ms": 18.64,
    "llm_time_ms": 1474.89,
    "total_latency_ms": 1493.54
  }
}
```

---

# Testing

The RAG test suite covers:

* PDF upload
* File type validation
* PDF text extraction
* Text normalization
* Text splitting
* Chunk statistics
* Embedding generation
* Vector storage
* Retrieval
* Answer generation
* Relevance handling
* Unrelated-question behavior
* API routes
* OpenAPI exposure
* Dependency injection
* Mocked RAG components

Current test status:

```text
12 passed
```

Run all tests:

```powershell
uv run pytest
```

Run only RAG tests:

```powershell
uv run pytest tests/test_rag.py
```

---

# Project Structure

```text
backend/
├── config.py
├── main.py
├── api/
│   └── routes/
│       └── rag.py
└── modules/
    └── rag/
        ├── embeddings.py
        ├── loader.py
        ├── llm.py
        ├── pipeline.py
        ├── prompt.py
        ├── retriever.py
        ├── splitter.py
        ├── uploads.py
        └── vectordb.py
```

Each module has a focused responsibility:

| Module          | Responsibility                  |
| --------------- | ------------------------------- |
| `loader.py`     | PDF text extraction             |
| `splitter.py`   | Text normalization and chunking |
| `embeddings.py` | Embedding model management      |
| `vectordb.py`   | ChromaDB collection management  |
| `retriever.py`  | Semantic retrieval              |
| `prompt.py`     | Grounded prompt construction    |
| `llm.py`        | Ollama communication            |
| `pipeline.py`   | RAG orchestration               |
| `uploads.py`    | Uploaded-file handling          |
| `rag.py`        | RAG API endpoints               |

---

# Working With Real PDFs

## 1. Start the backend

```powershell
uv run uvicorn backend.main:app --reload
```

## 2. Open the API documentation

```text
http://localhost:8000/docs
```

## 3. Upload a PDF

Use:

```text
POST /rag/upload
```

Uploaded documents are stored in:

```text
data/uploads/
```

and indexed into:

```text
data/vector_store/
```

## 4. Test retrieval

Use:

```text
POST /rag/retrieve
```

with a question related to the uploaded document.

## 5. Generate an answer

Use:

```text
POST /rag/ask
```

The response contains:

* Generated answer
* Retrieved sources
* Relevance scores
* Retrieval latency
* LLM latency
* Total latency

## 6. Test the agent

Use:

```text
POST /agent/chat
```

for document-based study tasks.

---

# Retrieval Evaluation

A retrieval study is maintained in:

```text
docs/rag-retrieval-study.md
```

The study evaluates:

* Chunk sizes
* Chunk overlap
* Retrieval latency
* Semantic queries
* Synonym queries
* Retrieval quality
* Unrelated-question behavior
* Hallucination resistance

The project prioritizes evaluation using **real PDF documents** rather than relying exclusively on synthetic benchmarks.

---

# Performance

After model and resource caching, typical local measurements observed during development are:

```text
Retrieval:     ~15–20 ms
LLM:           ~1–2 seconds
Total:         ~1–2 seconds
```

The first request can be significantly slower because the embedding model and local model resources may need to be loaded into memory.

Performance depends on:

* CPU/GPU
* Document size
* Number of retrieved chunks
* Ollama model
* Embedding model
* System caching

---

# Current Limitations

* Answer generation requires a local Ollama server.
* `llama3.2:3b` is lightweight but has lower reasoning and instruction-following capability than larger models.
* Scanned or image-only PDFs require OCR for reliable text extraction.
* PDF extraction quality depends on the structure of the source document.
* The vector store is currently local and persistent on disk.
* Multi-document retrieval is supported, but advanced document-level filtering is a future improvement.
* Hybrid retrieval combining semantic and lexical search is a potential future improvement.
* The agent currently communicates with the RAG API through HTTP, so the backend must be running when the agent uses the RAG service.
* A LangChain deprecation warning remains tracked in GitHub issue #1.

---

# Development

Install dependencies:

```powershell
uv sync
```

Run the backend:

```powershell
uv run uvicorn backend.main:app --reload
```

Run the complete test suite:

```powershell
uv run pytest
```

Run only the RAG tests:

```powershell
uv run pytest tests/test_rag.py
```

---

# Summary

The project currently provides a complete local RAG workflow:

```text
PDF
 ↓
Page-aware extraction
 ↓
Normalized overlapping chunks
 ↓
384-dimensional embeddings
 ↓
ChromaDB cosine retrieval
 ↓
Top-k semantic matches
 ↓
Relevance validation
 ↓
Grounded prompt
 ↓
llama3.2:3b
 ↓
Concise cited answer
```

The iterative RAG improvements produced measurable changes during development:

| Area                                |                          Before |                         After |
| ----------------------------------- | ------------------------------: | ----------------------------: |
| Best observed relevance score       |                        `0.6802` |                  **`0.8072`** |
| Relative score improvement          |                               — |                    **~18.7%** |
| First retrieval execution           |                       `~21.5 s` | **`~15–20 ms` after caching** |
| End-to-end first observed execution |                       `~23.9 s` |    **`~1.9 s` after caching** |
| Answer format                       | Internal `[Chunk X]` references |   **Natural `[1]` citations** |
| Source traceability                 |                         Limited |   **Document + page + chunk** |
| Prompt strategy                     |         Basic context injection | **Strict grounded prompting** |
| Automated RAG tests                 |                               — |                 **12 passed** |

The main outcome is a RAG system that is **faster after initialization, more traceable, better grounded, and significantly cleaner in its generated answers**, while remaining fully local and suitable for continued development of the AI Learning Assistant.
